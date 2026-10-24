"""Only the approved seed-0 dense anchor; no pruning or follow-on task queue."""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from core import evaluate, sam_step
from data import Images, make_loader, real_data
from model import build_model
from state import (atomic_checkpoint,atomic_json,atomic_text,checkpoint_payload,disk_check,
                   environment,gpu_check,load_checkpoint,object_sha,seed_all,sha_file,
                   source_freeze,tensor_sha,validate_checkpoint)

ROOT=Path(__file__).resolve().parent


def optimizer_for(model,c):
    optimizer=torch.optim.SGD(model.parameters(),lr=c['learning_rate'],momentum=c['momentum'],
                              weight_decay=c['weight_decay'],nesterov=c['nesterov'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['epochs'],eta_min=c['eta_min'])
    return optimizer,scheduler


def choose_best(previous,row):
    if previous is None or row['validation_accuracy']>previous['validation_accuracy']:
        return {'epoch':row['epoch'],'validation_accuracy':row['validation_accuracy'],
                'validation_loss_at_best_accuracy':row['validation_loss']},True
    return previous,False


def save_metrics(out,history):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(history[0]))
    writer.writeheader(); writer.writerows(history)
    atomic_text(out/'metrics.csv',stream.getvalue())
    atomic_text(out/'epoch_metrics.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))


def checkpoint_index(out):
    paths=[out/n for n in ['best_val.pt','last.pt','epoch_200.pt'] if (out/n).exists()]
    atomic_text(out/'checkpoint_sha256.txt',''.join(f'{sha_file(p)}  {p.name}\n' for p in paths))


def run(c,resume=False):
    if c['purpose']!='formal' or c['seed']!=0 or c['epochs']!=200:
        raise ValueError('Only approved formal seed0/200epoch configuration allowed')
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    frozen=source_freeze(ROOT)
    if sha_file(ROOT/'config.yaml')!=frozen['source_sha256']['config.yaml']:
        raise ValueError('Configuration freeze')
    gpu=gpu_check(c)
    disk_check(ROOT,c)
    seed_all(c['seed'],c['threads'])
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major!=12:
        raise RuntimeError('Approved CUDA environment unavailable')
    train_data,val_data,split=real_data(ROOT,c)
    model=build_model().to(c['device'])
    optimizer,scheduler=optimizer_for(model,c)
    initialization_hash=tensor_sha(model.state_dict())
    acceptance=json.loads((ROOT/'artifacts/precheck.json').read_text())
    if (acceptance['status']!='PASS' or acceptance['source_commit']!=frozen['commit']
        or acceptance['initialization_hash']!=initialization_hash
        or acceptance['split_sha256']!=sha_file(ROOT/c['split_file'])
        or acceptance['config_sha256']!=object_sha(c)
        or acceptance['gpu']['uuid']!=gpu['uuid']):
        raise ValueError('G1 precheck does not match formal run identity')
    identity={'schema':2,'commit':frozen['commit'],'source_sha256':frozen['source_sha256'],
              'config':c,'config_sha256':object_sha(c),'config_file_sha256':sha_file(ROOT/'config.yaml'),
              'split_sha256':sha_file(ROOT/c['split_file']),
              'gpu':{'uuid':gpu['uuid'],'name':gpu['name']},'environment':environment(),
              'initialization_hash':initialization_hash}
    history,best,start=[],None,0
    if resume:
        restored=load_checkpoint(out/'last.pt',model,optimizer,scheduler,identity)
        history,best,start=restored['history'],restored['best'],restored['epoch']
        if start>=200:
            raise ValueError('Already completed; no automatic re-evaluation')
        saved_best=torch.load(out/'best_val.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(saved_best,identity)
        if saved_best['epoch']!=best['epoch']:
            raise ValueError('Best checkpoint and resumed last state disagree')
    elif any((out/n).exists() for n in ['last.pt','best_val.pt','epoch_200.pt','epoch_metrics.jsonl']):
        raise FileExistsError('Existing run; refusing overwrite')
    else:
        atomic_text(out/'config.yaml',(ROOT/'config.yaml').read_text())
        atomic_text(out/'resolved_config.yaml',yaml.safe_dump(c,sort_keys=False))
        split_receipt={'reference_manifest_sha256':identity['split_sha256'],
                       'historical_data_fingerprint':split.get('fingerprint'),
                       'train_indices':split['train_indices'],'validation_indices':split['validation_indices'],
                       'train_classes':np.bincount([train_data.dataset.targets[i] for i in split['train_indices']],minlength=10).tolist(),
                       'validation_classes':np.bincount([train_data.dataset.targets[i] for i in split['validation_indices']],minlength=10).tolist(),
                       'augmentation':'SSAM2022 normalize and post-normalization Cutout; sample-keyed RNG',
                       'sample_rng_key':'seed_for(seed, augment, zero_based_epoch, original_sample_id)',
                       'epoch_order_key':'seed_for(seed, order, zero_based_epoch)', 'test_used_for_selection':False}
        atomic_json(out/'split_manifest.json',split_receipt)
        atomic_json(out/'run_manifest.json',{'identity':identity,'model_structure':str(model),
                    'parameter_names_shapes':{n:list(p.shape) for n,p in model.named_parameters()},
                    'parameter_count':sum(p.numel() for p in model.parameters()),
                    'torchvision_version':__import__('torchvision').__version__,
                    'source_model':json.loads((ROOT/'reference/model_source.json').read_text()),
                    'differences_from_ssam':'45k/5k validation selection; single-update BN buffers; deterministic keyed RNG; tail kept',
                    'gpu_precheck':gpu,'free_disk_bytes_at_start':disk_check(ROOT,c)})
    torch.cuda.reset_peak_memory_stats()
    start_time=time.monotonic()
    steps_file=out/'step_metrics.jsonl'
    for epoch in range(start,c['epochs']):
        source_freeze(ROOT); disk=disk_check(ROOT,c); gpu_check(c,allow_own=True)
        lr=optimizer.param_groups[0]['lr']
        begun=time.monotonic()
        total_loss,total_adv,correct,count=0.,0.,0,0
        order=hashlib.sha256(); values=[]
        loader=make_loader(train_data,c,epoch,shuffle=True)
        with steps_file.open('a',encoding='utf-8') as stream:
            for batch,(x,y,ids) in enumerate(loader):
                x=x.to(c['device'],non_blocking=True); y=y.to(c['device'],non_blocking=True)
                result=sam_step(model,optimizer,x,y,c['rho'],c['numerical_epsilon'])
                total_loss+=result['train_loss']*result['count']; total_adv+=result['adversarial_train_loss']*result['count']
                correct+=result['correct']; count+=result['count']; values.append(result)
                order.update(ids.numpy().astype('<i8').tobytes())
                stream.write(json.dumps({'epoch':epoch+1,'batch':batch,'step':epoch*352+batch+1,
                                         'lr':lr,**result},allow_nan=False)+'\n')
                if (batch+1)%50==0:
                    stream.flush()
                    atomic_json(out/'latest_telemetry.json',{'status':'training','epoch':epoch+1,
                                'batch':batch+1,'steps':epoch*352+batch+1,'loss':result['train_loss']})
        if len(values)!=352 or count!=45000:
            raise RuntimeError('Training step/sample coverage mismatch')
        validation=evaluate(model,make_loader(val_data,c,epoch),c['device'])
        scheduler.step()  # Exactly once after epoch training; payload stores the next epoch LR.
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'steps':(epoch+1)*352,'learning_rate':lr,
             'train_loss':total_loss/count,'adversarial_train_loss':total_adv/count,
             'train_accuracy':correct/count,'validation_accuracy':validation['accuracy'],
             'validation_loss':validation['loss'],'order_sha256':order.hexdigest(),
             'g1_norm_mean':float(np.mean([v['g1_norm'] for v in values])),
             'g2_norm_mean':float(np.mean([v['g2_norm'] for v in values])),
             'perturbation_l2_min':min(v['perturbation_l2'] for v in values),
             'perturbation_l2_max':max(v['perturbation_l2'] for v in values),
             'epoch_seconds':time.monotonic()-begun,'elapsed_this_process_seconds':time.monotonic()-start_time,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'free_disk_bytes':disk}
        history.append(row)
        best,is_best=choose_best(best,row)
        payload=checkpoint_payload(model,optimizer,scheduler,epoch+1,history,identity,best)
        if is_best:
            digest=atomic_checkpoint(out/'best_val.pt',payload)
            atomic_json(out/'best_val.json',{**best,'checkpoint_sha256':digest,
                        'selection':'strict maximum validation accuracy, earliest tie; loss is from the same epoch'})
        atomic_checkpoint(out/'last.pt',payload)
        if epoch+1==200:
            atomic_checkpoint(out/'epoch_200.pt',payload)
        checkpoint_index(out); save_metrics(out,history)
        atomic_json(out/'status.json',{'status':'training','completed_epochs':epoch+1,
                                     'steps':(epoch+1)*352,'best':best,'commit':frozen['commit']})
        print(json.dumps(row),flush=True)
    if len(history)!=200 or history[-1]['steps']!=70400:
        raise RuntimeError('Incomplete trajectory')
    # No Test access before this point. Exactly one evaluation, only of frozen Best Validation.
    if (out/'test_eval.json').exists() or (out/'test_evaluation_started.json').exists():
        raise RuntimeError('Test evaluation already attempted; refusing another automatic evaluation')
    source_freeze(ROOT)
    selected=torch.load(out/'best_val.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(selected,identity)
    if selected['epoch']!=best['epoch']:
        raise ValueError('Selected checkpoint epoch mismatch')
    model.load_state_dict(selected['model'])
    from torchvision.datasets import CIFAR10
    test_dataset=CIFAR10(str(ROOT/c['data_root']),train=False,download=False)
    test_data=Images(test_dataset,list(range(10000)),c)
    atomic_json(out/'test_evaluation_started.json',{'checkpoint_sha256':sha_file(out/'best_val.pt'),
                                                  'epoch':selected['epoch']})
    test=evaluate(model,make_loader(test_data,c),c['device'])
    atomic_json(out/'test_eval.json',{'metrics':test,'best_epoch':best['epoch'],
                                    'best_validation':best,'checkpoint_sha256':sha_file(out/'best_val.pt'),
                                    'config_sha256':identity['config_sha256'],'test_evaluations':1})
    minimum_loss=min(history,key=lambda r:(r['validation_loss'],r['epoch']))
    checkpoint_index(out)
    summary=(f"# Dense-SAM anchor completed\n\nCommit: `{frozen['commit']}`\n\n"
             f"Config SHA256: `{identity['config_sha256']}`; split: `{identity['split_sha256']}`.\n\n"
             f"Completed 200 epochs / 70400 SGD updates. Best selection: validation Accuracy, strict improvement, earliest tie.\n\n"
             f"Best Validation: {json.dumps(best)}\n\nEpoch200: {json.dumps(history[-1])}\n\n"
             f"Lowest Validation Loss (separate): {json.dumps(minimum_loss)}\n\nFinal one-time Test: {json.dumps(test)}\n\n"
             f"GPU: {gpu['uuid']}; elapsed train processes sum: {sum(r['epoch_seconds'] for r in history):.3f}s.\n\n"
             "BN uses batch stats twice, restores all running buffers after the second backward. Standard global SAM; no FriendlySAM gradients.\n\n"
             "SSAM differences: 45k/5k held-out validation; Best Validation selection; single-update BN; deterministic sample-keyed RNG; retained tail.\n\n"
             "96.83% is a literature Test reference, not an acceptance threshold. No parameter search or additional method was run.\n\n"
             "See run_manifest.json for model structure/names/shapes, environment and source hashes; checkpoint_sha256.txt for file hashes.\n")
    for name in ['best_val.pt','last.pt','epoch_200.pt']:
        summary+=f"\n- {out/name}: {(out/name).stat().st_size} bytes; SHA256 `{sha_file(out/name)}`\n"
    atomic_text(out/'training_summary.md',summary)
    atomic_json(out/'status.json',{'status':'completed','completed_epochs':200,'steps':70400,
                                  'best':best,'test':test,'commit':frozen['commit']})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    c=yaml.safe_load((ROOT/'config.yaml').read_text())
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    # Linux lock held for the complete run, including final Test and artifact audit.
    import fcntl
    with open('/tmp/dense_sam_anchor_'+c['gpu_uuid']+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(c,args.resume)
        except BaseException as exc:
            atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),
                                          'traceback':traceback.format_exc(),'automatic_restart':False})
            atomic_json(out/'status.json',{'status':'failed','error':str(exc)})
            raise


if __name__=='__main__':
    main()
