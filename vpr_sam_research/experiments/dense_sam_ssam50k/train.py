"""One SSAM-aligned Dense-SAM trajectory, full50k / official10k Test selection."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from audit import final_audit,sums
from core import evaluate,sam_step
from data import make_loader,real_data
from model import build_model
from scheduler import CosineLRscheduler
from state import (atomic_checkpoint,atomic_json,atomic_text,checkpoint_payload,disk_check,environment,
                   gpu_check,object_sha,seed_all,sha_file,source_freeze,tensor_sha)

ROOT=Path(__file__).resolve().parent


def optimizer_for(model,c):
    opt=torch.optim.SGD(model.parameters(),lr=c['learning_rate'],momentum=c['momentum'],
                        weight_decay=c['weight_decay'],nesterov=c['nesterov'])
    scheduler=CosineLRscheduler(opt,T_max=c['epochs'],eta_min=c['eta_min'],
                               warmup_epoch=c['warmup_epoch'],warmup_init_lr=c['warmup_init_lr'])
    return opt,scheduler


def choose_best(previous,row):
    if previous is None or row['test_accuracy']>previous['test_accuracy']:
        return {'epoch':row['epoch'],'test_accuracy':row['test_accuracy'],
                'test_loss_at_best_accuracy':row['test_loss'],'train_loss':row['train_loss']},True
    return previous,False


def save_metrics(out,history):
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(history[0]))
    writer.writeheader(); writer.writerows(history)
    atomic_text(out/'epoch_metrics.csv',stream.getvalue())
    atomic_text(out/'epoch_metrics.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))


def run(c):
    if c['seed']!=0 or c['epochs']!=200 or c['purpose']!='formal' or not c['drop_last'] or c['validation_size']!=0:
        raise ValueError('Approved unique seed0/full50k/drop_last/200epoch protocol required')
    frozen=source_freeze(ROOT); gpu=gpu_check(c); disk=disk_check(ROOT,c)
    out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ['last.pt','best_test.pt','epoch_metrics.jsonl']):
        raise FileExistsError('Existing trajectory; no automatic restart or overwrite')
    seed_all(c['seed'],c['threads'])
    train,test,dataset_manifest=real_data(c)
    model=build_model().to(c['device'])
    opt,scheduler=optimizer_for(model,c)
    initialization=tensor_sha(model.state_dict())
    acceptance=json.loads((ROOT/'artifacts/precheck.json').read_text())
    if (acceptance['status']!='PASS' or acceptance['commit']!=frozen['commit']
        or acceptance['initialization_hash']!=initialization or acceptance['config_sha256']!=object_sha(c)
        or acceptance['data_manifest_sha256']!=object_sha(dataset_manifest)):
        raise ValueError('Precheck/run identity mismatch')
    if not torch.cuda.is_available() or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('CUDA / FP32 required')
    identity={'schema':2,'commit':frozen['commit'],'source_sha256':frozen['source_sha256'],
              'config':c,'config_sha256':object_sha(c),'config_file_sha256':sha_file(ROOT/'config.yaml'),
              'data_manifest_sha256':object_sha(dataset_manifest),'initialization_hash':initialization,
              'environment':environment(),'gpu_uuid':gpu['uuid'],'seed':c['seed'],
              'random_initialization':True,'warm_start_checkpoint':None}
    atomic_json(out/'resolved_config.json',c); atomic_json(out/'environment.json',identity['environment'])
    atomic_json(out/'learning_rates_expected.json',json.loads((ROOT/'reference/LR_EXPECTED.json').read_text()))
    atomic_json(out/'worker_rng_audit.json',acceptance['worker_rng_audit'])
    atomic_json(out/'data_manifest.json',dataset_manifest)
    atomic_json(out/'source_manifest.json',{'identity':identity,
                'upstream':json.loads((ROOT/'reference/UPSTREAM_SHA256.json').read_text()),
                'compatibility_patches':['Exact parameter copy restoration instead of subtracting perturbation',
                    'Current PyTorch2.8/CUDA12.8; deterministic FP32, TF32 off; workers2 rather than author8',
                    'Finite/BN/source guards, complete schema2 SGD+RNG state, atomic verified checkpoint publication',
                    'User seed0 override of task seed1234; no pretrained weights'],
                'model_structure':str(model),'parameter_count':sum(p.numel() for p in model.parameters()),
                'parameter_names_shapes':{n:list(p.shape) for n,p in model.named_parameters()},
                'import_provenance':{'model':str(__import__('model').__file__),'core':str(__import__('core').__file__),
                                     'data':str(__import__('data').__file__),'scheduler':str(__import__('scheduler').__file__)}})
    history=[]; best=None; best_sha=None
    torch.cuda.reset_peak_memory_stats()
    process_start=time.monotonic(); utc_start=time.time()
    for epoch in range(200):
        source_freeze(ROOT); free=disk_check(ROOT,c); gpu_observed=gpu_check(c,allow_own=True)
        begun=time.monotonic(); lr=opt.param_groups[0]['lr']
        total_loss=total_adv=0.; correct=count=0; values=[]; seen=[]; digest=hashlib.sha256()
        with (out/'step_metrics.jsonl').open('a',encoding='utf-8') as log:
            for batch,(x,y,ids) in enumerate(make_loader(train,c,training=True)):
                result=sam_step(model,opt,x.to(c['device'],non_blocking=True),y.to(c['device'],non_blocking=True),
                                c['rho'],c['numerical_epsilon'],audit_bn=(batch==0))
                total_loss+=result['train_loss']*result['count']; total_adv+=result['adversarial_train_loss']*result['count']
                correct+=result['correct']; count+=result['count']; values.append(result)
                seen.extend(ids.tolist()); digest.update(ids.numpy().astype('<i8').tobytes())
                log.write(json.dumps({'epoch':epoch+1,'batch':batch+1,'step':epoch*390+batch+1,'lr':lr,**result},allow_nan=False)+'\n')
                if (batch+1)%50==0:
                    log.flush()
                    atomic_json(out/'latest_telemetry.json',{'status':'training','epoch':epoch+1,'batch':batch+1,
                                'steps':epoch*390+batch+1,'unix_time':time.time(),'training_pid':os.getpid(),
                                'gpu_allocated_bytes':torch.cuda.memory_allocated(),'loss':result['train_loss']})
        if len(values)!=390 or count!=49920 or len(set(seen))!=49920:
            raise ValueError('Train batch/sample coverage')
        omitted=sorted(set(range(50000))-set(seen))
        if len(omitted)!=80: raise ValueError('Expected exactly80 omitted samples')
        scheduler.step(epoch)  # EXACT official call: zero-based epoch, AFTER training, BEFORE Test.
        measured=evaluate(model,make_loader(test,c),c['device'])
        if measured['count']!=10000: raise ValueError('Test must include all10000 examples')
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'steps':(epoch+1)*390,'train_samples':count,'dataset_train_size':50000,
             'omitted_samples':80,'omitted_sample_ids':omitted,'train_order_sha256':digest.hexdigest(),
             'learning_rate_used':lr,'learning_rate_after_epoch':opt.param_groups[0]['lr'],
             'train_loss':total_loss/count,'adversarial_train_loss':total_adv/count,'train_accuracy':correct/count,
             'test_accuracy':measured['accuracy'],'test_loss':measured['loss'],'test_count':measured['count'],
             'g1_norm_mean':float(np.mean([v['g1_norm'] for v in values])),
             'g2_norm_mean':float(np.mean([v['g2_norm'] for v in values])),
             'perturbation_l2_min':min(v['perturbation_l2'] for v in values),
             'perturbation_l2_max':max(v['perturbation_l2'] for v in values),
             'bn_counters_per_step':2,'epoch_seconds_excluding_publication':time.monotonic()-begun,
             'elapsed_seconds_before_publication':time.monotonic()-process_start,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'free_disk_bytes':free,
             'gpu_epoch_start':gpu_observed}
        history.append(row); best,is_best=choose_best(best,row)
        payload=checkpoint_payload(model,opt,scheduler,epoch+1,history,identity,best)
        if is_best:
            best_sha=atomic_checkpoint(out/'best_test.pt',payload)
        last_sha=atomic_checkpoint(out/'last.pt',payload)
        atomic_json(out/'checkpoint_hashes.json',{'epoch':epoch+1,'best_test.pt':best_sha,'last.pt':last_sha})
        save_metrics(out,history)
        atomic_json(out/'learning_rates_actual.json',[{'epoch':r['epoch'],'used':r['learning_rate_used'],
                     'after_epoch':r['learning_rate_after_epoch']} for r in history])
        atomic_json(out/'selection_metadata.json',{'best':best,'latest':row,
                    'selection':'strict maximum Test Accuracy; earliest tie; Test used for model selection',
                    'independent_generalization_estimate':False})
        atomic_json(out/'status.json',{'status':'training','completed_epochs':epoch+1,'steps':(epoch+1)*390,
                                     'best':best,'training_pid':os.getpid(),'unix_time':time.time()})
        print(json.dumps(row),flush=True)
    # No extra Test call, retuning, pruning or subsequent training.
    audit=final_audit(ROOT,out)
    finish=time.time()
    report=['# Dense-SAM full50k / Test-selection final report','',
            f"Commit: `{frozen['commit']}`; seed0; completed200epochs /78000optimizer updates.",
            f"Config SHA256: `{identity['config_sha256']}`; config file SHA256: `{identity['config_file_sha256']}`.",
            '50,000 training IDs;49,920 presentations per epoch;80 shuffled tail examples dropped;9,984,000 presentations total.',
            '**Test is used for checkpoint selection each epoch; Best Test is not an unbiased independent generalization estimate.**','',
            '| Metric | Best Test checkpoint | Last checkpoint |','|---|---:|---:|']
    b,l=audit['selected']['best_test.pt'],audit['selected']['last.pt']
    for label,key in [('Epoch','epoch'),('Test Accuracy (%)','test_accuracy'),('Test Loss','test_loss'),('Train Loss','train_loss'),('Checkpoint SHA256','sha256')]:
        x,y=b[key],l[key]
        if key=='test_accuracy': x,y=x*100,y*100
        report.append(f'| {label} | {x} | {y} |')
    report += ['',f"Best minus literature96.83%: {100*b['test_accuracy']-96.83:.4f} percentage points.",
               'BN: both training forwards update running mean, variance and counters. No BN freezing or recalibration.',
               'SSAM scheduler copied exactly: epoch-end step(zero-based epoch); first two LR values=.05.',
               'F-SAM differences and compatibility deviations: see CONFIG_COMPARISON.md and source_manifest.json.',
               f"Wall time through training and final independent audit: {finish-utc_start:.3f}s; start Unix={utc_start}, finish Unix={finish}.",
               f"Peak allocated bytes={torch.cuda.max_memory_allocated()}, peak reserved bytes={torch.cuda.max_memory_reserved()}, free disk bytes={disk_check(ROOT,c)}.",
               f"Historical files verified unchanged: {audit['historical_files_unchanged']}.",
               f"Best path: {out/'best_test.pt'}; Last path: {out/'last.pt'}.",
               'The healthy selected checkpoint may be frozen as a common Dense parent; downstream VPR/Guide is not authorized or launched by this run.',
               'Single seed only; no performance threshold, hyperparameter search, auto-restart, extra seed or pruning.']
    atomic_text(out/'FINAL_REPORT.md','\n'.join(report)+'\n')
    atomic_json(out/'status.json',{'status':'completed','completed_epochs':200,'steps':78000,'best':best,
                                 'final':history[-1],'final_audit':'PASS','follow_on_gpu_tasks':False})
    sums(out)


def main():
    import fcntl
    c=yaml.safe_load((ROOT/'config.yaml').read_text()); out=ROOT/c['output']; out.mkdir(parents=True,exist_ok=True)
    with open('/tmp/dense_sam_ssam50k_'+c['gpu_uuid']+'.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: run(c)
        except BaseException as exc:
            atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),
                        'traceback':traceback.format_exc(),'automatic_restart':False,'unix_time':time.time()})
            atomic_json(out/'status.json',{'status':'failed','error':str(exc)})
            raise


if __name__=='__main__': main()
