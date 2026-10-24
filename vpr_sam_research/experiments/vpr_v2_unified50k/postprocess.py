"""Three matched branches. Raw -> independent BN-only, and Raw -> fresh fixed-mask SGD."""
import copy,csv,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from core import evaluate,require_finite
from data import Images,make_loader,real_data
from model import build_model
from masks import build_masks,project,validate_masks
from state import (atomic_checkpoint,atomic_json,atomic_text,checkpoint_payload,disk_check,object_sha,
                   seed_all,sha_file,source_freeze,tensor_sha,validate_checkpoint)
from anchor import verify_anchor

ROOT=Path(__file__).resolve().parent


def save_artifact(path,model,masks,identity,parent,geometry_reference,metrics):
    # Complete weights/mask, immutable geometry parent reference; no duplicated geometry arrays.
    opt=torch.optim.SGD(model.parameters(),lr=.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
    value=checkpoint_payload(model,opt,sched,0,[],identity,None)
    value.update(stage='pruned',masks=masks,mask_sha256=tensor_sha(masks),parent=parent,
                 geometry_reference=geometry_reference,metrics=metrics)
    return atomic_checkpoint(path,value)


def recover(raw_path,raw,train,test,c,output):
    model=build_model().to(c['device']);model.load_state_dict(raw['model']);masks=raw['masks']
    rc=c['recovery'];opt=torch.optim.SGD(model.parameters(),lr=rc['learning_rate'],momentum=rc['momentum'],
                      weight_decay=rc['weight_decay'],nesterov=rc['nesterov'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,rc['epochs'],eta_min=rc['eta_min'])
    history=[];best=None;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(20):
        source_freeze(ROOT);disk_check(ROOT,c)
        loss_sum=correct=count=0;order=__import__('hashlib').sha256();lr=opt.param_groups[0]['lr']
        model.train()
        for x,y,ids in make_loader(train,c,training=True):
            x,y=x.to(c['device']),y.to(c['device']);opt.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y);require_finite([loss],'Recovery loss');loss.backward()
            project(model,masks,opt);require_finite((p.grad for p in model.parameters() if p.grad is not None),'Recovery gradients')
            opt.step();project(model,masks,opt);require_finite(model.state_dict().values(),'Recovery state')
            loss_sum+=loss.item()*len(y);correct+=int((logits.argmax(1)==y).sum());count+=len(y)
            order.update(ids.numpy().astype('<i8').tobytes())
        if count!=49920:raise ValueError('Recovery sample count')
        validate_masks(model,masks);metrics=evaluate(model,make_loader(test,c),c['device']);sched.step()
        for name,p in model.named_parameters():
            if name in masks and torch.count_nonzero(opt.state[p]['momentum_buffer'][~masks[name].to(p.device)]):
                raise ValueError('Masked momentum regrowth')
        row={'epoch':epoch+1,'steps':(epoch+1)*390,'learning_rate':lr,'train_samples':count,
             'train_loss':loss_sum/count,'train_accuracy':correct/count,'test_accuracy':metrics['accuracy'],
             'test_loss':metrics['loss'],'test_count':metrics['count'],'order_sha256':order.hexdigest(),
             'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        history.append(row);better=best is None or row['test_accuracy']>best['test_accuracy']
        if better:best={k:row[k] for k in ['epoch','test_accuracy','test_loss']}
        payload=checkpoint_payload(model,opt,sched,epoch+1,history,raw['identity'],best)
        payload.update(stage='recovery',masks=masks,mask_sha256=raw['mask_sha256'],
                       parent={'path':str(raw_path),'sha256':sha_file(raw_path)},geometry_reference=raw['geometry_reference'])
        atomic_checkpoint(output/f'epoch_{epoch+1:03d}.pt',payload)
        if better:atomic_checkpoint(output/'best_test.pt',payload)
        if epoch==19:atomic_checkpoint(output/'last.pt',payload)
        atomic_json(output/'epochs.json',history);atomic_json(output/'status.json',{'epoch':epoch+1,'best':best,'latest':row})
        print(json.dumps({'stage':'recovery','branch':output.parent.name,**row}),flush=True)
    return {'best':best,'final':history[-1],'epochs':history,'initial_raw_sha256':sha_file(raw_path),
            'mask_sha256':raw['mask_sha256'],'last_sha256':sha_file(output/'last.pt'),
            'best_sha256':sha_file(output/'best_test.pt')}


def main():
    from state import gpu_check
    c=yaml.safe_load((ROOT/'config.yaml').read_text());frozen=source_freeze(ROOT)
    gpu_check(c)
    root=ROOT/'artifacts/comparison';root.mkdir(exist_ok=False)
    train,test,manifest=real_data(c)
    anchor,_,anchor_receipt=verify_anchor(c)
    vpr_path=ROOT/c['output']/'best_test.pt';vpr=torch.load(vpr_path,map_location='cpu',weights_only=False);validate_checkpoint(vpr)
    if vpr['identity']['initialization_hash']!=anchor_receipt['initialization_hash']:raise ValueError('Unpaired initialization')
    if vpr['identity']['data_manifest_sha256']!=anchor_receipt['data_manifest_sha256']:raise ValueError('Unpaired data')
    branches=[('sam_magnitude',anchor,Path(anchor_receipt['path']),'magnitude'),
              ('vpr_magnitude',vpr,vpr_path,'magnitude'),('vpr_guided',vpr,vpr_path,'geometry_guided')]
    summaries=[];recovery_orders=None
    ids=json.loads((ROOT/'reference/bn_train_ids.json').read_text())
    if len(ids)!=4096 or len(set(ids))!=4096 or not set(ids)<=set(range(50000)):raise ValueError('BN training IDs')
    for name,parent,parent_path,score in branches:
        disk_check(ROOT,c);out=root/name;out.mkdir()
        seed_all(c['seed'],c['threads']);model=build_model().to(c['device']);model.load_state_dict(parent['model'])
        geometry=parent.get('geometry');geometry_ref=None if geometry is None else {
            'parent_checkpoint_sha256':sha_file(parent_path),'geometry_sha256':parent['geometry_sha256'],
            'epoch':parent['epoch'],'method':'fixed_l2'}
        dense=evaluate(model,make_loader(test,c),c['device'])
        # Evaluation DataLoader RNG consumption is isolated from the common recovery stream below.
        masks=build_masks(model,score,geometry);project(model,masks);validate_masks(model,masks)
        hard=evaluate(model,make_loader(test,c),c['device'])
        identity={'commit':frozen['commit'],'source_sha256':frozen['source_sha256'],'config':c,
                  'config_sha256':object_sha(c),'stage':'postprocess','branch':name,'seed':0,
                  'parent':{'path':str(parent_path),'sha256':sha_file(parent_path)}}
        raw_path=out/'raw_hard.pt';save_artifact(raw_path,model,masks,identity,identity['parent'],geometry_ref,hard)
        raw=torch.load(raw_path,map_location='cpu',weights_only=False);validate_checkpoint(raw)
        calibrated=copy.deepcopy(model).eval();before=tensor_sha(dict(calibrated.named_parameters()))
        for p in calibrated.parameters():p.requires_grad_(False)
        for module in calibrated.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.reset_running_stats();module.momentum=None;module.train()
        calibration=torch.utils.data.Subset(Images(train.dataset,c,False),ids)
        with torch.no_grad():
            for x,_,_ in make_loader(calibration,c,workers=0):calibrated(x.to(c['device']))
        if tensor_sha(dict(calibrated.named_parameters()))!=before:raise ValueError('BN-only changed trainable weights')
        for module in calibrated.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm) and int(module.num_batches_tracked)!=32:raise ValueError('BN calibration count')
        bn=evaluate(calibrated,make_loader(test,c),c['device'])
        save_artifact(out/'bn_only.pt',calibrated,masks,identity,{'path':str(raw_path),'sha256':sha_file(raw_path)},geometry_ref,bn)
        # Every recovery starts from untouched Raw with a fresh optimizer and identical RNG stream.
        seed_all(c['seed'],c['threads'])
        recovery=recover(raw_path,raw,train,test,c,out/'recovery')
        orders=[r['order_sha256'] for r in recovery['epochs']]
        if recovery_orders is None:recovery_orders=orders
        elif orders!=recovery_orders:raise ValueError('Recovery data orders differ')
        summary={'method':name,'dense':dense,'raw_hard':hard,'bn_only':bn,'recovery_best':recovery['best'],
                 'recovery_final':recovery['final'],'recovery':recovery,'parent':identity['parent'],
                 'mask_sha256':tensor_sha(masks),'geometry_reference':geometry_ref,
                 'raw_sha256':sha_file(raw_path),'bn_only_sha256':sha_file(out/'bn_only.pt'),
                 'accuracy_damage_pp':100*(dense['accuracy']-hard['accuracy']),
                 'loss_damage':hard['loss']-dense['loss'],
                 'recovery_best_gain_pp':100*(recovery['best']['test_accuracy']-hard['accuracy'])}
        atomic_json(out/'summary.json',summary);summaries.append(summary)
    if summaries[1]['parent']!=summaries[2]['parent']:raise ValueError('VPR mask branches have different parents')
    atomic_json(root/'summary.json',summaries)
    lines=['# SAM / VPR-v2 paired seed0 comparison','','All endpoints use official Test. Best Test selection is not independent unbiased generalization evidence.','',
           '| Method | Dense Acc | Raw Acc/Loss | BN Acc/Loss | Recovery Best/Epoch/Loss | Recovery Final Acc/Loss |',
           '|---|---:|---|---|---|---|']
    for s in summaries:
        b=s['recovery_best'];f=s['recovery_final']
        lines.append(f"| {s['method']} | {100*s['dense']['accuracy']:.2f} | {100*s['raw_hard']['accuracy']:.2f} / {s['raw_hard']['loss']:.8f} | {100*s['bn_only']['accuracy']:.2f} / {s['bn_only']['loss']:.8f} | {100*b['test_accuracy']:.2f} / {b['epoch']} / {b['test_loss']:.8f} | {100*f['test_accuracy']:.2f} / {f['test_loss']:.8f} |")
    lines+=['','BN-only and recovery use separate Raw copies; three recovery orders match. All20 recovery epoch checkpoints retained.',
            'One seed; no significance, SOTA or universal geometry advantage claim. No additional seed, scoring method or Gamma search.']
    atomic_text(root/'FINAL_COMPARISON.md','\n'.join(lines)+'\n')
    # Standalone scientific curves, no model execution.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for s in summaries:
        rows=s['recovery']['epochs'];x=[r['epoch'] for r in rows]
        axes[0].plot(x,[100*r['test_accuracy'] for r in rows],label=s['method']);axes[1].plot(x,[r['test_loss'] for r in rows],label=s['method'])
    for ax,title in zip(axes,['Recovery Test Accuracy (%)','Recovery Test CE Loss']):ax.set(xlabel='Epoch',ylabel=title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'recovery_curves.png',dpi=180);plt.close(fig)
    # Dense curves contain the full audited history, not only selected epochs.
    old=[json.loads(line) for line in (Path(c['anchor']['root'])/c['anchor']['run']/'epoch_metrics.jsonl').read_text().splitlines()]
    new=[json.loads(line) for line in (ROOT/c['output']/'epoch_metrics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,rows in [('SAM',old),('VPR-L2',new)]:
        axes[0].plot([r['epoch'] for r in rows],[100*r['test_accuracy'] for r in rows],label=name)
        axes[1].plot([r['epoch'] for r in rows],[r['test_loss'] for r in rows],label=name)
    for ax,label in zip(axes,['Test Accuracy (%)','Test CE Loss']):ax.set(xlabel='Epoch',ylabel=label);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'dense_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['epoch'] for r in new],[r['geometry']['log_radius_min'] for r in new],label='min')
    axes[0].plot([r['epoch'] for r in new],[r['geometry']['log_radius_max'] for r in new],label='max')
    axes[1].plot([r['epoch'] for r in new],[r['geometry']['log_radius_rms'] for r in new],label='RMS')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'geometry_curves.png',dpi=180);plt.close(fig)
    final_post_audit(root,summaries,c)


def final_post_audit(root,summaries,c):
    from audit import verify_history
    frozen=source_freeze(ROOT);files={}
    model=build_model()
    for summary in summaries:
        branch=root/summary['method'];rows=summary['recovery']['epochs'];raw_path=branch/'raw_hard.pt'
        raw=torch.load(raw_path,map_location='cpu',weights_only=False);validate_checkpoint(raw)
        bn=torch.load(branch/'bn_only.pt',map_location='cpu',weights_only=False);validate_checkpoint(bn)
        if raw['parent']['sha256']!=sha_file(raw['parent']['path']):raise ValueError('Raw parent changed')
        if tensor_sha(raw['masks'])!=raw['mask_sha256'] or tensor_sha(bn['masks'])!=raw['mask_sha256']:raise ValueError('Mask hash')
        model.load_state_dict(raw['model']);validate_masks(model,raw['masks'])
        parameters=set(dict(model.named_parameters()))
        if any(not torch.equal(raw['model'][n],bn['model'][n]) for n in parameters):raise ValueError('BN-only changed weight')
        if [r['epoch'] for r in rows]!=list(range(1,21)) or rows[-1]['steps']!=7800:raise ValueError('Recovery coverage')
        best=max(rows,key=lambda r:(r['test_accuracy'],-r['epoch']))
        for name,epoch in [(f'epoch_{i:03d}.pt',i) for i in range(1,21)]+[('best_test.pt',best['epoch']),('last.pt',20)]:
            path=branch/'recovery'/name;value=torch.load(path,map_location='cpu',weights_only=False);validate_checkpoint(value)
            if value['epoch']!=epoch or value['history'][-1]!=rows[epoch-1] or value['steps']!=epoch*390:raise ValueError('Recovery selection')
            if value['identity']['commit']!=frozen['commit'] or value['identity']['source_sha256']!=frozen['source_sha256']:raise ValueError('Recovery source')
            if value['parent']['sha256']!=sha_file(raw_path) or value['mask_sha256']!=raw['mask_sha256']:raise ValueError('Recovery parent/mask')
            if value['geometry_reference']!=raw['geometry_reference']:raise ValueError('Geometry parent mismatch')
            model.load_state_dict(value['model']);validate_masks(model,value['masks'])
            files[str(path.relative_to(ROOT))]=sha_file(path)
        for path in [raw_path,branch/'bn_only.pt']:files[str(path.relative_to(ROOT))]=sha_file(path)
    unchanged=verify_history(ROOT/'artifacts/historical_before.json')
    atomic_json(root/'FINAL_AUDIT.json',{'status':'PASS','commit':frozen['commit'],'seed':0,
                'new_dense_trajectories':1,'recovery_trajectories':3,'recovery_epochs':20,
                'historical_files_unchanged':unchanged,'checkpoint_sha256':files,
                'test_used_for_checkpoint_selection':True,'additional_gpu_tasks':False})
    atomic_text(root/'SHA256SUMS',''.join(f'{digest}  {name}\n' for name,digest in sorted(files.items())))


if __name__=='__main__':main()
