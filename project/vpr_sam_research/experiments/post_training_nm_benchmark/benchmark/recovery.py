"""Original20epoch fixed-Mask SGD protocol, with exact healthy epoch-boundary resume."""
import hashlib,time
from pathlib import Path
import torch
from torch import nn
from . import artifacts
from .common import atomic_json,finite,restore_rng,rng_state
from .evaluation import evaluate
from .nm import project,validate_layer_mask

def recover(model,masks,train_factory,test_factory,config,out,identity,parent,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rc=config['recovery'];device=config['device']
    if config['purpose']=='formal' and rc['epochs']!=20:raise ValueError('Formal recovery is exactly20epochs')
    optimizer=torch.optim.SGD(model.parameters(),lr=rc['learning_rate'],momentum=rc['momentum'],
        weight_decay=rc['weight_decay'],nesterov=rc['nesterov'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,rc['epochs'],eta_min=rc['eta_min'])
    history=[];best=None;start=0;best_sha=None;begun=time.monotonic()
    if resume:
        value=artifacts.load(out/'last.pt',identity)
        if value['stage']!='recovery' or value['parent']!=parent:raise ValueError('Recovery resume parent/stage')
        model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
        if artifacts.tensor_sha(masks)!=value['mask_sha256']:raise ValueError('Recovery resume Mask changed')
        history=value['history'];best=value['best'];start=value['epoch'];restore_rng(value['rng'])
        if (out/'best_test.pt').exists():best_sha=artifacts.sha_file(out/'best_test.pt')
    elif (out/'last.pt').exists():raise FileExistsError('Use explicit resume; no overwrite of an old trajectory')
    for epoch in range(start,rc['epochs']):
        model.train();loss_sum=correct=count=steps=0;order=hashlib.sha256();lr=optimizer.param_groups[0]['lr']
        for x,y,ids in train_factory():
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y);finite([loss],'Recovery center loss');loss.backward()
            project(model,masks,optimizer);finite((p.grad for p in model.parameters() if p.grad is not None),'Recovery gradients')
            optimizer.step();project(model,masks,optimizer);finite(model.state_dict().values(),'Recovery model')
            loss_sum+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());count+=len(y);steps+=1
            order.update(ids.numpy().astype('<i8').tobytes())
        if steps==0:raise ValueError('Empty recovery data')
        if config['purpose']=='formal' and (steps,count)!=(390,49920):raise ValueError('Formal recovery coverage')
        for name,p in model.named_parameters():
            if name not in masks:continue
            validate_layer_mask(p,masks[name])
            if 'momentum_buffer' in optimizer.state[p] and torch.count_nonzero(optimizer.state[p]['momentum_buffer'][~masks[name].to(device)]):
                raise ValueError('Masked momentum regrowth')
        metrics=evaluate(model,test_factory(),device)
        if config['purpose']=='formal' and metrics['count']!=10000:raise ValueError('Full official Test required')
        scheduler.step() # exactly after training and Test evaluation, matching existing recovery
        row={'epoch':epoch+1,'steps':(history[-1]['steps'] if history else 0)+steps,'learning_rate':lr,
            'train_samples':count,'train_loss':loss_sum/count,'train_accuracy':correct/count,
            'test_accuracy':metrics['accuracy'],'test_loss':metrics['loss'],'test_count':metrics['count'],
            'order_sha256':order.hexdigest(),'elapsed_seconds':time.monotonic()-begun}
        history.append(row);improved=best is None or row['test_accuracy']>best['test_accuracy']
        if improved:best={k:row[k] for k in ('epoch','test_accuracy','test_loss')}
        value={'identity':identity,'stage':'recovery','parent':parent,'model':model.state_dict(),'masks':masks,
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'steps':row['steps'],
            'history':history,'best':best,'rng':rng_state(),'selection':'strict maximum Test Accuracy, earliest tie; Test used for selection'}
        artifacts.save(out/f'epoch_{epoch+1:03d}.pt',value)
        if improved:best_sha=artifacts.save(out/'best_test.pt',value)
        last_sha=artifacts.save(out/'last.pt',value)
        atomic_json(out/'epochs.json',history)
        atomic_json(out/'checkpoint_hashes.json',{'best_test.pt':best_sha,'last.pt':last_sha})
        print(f'Recovery epoch{epoch+1}/{rc["epochs"]} Test={metrics["accuracy"]:.6f}',flush=True)
    return {'history':history,'best':best,'final':history[-1]}
