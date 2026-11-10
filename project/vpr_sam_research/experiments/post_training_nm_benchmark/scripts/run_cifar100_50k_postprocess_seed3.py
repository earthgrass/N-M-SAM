"""One fixed CIFAR-100 2:4 pruning + BN-only + 20-epoch recovery branch."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np
import torch, yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.rose import ROSEAdapter
from baseline_adapters.sparsegpt import SparseGPTAdapter
from baseline_adapters.wanda import WandaAdapter
from benchmark.nm import eligible_layers,magnitude_mask,project,validate_masks
PROTOCOL='cifar100_50k10k_final_v1'

def atomic_save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path);return train.digest_file(path)

def model_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class Subset(Dataset):
    def __init__(self,data,labels,ids,transform):self.data,self.labels,self.ids,self.transform=data,labels,list(ids),transform
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        sid=int(self.ids[i]);image=self.data[sid].reshape(3,32,32).transpose(1,2,0)
        return self.transform(image),int(self.labels[sid]),sid

def loader(dataset,batch=128):return DataLoader(dataset,batch_size=batch,shuffle=False,drop_last=False,num_workers=2,pin_memory=True)

def guided_masks(model,geometry):
    mapping=[(n,tuple(p.shape)) for n,p in model.named_parameters()]
    if geometry['method']!='vpr' or geometry['mapping']!=mapping:raise ValueError('Matching VPR geometry required')
    logs={n:s for (n,_),s in zip(geometry['mapping'],geometry['logs'])};masks={}
    for name,layer in eligible_layers(model).items():
        key=name+'.weight';w=layer.weight.detach();s=logs[key].to(w.device)
        if s.dtype!=torch.float64 or s.shape!=w.shape or not torch.isfinite(s).all():raise ValueError('Geometry tensor mismatch')
        score=w.abs().double()*torch.exp(-s);flat=score.reshape(score.shape[0],-1);complete=flat.shape[1]//4*4
        mask=torch.ones_like(flat,dtype=torch.bool);groups=flat[:,:complete].reshape(flat.shape[0],-1,4)
        keep=groups.argsort(dim=-1,descending=True,stable=True)[...,:2];head=torch.zeros_like(groups,dtype=torch.bool);head.scatter_(-1,keep,True)
        mask[:,:complete]=head.reshape(flat.shape[0],complete);masks[key]=mask.reshape_as(w)
    return masks

def prune(model,method,batches,parent,device):
    layers=eligible_layers(model);adapters={};hooks=[];started=time.perf_counter()
    if method in {'wanda','sparsegpt','rose'}:
        for name,layer in layers.items():
            adapter=WandaAdapter(layer) if method=='wanda' else (SparseGPTAdapter(layer,compute_dtype=torch.float32) if method=='sparsegpt' else ROSEAdapter(layer,compute_dtype=torch.float32))
            adapters[name]=adapter;hooks.append(layer.register_forward_pre_hook(lambda _m,x,item=adapter:item.add_batch(x[0])))
        model.eval()
        with torch.no_grad():
            for x,_,_ in batches:model(x.to(device,non_blocking=True))
        for hook in hooks:hook.remove()
    calibration_seconds=time.perf_counter()-started;started=time.perf_counter();masks={}
    if method=='apbr_sam':masks=guided_masks(model,parent['stepper']['geometry']);project(model,masks)
    else:
        for name,layer in layers.items():
            if method=='magnitude':
                keep=magnitude_mask(layer.weight,2,4)
                with torch.no_grad():layer.weight.mul_(keep)
            elif method=='wanda':keep=adapters[name].prune(keep_n=2,m=4)
            elif method=='sparsegpt':keep=adapters[name].prune(keep_n=2,m=4,blocksize=128,percdamp=.01)
            else:keep,_=adapters[name].prune(keep_n=2,m=4,blocksize=128,percdamp=.01,threshold=.5)
            masks[name+'.weight']=keep.detach().cpu();adapters.pop(name,None)
    validate_masks(model,masks,n=2,m=4,complete=True)
    return masks,{'calibration_seconds':calibration_seconds,'pruning_seconds':time.perf_counter()-started,'mask_sha256':train.tensor_sha(masks)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--method',choices=('magnitude','wanda','sparsegpt','rose','apbr_sam'),required=True);p.add_argument('--seed',type=int,choices=(3,),required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Refusing nonempty method output')
    c=yaml.safe_load(a.config.read_text());parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    expected='apbr_sam' if a.method=='apbr_sam' else 'dense_sam'
    if parent.get('protocol_id')!=PROTOCOL or parent.get('epoch')!=200 or parent['identity']['method']!=expected or parent['identity']['seed']!=a.seed:raise ValueError('Final parent identity mismatch')
    if parent.get('test_accessed') is not False:raise ValueError('Parent Test policy')
    train.seed_all(a.seed);device=torch.device('cuda:0')
    with (a.dataset_root/'extracted/cifar-100-python/train').open('rb') as f:raw=pickle.load(f,encoding='bytes')
    data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64)
    plain=transforms.Compose([transforms.ToTensor(),transforms.Normalize(c['dataset']['normalize_mean'],c['dataset']['normalize_std'])])
    augment=transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(.5),transforms.ToTensor(),transforms.Normalize(c['dataset']['normalize_mean'],c['dataset']['normalize_std'])])
    calib=np.random.default_rng(c['formal_calibration']['selection_seed']).permutation(50000)[:1024].tolist()
    bnids=np.random.default_rng(c['bn_only']['selection_seed']).permutation(50000)[:4096].tolist()
    model=build_cifar100_model().to(device);model.load_state_dict(parent['model']);torch.cuda.reset_peak_memory_stats()
    masks,report=prune(model,a.method,loader(Subset(data,labels,calib,plain)),parent,device)
    raw_payload={'schema':1,'status':'healthy','protocol_id':PROTOCOL,'stage':'raw_hard','method':a.method,'seed':a.seed,'epoch':0,
      'model':model_state(model),'masks':{k:v.cpu() for k,v in masks.items()},'mask_sha256':train.tensor_sha(masks),
      'parent':{'path':str(a.parent),'sha256':train.digest_file(a.parent)},'pruning':report,'calibration_ids':calib,'test_accessed':False}
    raw_path=out/'raw_hard.pt';raw_sha=atomic_save(raw_path,raw_payload)
    calibrated=copy.deepcopy(model);before=train.tensor_sha(dict(calibrated.named_parameters()))
    for module in calibrated.modules():
        module.eval()
        if isinstance(module,nn.BatchNorm2d):module.reset_running_stats();module.momentum=None;module.train()
    with torch.no_grad():
        for x,_,_ in loader(Subset(data,labels,bnids,plain)):calibrated(x.to(device,non_blocking=True))
    if train.tensor_sha(dict(calibrated.named_parameters()))!=before:raise ValueError('BN-only changed parameters')
    bn_payload=dict(raw_payload);bn_payload.update(stage='bn_only',model=model_state(calibrated),bn_ids=bnids,raw_parent_sha256=raw_sha)
    bn_sha=atomic_save(out/'bn_only.pt',bn_payload);del calibrated
    full=train.OfficialTrain(data,labels,augment);optimizer=torch.optim.SGD(model.parameters(),lr=.005,momentum=.9,weight_decay=.0005,nesterov=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=0.);history=[];started=time.perf_counter();final=None
    for epoch in range(20):
        model.train();loss_sum=0.;correct=count=batches=0;order=hashlib.sha256();lr=optimizer.param_groups[0]['lr'];begun=time.perf_counter()
        for x,y,ids in train.training_loader(full,128,a.seed,epoch):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Recovery loss')
            loss.backward();project(model,masks,optimizer);optimizer.step();project(model,masks,optimizer)
            loss_sum+=float(loss)*len(y);correct+=int((logits.argmax(1)==y).sum());count+=len(y);batches+=1;order.update(ids.numpy().astype('<i8').tobytes())
        if count!=50000 or batches!=391:
            raise ValueError('Recovery coverage')
        validate_masks(model,masks,n=2,m=4,complete=True);scheduler.step()
        row={'epoch':epoch+1,'steps':(epoch+1)*391,'train_samples':count,'batches':batches,'train_order_sha256':order.hexdigest(),'learning_rate':lr,'train_loss':loss_sum/count,'train_accuracy':correct/count,'epoch_seconds':time.perf_counter()-begun,'test_accessed':False};history.append(row)
        final={'schema':1,'status':'healthy','protocol_id':PROTOCOL,'stage':'recovery','selection_mode':'fixed_final_epoch','method':a.method,'seed':a.seed,'epoch':epoch+1,'steps':(epoch+1)*391,'model':model_state(model),'masks':raw_payload['masks'],'mask_sha256':raw_payload['mask_sha256'],'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'history':history,'parent':{'path':str(raw_path),'sha256':raw_sha},'test_accessed':False}
        last_sha=atomic_save(out/'recovery/last.pt',final);train.atomic_json(out/'status.json',{'status':'recovery','epoch':epoch+1,'steps':final['steps'],'last_sha256':last_sha,'test_accessed':False});print(json.dumps({'method':a.method,**row}),flush=True)
    if final['epoch']!=20 or final['steps']!=7820:raise RuntimeError('Recovery budget')
    final_sha=atomic_save(out/'recovery/final_epoch_020.pt',final)
    train.atomic_json(out/'status.json',{'status':'completed','protocol_id':PROTOCOL,'method':a.method,'seed':a.seed,'raw_hard_sha256':raw_sha,'bn_only_sha256':bn_sha,'recovery_final_sha256':final_sha,'mask_sha256':raw_payload['mask_sha256'],'recovery_epochs':20,'recovery_steps':7820,'test_accessed':False,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.perf_counter()-started})

if __name__=='__main__':
    try:main()
    except Exception as e:
        if '--output-dir' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output-dir')+1]);train.atomic_json(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'automatic_retry':False});train.atomic_json(out/'status.json',{'status':'failed','message':str(e)})
        raise
