"""Seed-0 full-network Shared-Q Train-side diagnostic screen; never accesses Test."""
from __future__ import annotations
import argparse,csv,hashlib,json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_apbr_weights_original_sparsegpt_one as common
import run_cifar100_50k_formal_training as train
from baseline_adapters.apbr_gpt import apbr_gpt_matrix,fixedmask_shared_q_obs_matrix,normalized_inverse_square
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter,sparsegpt_matrix
from benchmark.nm import eligible_layers,validate_masks
ALPHAS=(0.,1e-4,1e-3,1e-2,1e-1,1.); METHODS=('HQ','isotropic','shuffled','QH','QQ')

def csvwrite(p,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--original-mask',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError(out)
 if not torch.cuda.is_available():raise RuntimeError('CUDA')
 cfg=yaml.safe_load(a.config.read_text());parent=torch.load(a.parent,map_location='cpu',weights_only=False);dense=parent['model'];orig_masks=torch.load(a.original_mask,map_location='cpu',weights_only=False)
 if parent.get('epoch')!=200 or parent.get('identity',{}).get('method')!='apbr_sam' or parent.get('identity',{}).get('seed')!=0:raise ValueError('parent')
 device=torch.device('cuda');base=build_cifar100_model().to(device);base.load_state_dict(dense);base.eval();layers=eligible_layers(base)
 geom=parent['stepper']['geometry'];mapping=[(n,tuple(s)) for n,s in geom['mapping']];expected=[(n,tuple(p.shape)) for n,p in base.named_parameters()]
 if geom['method']!='vpr' or mapping!=expected:raise ValueError('geometry')
 logs={n:s for (n,_),s in zip(mapping,geom['logs'])};qmap,qmean=normalized_inverse_square({n+'.weight':logs[n+'.weight'] for n in layers})
 trainf=a.dataset_root/'extracted'/'cifar-100-python'/'train'
 with trainf.open('rb') as f:raw=pickle.load(f,encoding='bytes')
 data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64);cal=np.random.default_rng(20261027).permutation(50000)[:1024].tolist();cs=set(cal);diag=[int(x) for x in np.random.default_rng(20261031).permutation(50000) if int(x) not in cs][:1024]
 tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(cfg['dataset']['normalize_mean'],cfg['dataset']['normalize_std'])]);cal_loader=DataLoader(common.CalibrationSubset(data,labels,cal,tf),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);diag_loader=DataLoader(common.CalibrationSubset(data,labels,diag,tf),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
 adapters={};hooks=[]
 for n,l in layers.items():
  ad=SparseGPTAdapter(l,compute_dtype=torch.float32);adapters[n]=ad;hooks.append(l.register_forward_pre_hook(lambda _m,x,item=ad:item.add_batch(x[0])))
 with torch.inference_mode():
  for x,_y,_i in cal_loader:base(x.to(device,non_blocking=True))
 for h in hooks:h.remove()
 hess={n:ad.hessians()[0].clone() for n,ad in adapters.items()}
 def evaluate(model):
  model.eval();correct=count=0;loss=0.
  with torch.inference_mode():
   for x,y,_i in diag_loader:
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);z=model(x);loss+=float(nn.functional.cross_entropy(z,y,reduction='sum'));correct+=int((z.argmax(1)==y).sum());count+=len(y)
  return correct/count,loss/count
 def build(method,alpha):
  model=build_cifar100_model().to(device);model.load_state_dict(dense);model.eval();ms={};layerrows=[];gen=torch.Generator().manual_seed(20261031);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
  for n,l in eligible_layers(model).items():
   mat=weight_to_matrix(l.weight);before=mat.clone();h=hess[n];qmat=weight_to_matrix(qmap[n+'.weight'].float().to(device));qshared=qmat.mean(0)
   if method=='no_comp': rweight=before*weight_to_matrix(orig_masks[n+'.weight'].to(device));mask=weight_to_matrix(orig_masks[n+'.weight'].to(device))
   elif method=='Hfixed':
    rr=fixedmask_shared_q_obs_matrix(before,h,weight_to_matrix(orig_masks[n+'.weight'].to(device)),torch.ones(mat.shape[1],device=device),lambda_value=0.);rweight=rr.weight;mask=rr.keep_mask
   elif method in ('HQ','isotropic','shuffled'):
    qs=torch.ones_like(qshared) if method=='isotropic' else qshared
    if method=='shuffled':qs=qs[torch.randperm(len(qs),generator=gen).to(device)]
    rr=fixedmask_shared_q_obs_matrix(before,h,weight_to_matrix(orig_masks[n+'.weight'].to(device)),qs,lambda_value=alpha);rweight=rr.weight;mask=rr.keep_mask
   elif method in ('QQ','QH'):
    qr=apbr_gpt_matrix(before,h,qmat,lambda_value=alpha,mode='shared_geometry',compute_dtype=torch.float32);mask=qr.keep_mask
    if method=='QQ':rweight=qr.weight
    else:rweight=fixedmask_shared_q_obs_matrix(before,h,mask,torch.ones(mat.shape[1],device=device),lambda_value=0.).weight
   else:
    rr=sparsegpt_matrix(before,h,compute_dtype=torch.float32);rweight=rr.weight;mask=rr.keep_mask
   with torch.no_grad():mat.copy_(rweight)
   ms[n+'.weight']=mask.reshape_as(l.weight).detach().cpu();delta=rweight-before;layerrows.append(dict(method=method,alpha=alpha,layer=n,reconstruction_energy=float(.5*torch.einsum('ri,ij,rj->',delta,h,delta)/delta.shape[0]),compensation_rms=float(delta[mask].square().mean().sqrt()),q_shared_mean=float(qshared.mean()),q_shared_std=float(qshared.std(unbiased=False)),q_shared_min=float(qshared.min()),q_shared_max=float(qshared.max()),q_hdiag_spearman=''))
  validate_masks(model,ms,n=2,m=4,complete=True);torch.cuda.synchronize();prune=time.perf_counter()-started;acc,ce=evaluate(model);payload=dict(schema=1,status='healthy',protocol_id='cifar100_50k10k_final_v1',stage='raw_hard',method=method,alpha=alpha,model={k:v.detach().cpu() for k,v in model.state_dict().items()},masks=ms,parent_sha256=common.sha_file(a.parent),test_accessed=False,diagnostic_train_only=True);d=out/f'{method}_alpha_{alpha:g}';d.mkdir();sha=common.atomic_torch(d/'checkpoint.pt',payload);common.atomic_torch(d/'mask.pt',ms);return dict(method=method,alpha=alpha,diagnostic_accuracy=acc,diagnostic_ce=ce,pruning_seconds=prune,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_sha256=sha,test_accessed=False),layerrows
 metrics=[];details=[]
 for method,alpha in [('no_comp',0.),('Hfixed',0.),('HH',0.)]:
  row,lr=build(method,alpha);metrics.append(row);details+=lr;print(json.dumps(row),flush=True)
 for alpha in ALPHAS:
  for method in METHODS:
   row,lr=build(method,alpha);metrics.append(row);details+=lr;print(json.dumps(row),flush=True)
 csvwrite(out/'metrics_train_diagnostic.csv',metrics);csvwrite(out/'layer_diagnostics.csv',details);common.atomic_json(out/'status.json',dict(status='completed',seed=0,alphas=list(ALPHAS),diagnostic_ids_sha256=hashlib.sha256(np.asarray(diag,dtype='<i8').tobytes()).hexdigest(),diagnostic_is_train_side=True,test_accessed=False,q_global_mean=qmean,full_methods=len(metrics)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if '--output-dir' in sys.argv:
   p=Path(sys.argv[sys.argv.index('--output-dir')+1]);p.mkdir(parents=True,exist_ok=True);common.atomic_json(p/'failure.json',dict(status='failed',type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  raise
