"""GPU-only preregistered APBR-GPT pilot on layer1.0.conv1."""
from __future__ import annotations
import argparse,csv,hashlib,json,pickle,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_apbr_weights_original_sparsegpt_one as common
from baseline_adapters.apbr_gpt import apbr_gpt_matrix,normalized_inverse_square
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter,sparsegpt_matrix
from benchmark.nm import eligible_layers

LAYER='layer1.0.conv1'; ALPHAS=(0.,1e-4,1e-3,1e-2,1e-1,1.); PERCDAMP=.01

def write_csv(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  fields=[]
  for row in rows:
   for key in row:
    if key not in fields:fields.append(key)
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def energy(delta,h): return float(.5*torch.einsum('ri,ij,rj->',delta,h,delta)/delta.shape[0])

@torch.no_grad()
def fixed(weight,keep,hraw,q,alpha,kind):
 damp=PERCDAMP*torch.diag(hraw).mean(); hbase=hraw.clone();hbase.diagonal().add_(damp);mu=torch.diag(hraw).mean()
 out=weight.clone();out[~keep]=0; mx=sm=0.; gen=torch.Generator().manual_seed(20261031)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
 for row in range(weight.shape[0]):
  k=torch.where(keep[row])[0];d=torch.where(~keep[row])[0];a=hbase[k][:,k].clone();rhs=hbase[k][:,d]@weight[row,d]
  if kind=='isotropic': pen=torch.ones(len(k),device=weight.device)
  else:
   pen=q[row,k]
   if kind=='shuffled':pen=pen[torch.randperm(len(k),generator=gen).to(weight.device)]
  if alpha:a.diagonal().add_(alpha*mu*pen)
  chol,info=torch.linalg.cholesky_ex(a)
  if int(info):raise RuntimeError(f'cholesky row {row} {kind} {alpha}')
  dk=torch.cholesky_solve(rhs[:,None],chol).squeeze(1);res=torch.linalg.vector_norm(a@dk-rhs)/(torch.linalg.vector_norm(rhs)+1e-12)
  out[row,k]=weight[row,k]+dk;out[row,d]=0;mx=max(mx,float(res));sm+=float(res)
 torch.cuda.synchronize()
 return out,dict(runtime_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),max_relative_residual=mx,mean_relative_residual=sm/weight.shape[0],damping=float(damp),mu_h=float(mu))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--mask',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError(out)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 cfg=yaml.safe_load(a.config.read_text());parent=torch.load(a.parent,map_location='cpu',weights_only=False)
 if parent.get('epoch')!=200 or parent.get('steps')!=78200 or parent.get('identity',{}).get('method')!='apbr_sam' or parent.get('identity',{}).get('seed')!=0:raise ValueError('parent')
 model=build_cifar100_model().cuda();model.load_state_dict(parent['model']);model.eval();layers=eligible_layers(model);layer=layers[LAYER]
 geom=parent['stepper']['geometry'];mapping=[(n,tuple(s)) for n,s in geom['mapping']];expected=[(n,tuple(p.shape)) for n,p in model.named_parameters()]
 if geom['method']!='vpr' or mapping!=expected:raise ValueError('geometry')
 logs={n:s for (n,_),s in zip(mapping,geom['logs'])}; eligible={n+'.weight':logs[n+'.weight'] for n in layers};qmap,qmean=normalized_inverse_square(eligible)
 q=weight_to_matrix(qmap[LAYER+'.weight'].float().cuda());weight=weight_to_matrix(layer.weight).float().detach().clone();masks=torch.load(a.mask,map_location='cpu',weights_only=False);keep=weight_to_matrix(masks[LAYER+'.weight'].cuda()).bool()
 trainf=a.dataset_root/'extracted'/'cifar-100-python'/'train'
 with trainf.open('rb') as f:raw=pickle.load(f,encoding='bytes')
 data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64);cal=np.random.default_rng(20261027).permutation(50000)[:1024].tolist();cs=set(cal);diag=[int(x) for x in np.random.default_rng(20261031).permutation(50000) if int(x) not in cs][:1024]
 tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(cfg['dataset']['normalize_mean'],cfg['dataset']['normalize_std'])])
 def collect(ids):
  ad=SparseGPTAdapter(layer,compute_dtype=torch.float32);hook=layer.register_forward_pre_hook(lambda _m,x:ad.add_batch(x[0]));loader=DataLoader(common.CalibrationSubset(data,labels,ids,tf),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
  with torch.inference_mode():
   for x,_y,_i in loader:model(x.cuda(non_blocking=True))
  hook.remove();return ad.hessians()[0].clone()
 hcal=collect(cal);hdiag=collect(diag);rows=[]
 maskonly=weight.clone();maskonly[~keep]=0;delta=maskonly-weight;rows.append(dict(family='fixed',method='no_compensation',alpha=0.,calibration_mse=energy(delta,hcal),diagnostic_mse=energy(delta,hdiag),runtime_seconds=0.,peak_allocated_bytes=0,peak_reserved_bytes=0,max_relative_residual=0.,mask_flip_rate=0.))
 standard,st=fixed(weight,keep,hcal,q,0.,'isotropic');delta=standard-weight;rows.append(dict(family='fixed',method='standard',alpha=0.,calibration_mse=energy(delta,hcal),diagnostic_mse=energy(delta,hdiag),mask_flip_rate=0.,**st))
 for alpha in ALPHAS:
  for kind in ('isotropic','geometry','shuffled'):
   solved,st=fixed(weight,keep,hcal,q,alpha,kind)
   if alpha==0:torch.testing.assert_close(solved,standard,rtol=0,atol=0)
   delta=solved-weight;rows.append(dict(family='fixed',method=kind,alpha=alpha,calibration_mse=energy(delta,hcal),diagnostic_mse=energy(delta,hdiag),mask_flip_rate=0.,**st))
 original=sparsegpt_matrix(weight,hcal,blocksize=128,percdamp=.01,compute_dtype=torch.float32)
 for alpha in ALPHAS:
  for mode in ('exact_rowwise','shared_geometry'):
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();r=apbr_gpt_matrix(weight,hcal,q,lambda_value=alpha,mode=mode,blocksize=128,percdamp=.01,compute_dtype=torch.float32);torch.cuda.synchronize();elapsed=time.perf_counter()-t;delta=r.weight-weight
   if alpha==0:
    assert torch.equal(r.keep_mask,original.keep_mask);torch.testing.assert_close(r.weight,original.weight,rtol=0,atol=0)
   rows.append(dict(family='dynamic',method=mode,alpha=alpha,calibration_mse=energy(delta,hcal),diagnostic_mse=energy(delta,hdiag),runtime_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),max_relative_residual='',mask_flip_rate=float((r.keep_mask!=original.keep_mask).float().mean()),damping=r.original_damp,mu_h=r.mu_h,mean_relative_residual=''))
 write_csv(out/'layer_pilot_results.csv',rows);write_csv(out/'runtime_memory.csv',rows)
 common.atomic_json(out/'status.json',dict(status='completed',layer=LAYER,weight_shape=list(layer.weight.shape),matrix_shape=list(weight.shape),q_global_mean=qmean,alphas=list(ALPHAS),test_accessed=False,full_model_started=False,parent_sha256=common.sha_file(a.parent),mask_sha256=common.sha_file(a.mask)))
 print(json.dumps(json.loads((out/'status.json').read_text())))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if '--output-dir' in sys.argv:
   p=Path(sys.argv[sys.argv.index('--output-dir')+1]);p.mkdir(parents=True,exist_ok=True);common.atomic_json(p/'failure.json',dict(status='failed',type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  raise
