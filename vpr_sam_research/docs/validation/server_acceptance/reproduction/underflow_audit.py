import runpy,json,math
from pathlib import Path
import torch
from vpr_sam.optim.geometry import Geometry
m=runpy.run_path('scripts/server_validation.py'); root=Path.cwd(); out=root/'artifacts/server_acceptance'; estimates=json.loads((out/'gamma_estimate.json').read_text()); original=Geometry.perturbations; events=[]; smallest_subnormal=2.0**-149

def inspect(self,rho,*args,**kwargs):
 result=original(self,rho,*args,**kwargs); eps=result[0]
 masks=[((p.grad!=0)&(e==0)) if p.grad is not None else torch.zeros_like(e,dtype=torch.bool) for p,e in zip(self.params,eps)]
 if any(bool(mask.any()) for mask in masks):
  norm=sum(((torch.zeros_like(p) if p.grad is None else p.grad).double()*s.double().exp()).square().sum() for p,s in zip(self.params,self.logs)).sqrt()
  total_eps=sum(e.double().square().sum() for e in eps)
  for (name,_),p,s,mask in zip(self.mapping,self.params,self.logs,masks):
   if not bool(mask.any()):continue
   g=p.grad.double()[mask]; radii=s.double()[mask].exp(); reference=rho*radii.square()*g/norm
   values=reference.abs().cpu().tolist(); all_below_half=all(v<=.5*smallest_subnormal for v in values)
   event={'candidate':tag,'step':self.updates+1,'parameter':name,'count':int(mask.sum()),'gradient_abs':g.abs().cpu().tolist(),'reference_epsilon_abs':values,'missing_epsilon_energy_fraction':float(reference.square().sum()/total_eps),'half_min_fp32_subnormal':.5*smallest_subnormal,'below_half_min_subnormal':all_below_half}
   events.append(event)
   if not all_below_half:raise FloatingPointError('Representable FP32 epsilon unexpectedly zero; stop audit')
 return result
Geometry.perturbations=inspect
checks=[]
for index,gamma in enumerate(estimates['candidates']):
 tag=f'candidate_{index}'
 m['diagnostic_run'](root,out,'cuda:0','vpr','float64',100,gamma,'underflow_audit_'+tag)
 old=torch.load(out/f'vpr_candidate_{index}/diagnostic.pt',map_location='cpu',weights_only=False)
 new=torch.load(out/f'underflow_audit_{tag}/diagnostic.pt',map_location='cpu',weights_only=False)
 exact={field:m['identical'](old[field],new[field]) for field in ('model','stepper','scheduler','source_hashes')}; assert all(exact.values()),exact
 checks.append({'candidate':tag,'exact_replay':exact})
Geometry.perturbations=original
m['atomic_json'](out/'underflow_audit.json',{'status':'PASS','interpretation':'Observed coordinate zeros are FP32 round-to-zero below half minimum subnormal; no complete perturbation underflow or tolerance failure','events':events,'checks':checks,'source_hashes':m['source_hashes'](root)})
print('UNDERFLOW_AUDIT_PASS',len(events),max((r['missing_epsilon_energy_fraction'] for r in events),default=0.),flush=True)
