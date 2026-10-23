import sys,time,runpy,statistics
from pathlib import Path
import torch
from vpr_sam.training import engine
from vpr_sam.repro import atomic_json,source_hashes
root=Path.cwd(); m=runpy.run_path('scripts/server_validation.py'); method=sys.argv[1]; c=m['common'](root,'cuda:0',method,real=True); c['output']='runs/server_uninstrumented/'+method
before=source_hashes(root); timings={'state_capture':[],'seal':[],'serialize':[]}
original_state=engine.Stepper.state_dict; original_seal=engine.seal_artifact; original_save=engine.atomic_save

def capture(self):
 t=time.perf_counter(); result=original_state(self); timings['state_capture'].append(time.perf_counter()-t); return result

def seal(payload):
 t=time.perf_counter(); result=original_seal(payload); timings['seal'].append(time.perf_counter()-t); return result

def save(path,payload):
 t=time.perf_counter(); result=original_save(path,payload); timings['serialize'].append(time.perf_counter()-t); return result
engine.Stepper.state_dict=capture; engine.seal_artifact=seal; engine.atomic_save=save
try: engine.train(c,root,stop_after=12)
finally: engine.Stepper.state_dict=original_state; engine.seal_artifact=original_seal; engine.atomic_save=original_save
peak=torch.cuda.max_memory_allocated('cuda:0'); checkpoint=root/c['output']/'last.pt'; state,_=engine.read_checkpoint(checkpoint,root)
assert source_hashes(root)==before
result={'method':method,'dtype':'float64','schema':state['schema'],'steps':state['stepper']['steps'],'warmup_discarded':2,'mean_step_seconds':statistics.mean(r['step_seconds'] for r in state['step_rows'][2:]),'step_seconds':[r['step_seconds'] for r in state['step_rows']],'peak_cuda_bytes':peak,'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_component_seconds':timings,'initialization_hash':state['initialization_hash'],'order_hash':state['order_hash'],'source_hashes':before,'scope':'Unmodified engine step timer; checkpoint capture/seal/serialization timed separately; no per-step observers'}
atomic_json(root/'artifacts/server_acceptance'/('uninstrumented_'+method+'.json'),result)
print('PASS uninstrumented',method,result['mean_step_seconds'],flush=True)
