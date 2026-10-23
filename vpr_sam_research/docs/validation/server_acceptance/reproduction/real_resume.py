import copy,json,math,runpy
from pathlib import Path
from vpr_sam.training import engine
m=runpy.run_path('scripts/server_validation.py'); root=Path.cwd(); out=root/'artifacts/server_acceptance'; gamma=json.loads((out/'gamma_estimate.json').read_text())['candidates'][0]; results=[]
for method in ('sam','vpr'):
 c=m['common'](root,'cuda:0',method,real=True); c['geometry']['gamma']=gamma; c['output']='runs/real_resume_acceptance/'+method+'_continuous'
 target=math.ceil(45000/c['training']['batch_size'])+1
 m['train'](c,root,stop_after=target); expected,_=m['read_checkpoint'](c['output']+'/last.pt',root)
 print('PASS real continuous',method,target,flush=True)
 for boundary in ('batch','epoch'):
  split=copy.deepcopy(c); split['output']='runs/real_resume_acceptance/'+method+'_'+boundary
  if boundary=='batch':m['train'](split,root,stop_after=1)
  else:
   save=engine.atomic_save
   class BoundaryStop(Exception):pass
   def save_boundary(path,payload):
    save(path,payload)
    if path.name=='last.pt' and payload['epoch']==1 and payload['next_batch']==0:raise BoundaryStop()
   engine.atomic_save=save_boundary
   try:
    try:m['train'](split,root)
    except BoundaryStop:pass
   finally:engine.atomic_save=save
  m['train'](split,root,resume=split['output']+'/last.pt',stop_after=target)
  actual,_=m['read_checkpoint'](split['output']+'/last.pt',root)
  fields=('model','stepper','scheduler','rng','order_hash','epoch','next_batch','accumulator','epoch_rows','initialization_hash')
  checks={key:m['identical'](expected[key],actual[key]) for key in fields}; assert all(checks.values()),checks
  rows=lambda s:[{k:v for k,v in row.items() if k!='step_seconds'} for row in s['step_rows']]
  assert m['identical'](rows(expected),rows(actual))
  results.append({'method':method,'boundary':boundary,'workers':2,'steps':target,'checks':checks,'step_history_excluding_time_exact':True,'config':split})
  print('PASS real exact',method,boundary,target,flush=True)
 atomic=m['atomic_json']; atomic(out/'real_resume_progress.json',{'completed_method':method,'results':results})
m['atomic_json'](out/'real_resume.json',{'status':'PASS','dataset':'official CIFAR10','train_size':45000,'validation_size':5000,'validation_used_for_gamma':False,'gamma':gamma,'cosine_horizon_epochs':2,'results':results,'source_hashes':m['source_hashes'](root)})
