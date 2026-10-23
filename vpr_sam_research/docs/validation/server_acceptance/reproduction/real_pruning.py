import runpy,shutil,json
from pathlib import Path
m=runpy.run_path('scripts/server_validation.py'); root=Path.cwd(); results=[]
for method in ('sam','vpr'):
 source=f'runs/server_acceptance/{method}_real/last.pt'
 hard=m['prune'](source,root); state,parent=m['read_checkpoint'](hard['checkpoint'],root)
 recovery=m['train'](state['config'],root,recover_from=parent)
 expected,_=m['read_checkpoint'](parent.parent/'recovery/final.pt',root)
 clone=root/'runs/real_pruning_resume'/method; clone.mkdir(parents=True); shutil.copy2(parent,clone/'pruned.pt')
 m['train'](state['config'],root,recover_from=clone/'pruned.pt',stop_after=1)
 m['train'](state['config'],root,recover_from=clone/'pruned.pt',resume=clone/'recovery/last.pt')
 actual,_=m['read_checkpoint'](clone/'recovery/final.pt',root)
 fields=('model','stepper','scheduler','rng','order_hash','masks','epoch','next_batch','epoch_rows')
 checks={k:m['identical'](expected[k],actual[k]) for k in fields}; assert all(checks.values()),checks
 for name,mask in actual['masks'].items():assert not actual['model'][name][~mask.bool()].count_nonzero()
 results.append({'method':method,'dense_source':source,'dense_smoke_steps':3,'pruning':hard,'recovery':recovery,'recovery_resume_checks':checks,'checkpoint_schema':actual['schema']})
 print('PASS REAL CIFAR prune/recover/exact resume',method,flush=True)
assert results[0]['recovery']['steps']==results[1]['recovery']['steps']
assert results[0]['recovery']['order_hash']==results[1]['recovery']['order_hash']
m['atomic_json'](root/'artifacts/server_acceptance/real_pruning.json',{'status':'PASS','purpose':'engineering smoke only; 1:3 fixture','recovery_budget_epochs':1,'test_used':False,'results':results,'independent_seeds':1,'confidence_interval':None,'source_hashes':m['source_hashes'](root)})
