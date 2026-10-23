from pathlib import Path
import runpy,shutil
m=runpy.run_path('scripts/server_validation.py'); root=Path.cwd(); results=[]
for method in ('sam','vpr'):
 directory=root/'runs/server_acceptance'/f'{method}_continuous'
 parent=next((directory/'pruning').glob('*/pruned.pt'))
 expected,_=m['read_checkpoint'](parent.parent/'recovery/final.pt',root)
 clone=root/'runs/server_acceptance'/f'{method}_recovery_resume'; clone.mkdir(); shutil.copy2(parent,clone/'pruned.pt')
 state,_=m['read_checkpoint'](clone/'pruned.pt',root)
 m['train'](state['config'],root,recover_from=clone/'pruned.pt',stop_after=1)
 m['train'](state['config'],root,recover_from=clone/'pruned.pt',resume=clone/'recovery/last.pt')
 actual,_=m['read_checkpoint'](clone/'recovery/final.pt',root)
 fields=('model','stepper','scheduler','rng','order_hash','masks','epoch','next_batch','epoch_rows')
 checks={key:m['identical'](expected[key],actual[key]) for key in fields}; assert all(checks.values()),checks
 results.append({'method':method,'workers':2,'checks':checks}); print('PASS recovery exact',method,flush=True)
m['atomic_json'](root/'artifacts/server_acceptance/recovery_resume.json',{'status':'PASS','source_hashes':m['source_hashes'](root),'results':results})
