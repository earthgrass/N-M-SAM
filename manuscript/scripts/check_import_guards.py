"""Synthetic metadata fixtures test the importer, never an experiment/model/Test run."""
from pathlib import Path
import tempfile,json,copy
from import_final_results import validate,sha,EXPECTED,ROOT
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
records=[]
with tempfile.TemporaryDirectory() as temp:
    run=Path(temp);ev=run/'evaluation';ev.mkdir();(run/'training').mkdir()
    parent=run/'training/final.pt';parent.write_bytes(b'SYNTHETIC IMPORTER TEST FIXTURE; NOT MODEL WEIGHTS')
    report={'status':'SEED0_EXPLORATORY_COMPLETE','seed':0,'completed_epochs':200,'steps':78200,'recovery_epochs':0,'performance_gate':'FAIL','final_checkpoint_sha256':sha(parent),'results':{}}
    manifest={'status':'ALL_ENDPOINTS_FROZEN','test_accessed':False,'seed':0,'config_sha256':EXPECTED,'parent_sha256':sha(parent),'endpoints':{}}
    for name in ['dense','final_q_mask_only','final_q_fixed_obs','sparsegpt_native']:
        ck=ev/f'{name}.pt';ck.write_bytes(('SYNTHETIC:'+name).encode())
        item={'checkpoint_path':f'/fixture/{name}.pt','checkpoint_sha256':sha(ck),'mask_sha256':'a'*64}
        manifest['endpoints'][name]=item
        result={**item,'parent_sha256':sha(parent),'config_sha256':EXPECTED,'samples':10000,'accuracy_percent':50.,'ce':1.}
        report['results'][name]=result;dump(ev/f'{name}_evaluation.json',result)
    def reset():
        dump(run/'SEED0_EVIDENCE_REPORT.json',report);dump(ev/'all_endpoints_frozen.json',manifest)
        dump(ev/'test_started.json',{'frozen_manifest_sha256':sha(ev/'all_endpoints_frozen.json')})
    reset();validate(run);records.append({'check':'well_formed_fixture','status':'PASS'})
    for name,mutator in [
        ('reject_incomplete',lambda r:r.update(completed_epochs=199)),
        ('reject_wrong_seed',lambda r:r.update(seed=1)),
        ('reject_recovery',lambda r:r.update(recovery_epochs=20)),
        ('retain_performance_failure',lambda r:r.update(performance_gate='PASS'))]:
        reset();r=copy.deepcopy(report);mutator(r);dump(run/'SEED0_EVIDENCE_REPORT.json',r)
        try:validate(run)
        except ValueError:records.append({'check':name,'status':'PASS'})
        else:raise AssertionError(name)
    reset();(ev/'sparsegpt_native.pt').write_bytes(b'tampered')
    try:validate(run)
    except ValueError:records.append({'check':'reject_modified_checkpoint','status':'PASS'})
    else:raise AssertionError('hash guard')
(ROOT/'review/import_guard_checks.json').write_text(json.dumps({'scope':'synthetic metadata guard tests only; no actual evaluation imported','checks':records},indent=2)+'\n')
print(f'{len(records)} importer guard checks PASS')
