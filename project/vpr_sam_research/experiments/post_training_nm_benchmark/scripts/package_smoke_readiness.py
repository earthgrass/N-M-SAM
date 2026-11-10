"""New continuation evidence only. Keep all previous audit/experiment files intact."""
import importlib.metadata,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,sha_file
from benchmark.preflight import source_identity

def main():
    original=ROOT/'artifacts/smoke_readiness_20261025'
    target=ROOT/'docs/research/evidence/smoke_readiness_20261025';target.mkdir(exist_ok=False)
    cases={};batches={}
    for name in ('snows_repair_tests','snows_repair_retry_tests','obc_smoke_cpu_tests','smoke_positive_gate_tests'):
        tree=ET.parse(original/(name+'.xml'));suites=tree.getroot().findall('testsuite')
        batches[name]={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        for case in tree.findall('.//testcase'):
            cases[case.attrib['classname']+'::'+case.attrib['name']]='PASS' if case.find('failure') is None and case.find('error') is None else 'FAIL'
    if any(v!='PASS' for v in cases.values()):raise ValueError('Some new cases have not passed their targeted retry')
    old=json.loads((ROOT/'docs/research/evidence/acceptance_state.json').read_text(encoding='utf-8'))
    current=source_identity();changed=[p for p,h in old['source']['files'].items() if current['files'].get(p)!=h]
    added=sorted(set(current['files'])-set(old['source']['files']))
    core=['benchmark/obc.py','benchmark/pipeline.py','benchmark/nm.py','benchmark/model.py','benchmark/data.py','benchmark/evaluation.py','benchmark/recovery.py']
    if any(p in changed for p in core):raise ValueError('Previously accepted core changed; affected tests would need review')
    for path in original.iterdir():
        if path.is_file():shutil.copyfile(path,target/path.name)
    joblib=[d.version for d in importlib.metadata.distributions(path=[str(ROOT/'artifacts/python_deps')]) if d.metadata.get('Name','').lower()=='joblib']
    report={'status':'OFFLINE_PREPARATION_ONLY_GPU_NOT_READY','baseline_commit':'551c891a69dcd9631ec6a6ba16c8562ab54f82e4',
        'baseline_source_matched_before_edit':True,'prior_core_unchanged':core,'changed_prior_source_files':changed,
        'added_source_files':added,'source':current,'test_batches':batches,'latest_distinct_new_test_cases':cases,
        'distinct_new_cases':len(cases),'initial_failures_preserved':3,
        'failure_resolution':['joblib absent for2tests; installed locally into ignored artifacts/python_deps',
                              'FX optional label argument specialized to None for1test; model unchanged'],
        'joblib_isolated_version':joblib,'formal_calibration':'PENDING_APPROVAL','smoke_calibration':'PENDING_APPROVAL',
        'OBC':'BLOCKED_CALIBRATION_APPROVAL; config review also pending','SNOWS':'BLOCKED_SOURCE',
        'patch_sha256':sha_file(ROOT/'patches/snows_minimal_diagnostic.patch'),
        'formal_config_sha256':sha_file(ROOT/'configs/post_training_nm_benchmark.yaml'),
        'smoke_config_sha256':sha_file(ROOT/'configs/obc_cuda_smoke.yaml'),
        'GPU_tests_run':False,'server_contacted':False,'historical_experiments_repeated':False,
        'original_official_checkout_modified':False,'full_SN0WS_solver_accepted':False,
        'review_required':['separate smoke calibration indices/batch/rounds/augmentation/preprocessing',
                           'smoke config and source freeze','SNOWS solver step acceptance and actual CIFAR18 representation bridge']}
    atomic_json(target/'READINESS_EVIDENCE.json',report)
    atomic_json(target/'SHA256SUMS.json',{p.name:sha_file(p) for p in sorted(target.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps({'new_cases':len(cases),'all_latest_cases':'PASS','states':{'OBC':report['OBC'],'SNOWS':report['SNOWS']},
                      'original_core_unchanged':True,'initial_failures_preserved':3}))

if __name__=='__main__':main()
