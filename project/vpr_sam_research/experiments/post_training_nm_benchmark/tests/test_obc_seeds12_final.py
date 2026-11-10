import copy,importlib.util,json,subprocess
from pathlib import Path
import pytest,yaml
from benchmark.preflight import ROOT
from benchmark.calibration import approved_indices
from benchmark.common import sha_file

spec=importlib.util.spec_from_file_location('queue',ROOT/'scripts/run_obc_seeds12_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)

def configs():
    return [yaml.safe_load((ROOT/'configs'/name).read_text(encoding='utf-8')) for name in ['obc_seed0_4090_pilot.yaml','obc_seeds12_4090_final.yaml']]

def test_only_authorized_seed_registration_and_final_reporting_change():
    old,new=configs()
    assert queue.protocol_equal(old,new) and new['seeds']==[1,2]
    assert new['reporting']['primary_checkpoint']=='last.pt' and new['reporting']['primary_recovery_epoch']==20
    assert new['recovery']==old['recovery'] and new['common_bn']['enabled'] is False
    assert approved_indices(old)==approved_indices(new) and new['calibration']['random_seed']==0

@pytest.mark.parametrize('field',['calibration','data','recovery','pruning'])
def test_protocol_gate_rejects_unapproved_scientific_change(field):
    old,new=configs();new[field]['unapproved_change']=True
    assert not queue.protocol_equal(old,new)

def test_seed0_audit_reused_and_all_existing_implementation_bytes_unchanged():
    path=ROOT/'artifacts/obc_full_seed0_prepare_20261025/FINAL_READONLY_AUDIT.json'
    audit=json.loads(path.read_text(encoding='utf-8'))
    assert audit['status']=='PASS' and all(x['pass'] for x in audit['checks'].values())
    prefix='vpr_sam_research/experiments/post_training_nm_benchmark/'
    for file in (ROOT/'benchmark').glob('*.py'):
        if file.name=='obc_final_audit.py':continue
        old=subprocess.check_output(['git','-C',str(ROOT),'show','6c4e4d08b54813a07ba181a1f061c9690f6137b7:'+prefix+'benchmark/'+file.name])
        assert file.read_bytes()==old

def test_parent_records_cover_distinct_seeds_and_exact_local_byte_hashes():
    records=json.loads((ROOT/'configs/existing_inputs.json').read_text(encoding='utf-8'))['checkpoints']
    base=Path('C:/Users/earthgrass73/Desktop/NM稀疏/VPR-v2-unified50k/vpr_sam_research/experiments/vpr_v2_seeds12/artifacts/delivery_20261025_1021')
    parents=[next(x for x in records if x['seed']==seed and x['role']=='standard_dense_sam') for seed in (1,2)]
    assert parents[0]['sha256']!=parents[1]['sha256']
    for parent in parents:assert sha_file(base/parent['relative_path'])==parent['sha256']
