import copy,json,time
from argparse import Namespace
import pytest,torch,yaml
from benchmark import artifacts
from benchmark.common import tensor_sha
from benchmark.nm import magnitude_mask,project
from benchmark.obc_smoke import micro_recovery,review_errors,run
from benchmark.preflight import ROOT,source_identity
from benchmark.resource_monitor import Monitor
from benchmark.smoke_readiness import statuses
from test_pipeline import Tiny,batches

def configurations():
    return [yaml.safe_load((ROOT/p).read_text(encoding='utf-8')) for p in
        ('configs/post_training_nm_benchmark.yaml','configs/obc_cuda_smoke.yaml')]

def test_unapproved_smoke_exits_before_cuda_or_output_creation(monkeypatch,tmp_path):
    def forbidden():raise AssertionError('Approval gate must precede CUDA')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    _,smoke=configurations();smoke['review']['approved']=False;smoke['calibration']['approved']=False
    path=tmp_path/'unapproved.yaml';path.write_text(yaml.safe_dump(smoke),encoding='utf-8')
    args=Namespace(config=str(path),output=str(tmp_path/'must_not_exist'))
    with pytest.raises(ValueError,match='CALIBRATION'):run(args)
    assert not (tmp_path/'must_not_exist').exists()

def test_methods_have_independent_status_and_patch_is_in_source_identity():
    formal,smoke=configurations();s=statuses(formal,smoke)
    assert s['OBC']['status']=='BLOCKED_CONFIG_REVIEW' and s['SNOWS']['status']=='BLOCKED_SOURCE'
    assert 'patches/snows_minimal_diagnostic.patch' in source_identity()['files']
    # A method's readiness does not authorize or validate the other method.
    import benchmark.smoke_readiness as module
    original=module.review_errors
    try:
        module.review_errors=lambda *args:[]
        s=statuses(formal,smoke,True,True)
        assert s['OBC']['status']=='READY_FOR_GPU_SMOKE' and s['SNOWS']['status']=='BLOCKED_SOURCE'
    finally:module.review_errors=original

def test_new_micro_recovery_is_bounded_and_reloads_without_touching_native(tmp_path):
    formal,_=configurations();native=Tiny();masks={'middle.weight':magnitude_mask(native.middle.weight)}
    project(native,masks);before=tensor_sha(native.state_dict())
    result=micro_recovery(copy.deepcopy(native),masks,batches(),formal['recovery'],2,tmp_path/'smoke.pt',
        {'purpose':'synthetic_cpu_micro_test'},{'native':'synthetic-parent'})
    assert result['steps']==2 and result['reload']=='PASS' and tensor_sha(native.state_dict())==before
    saved=artifacts.load(tmp_path/'smoke.pt');assert saved['completed_steps']==2 and len(saved['train_losses'])==2
    assert 'best' not in saved and saved['selection'].startswith('none')

def test_monitor_records_resources_without_gpu_in_cpu_unit_test(monkeypatch,tmp_path):
    for name in ('memory_allocated','memory_reserved','max_memory_allocated'):monkeypatch.setattr(torch.cuda,name,lambda *args:0)
    with Monitor(tmp_path/'telemetry.jsonl','cpu',0,.01) as monitor:
        time.sleep(.04);monitor.check()
    rows=[json.loads(line) for line in (tmp_path/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows and all(r['disk_free_bytes']>0 and r['cuda_allocated']==0 for r in rows)

def test_monitor_failure_remains_explicit(monkeypatch,tmp_path):
    with pytest.raises(RuntimeError,match='monitor failure'):
        with Monitor(tmp_path/'failed.jsonl','cpu',10**30,.01):time.sleep(.03)

def test_complete_approved_candidate_budget_is_required_and_batches_must_cover_steps(tmp_path):
    from benchmark.calibration import index_sha
    formal,smoke=configurations()
    assert review_errors(smoke,formal)==[]
    smoke['calibration'].update(batch_size=32,forward_passes_per_layer=1)
    assert any('batches' in e for e in review_errors(smoke,formal))
