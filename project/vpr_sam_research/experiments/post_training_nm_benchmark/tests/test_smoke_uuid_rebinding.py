"""Only device/config checks; evaluate the actual runner AST without CUDA."""
import ast,json,subprocess
from types import SimpleNamespace
import pytest,yaml
from benchmark.common import sha_file
from benchmark.preflight import ROOT,require_config_hash

HISTORICAL='GPU-fc8784a7-85a2-2bcd-d2e2-f36ef021616c'
OLD='GPU-36255fcb-5df2-3c02-2935-79778f083920'
NEW='GPU-6d21beb3-4922-a03d-55b9-b081fff80cee'

def config():return yaml.safe_load((ROOT/'configs/obc_cuda_smoke.yaml').read_text(encoding='utf-8'))

def actual_guard_rejects(uuid,name):
    tree=ast.parse((ROOT/'benchmark/obc_smoke.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and 'str(props.uuid)' in ast.unparse(n.test))
    return eval(compile(ast.Expression(guard.test),'actual_runner_identity_guard','eval'),
                {'props':SimpleNamespace(uuid=uuid,name=name),'c':config()})

def test_only_uuid_bytes_changed_and_old_config_is_recoverable():
    relative='vpr_sam_research/experiments/post_training_nm_benchmark/configs/obc_cuda_smoke.yaml'
    before=subprocess.check_output(['git','-C',str(ROOT),'show','f642c8a860db9e786843c14072c7be9c389c826c:'+relative])
    after=(ROOT/'configs/obc_cuda_smoke.yaml').read_bytes()
    assert after==before.replace(('gpu_uuid: '+OLD+'\n').encode(),('gpu_uuid: '+NEW+'\ngpu_name: NVIDIA GeForce RTX 4090\n').encode())
    old=yaml.safe_load(before);new=config()
    assert old.pop('gpu_uuid')==OLD and new.pop('gpu_uuid')==NEW
    assert new.pop('gpu_name')=='NVIDIA GeForce RTX 4090' and old==new

@pytest.mark.parametrize('uuid',[NEW,NEW.removeprefix('GPU-')])
def test_current_runner_accepts_only_approved_device_identity(uuid):
    assert actual_guard_rejects(uuid,'NVIDIA GeForce RTX 4090') is False

@pytest.mark.parametrize('uuid,name',[(OLD,'NVIDIA GeForce RTX 4090'),(HISTORICAL,'NVIDIA GeForce RTX 4090'),('GPU-unapproved','NVIDIA GeForce RTX 4090'),(NEW,'NVIDIA GeForce RTX 5090')])
def test_current_runner_rejects_old_other_or_wrong_model(uuid,name):
    assert actual_guard_rejects(uuid,name) is True

def test_new_config_hash_passes_old_hash_fails_and_index_unchanged():
    path=ROOT/'configs/obc_cuda_smoke.yaml'
    expected='f1bd7395e4256ba3971f9f60916344fb5ebf7defb7664db59893243ea5d90e91'
    assert sha_file(path)==expected;require_config_hash(path,expected)
    with pytest.raises(ValueError,match='SHA256'):
        require_config_hash(path,'7eb57f4c63ce58aa2c03ea6fdc92cd07fcc15a75c3c7bbd802f3fa639e6fafc9')
    assert sha_file(ROOT/'configs/obc_smoke_train_ids.json')=='ef5df4e0b96367926d2d4608eed154b401b070239b404b4b77eaa266d8e5ad5e'

def test_only_smoke_manifest_binding_changes_formal_and_history_remain_old():
    m=json.loads((ROOT/'configs/gpu_execution_manifest.json').read_text(encoding='utf-8'))
    assert m['OBC']['smoke_expected_uuid']==NEW and m['expected_uuid']==HISTORICAL
    assert m['OBC']['smoke_expected_gpu_name']=='NVIDIA GeForce RTX 4090'
    for filename in ('historical_protocol.yaml','post_training_nm_benchmark.yaml'):
        assert yaml.safe_load((ROOT/'configs'/filename).read_text(encoding='utf-8'))['gpu_uuid']==HISTORICAL
