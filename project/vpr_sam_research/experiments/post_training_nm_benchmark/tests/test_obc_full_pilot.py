import copy,json,random,ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch,yaml
from PIL import Image
from benchmark.calibration import approved_indices
from benchmark.obc_calibration import factory,CalibrationImages
from benchmark.preflight import ROOT,source_identity
from benchmark.smoke_lineage import verify
from benchmark.common import sha_file

def config():return yaml.safe_load((ROOT/'configs/obc_seed0_4090_pilot.yaml').read_text(encoding='utf-8'))

def test_actual_official_defaults_and_frozen_indices():
    from benchmark.upstream import source
    main=ast.parse(source('OBC','main_trueobs.py'));data=ast.parse(source('OBC','datautils.py'))
    options={n.args[0].value:next((ast.literal_eval(k.value) for k in n.keywords if k.arg=='default'),None)
             for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=='add_argument' and isinstance(n.args[0],ast.Constant)}
    assert options['--batchsize']==-1 and options['--nrounds']==-1 and options['--nsamples']==1024
    text=ast.unparse(data);maintext=ast.unparse(main)
    assert 'batchsize = 128' in text and 'shuffle=True' in text
    assert "else 10" in maintext and 'args.nrounds = 1' in maintext
    c=config();ids=approved_indices(c);expected=np.arange(50000);np.random.RandomState(0).shuffle(expected)
    assert ids==expected[:1024].tolist()
    assert sha_file(c['calibration']['index_file'])==c['calibration']['index_file_sha256']
    assert c['calibration']['forward_passes_per_layer']==80 and c['seeds']==[0]
    assert c['common_bn']['enabled'] is False and c['recovery']['epochs']==20

@pytest.mark.parametrize('field,value',[('batch_size',32),('rounds',1),('augmentation','unknown')])
def test_calibration_budget_tampering_rejected(field,value):
    c=config();c['calibration'][field]=value
    with pytest.raises(ValueError):approved_indices(c)

class Images:
    def __len__(self):return 8
    def __getitem__(self,i):
        # Spatial texture distinguishes newly generated crop/flip views.
        return Image.fromarray(np.random.RandomState(i).randint(0,256,(32,32,3),dtype=np.uint8)),i%2

def test_ten_rounds_fresh_views_exact_layer_replay_and_rng_isolation():
    c=config();data=dict(c['data'],pin_memory=False);cal=dict(c['calibration'],batch_size=4)
    records=[];make=factory(Images(),list(range(8)),data,cal,receipt=records)
    random.seed(31);np.random.seed(31);torch.manual_seed(31)
    saved=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    first=list(make());second=list(make())
    assert len(first)==20 and records[0]==records[1] and records[0]['samples_including_rounds']==80
    assert all(torch.equal(a[0],b[0]) and torch.equal(a[2],b[2]) for a,b in zip(first,second))
    assert not torch.equal(torch.cat([x[0] for x in first[:2]]),torch.cat([x[0] for x in first[2:4]]))
    assert random.getstate()==saved[0] and np.array_equal(np.random.get_state()[1],saved[1][1])
    assert torch.equal(torch.get_rng_state(),saved[2])
    ds=CalibrationImages(Images(),list(range(8)),data)
    assert [type(x).__name__ for x in ds.transform.transforms]==['RandomResizedCrop','RandomHorizontalFlip','ToTensor','Normalize']
    assert ds[0][0].shape==(3,32,32)

def staged_config(tmp_path):
    c=config();p=ROOT/'artifacts/server4090_transfer_20261025_1792933192612411200'
    for name in ('report','audit'):
        actual=p/('SMOKE_REPORT.json' if name=='report' else 'SMOKE_FINAL_AUDIT.json')
        dest=tmp_path/actual.name;dest.write_bytes(actual.read_bytes());c['execution']['smoke_lineage'][name]=str(dest)
    return c

def test_real_smoke_lineage_preserves_tested_numerical_core(tmp_path):
    assert verify(staged_config(tmp_path),source_identity())['status']=='PASS'

@pytest.mark.parametrize('case',['core','device','bytes'])
def test_smoke_lineage_rejects_stale_core_wrong_device_and_changed_evidence(tmp_path,case):
    c=staged_config(tmp_path);src=copy.deepcopy(source_identity())
    if case=='core':src['files']['benchmark/obc.py']='changed'
    elif case=='device':c['gpu_uuid']='wrong'
    else:Path(c['execution']['smoke_lineage']['report']).write_text('{}')
    with pytest.raises(ValueError):verify(c,src)

def test_actual_full_gpu_model_guard_accepts4090_rejects5090():
    tree=ast.parse((ROOT/'benchmark/preflight.py').read_text(encoding='utf-8'))
    guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'props.name' in ast.unparse(n.test))
    def rejected(name):return eval(compile(ast.Expression(guard.test),'actual_guard','eval'),{'props':SimpleNamespace(name=name),'config':config()})
    assert not rejected('NVIDIA GeForce RTX 4090') and rejected('NVIDIA GeForce RTX 5090')

def test_index_bytes_and_smoke_config_tampering_rejected(tmp_path):
    c=config();p=tmp_path/'ids.json'
    p.write_text(json.dumps(approved_indices(c)))
    c['calibration']['index_file']=str(p)
    with pytest.raises(ValueError,match='byte identity'):approved_indices(c)
    c=staged_config(tmp_path);src=copy.deepcopy(source_identity())
    src['files']['configs/obc_cuda_smoke.yaml']='changed'
    with pytest.raises(ValueError,match='configuration changed'):verify(c,src)

def test_recovery_bn_skip_does_not_change_dense_or_optimization_core():
    import subprocess
    base='vpr_sam_research/experiments/post_training_nm_benchmark/'
    for name in ('obc.py','pipeline.py','recovery.py','evaluation.py','model.py','nm.py'):
        old=subprocess.check_output(['git','-C',str(ROOT),'show','19ff5dc9f3d1720cbf434c388741d8138b09a948:'+base+'benchmark/'+name])
        assert (ROOT/'benchmark'/name).read_bytes()==old
    tree=ast.parse((ROOT/'benchmark/run_obc.py').read_text(encoding='utf-8'))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and "get('enabled', True)" in ast.unparse(n.test))
    assert 'SKIPPED_BY_USER' in ast.unparse(ast.Module(body=branch.orelse,type_ignores=[]))
    assert 'common_bn.pt' not in ast.unparse(ast.Module(body=branch.orelse,type_ignores=[]))
