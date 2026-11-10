import copy,json
import pytest,torch,yaml
from PIL import Image
from torchvision import transforms
from benchmark import artifacts
from benchmark.common import sha_file,tensor_sha
from benchmark.data import Images
from benchmark.obc_smoke import micro_recovery
from benchmark.preflight import ROOT
from benchmark.smoke_provenance import verify
from benchmark.smoke_readiness import statuses
from benchmark.nm import magnitude_mask,project
from test_pipeline import Tiny,batches

def config():
    return [yaml.safe_load((ROOT/p).read_text(encoding='utf-8')) for p in
            ('configs/post_training_nm_benchmark.yaml','configs/obc_cuda_smoke.yaml')]

def test_frozen_ids_parent_normalize_file_order_and_ready_without_cuda_claim():
    formal,c=config();proof=verify(c,formal)
    assert len(proof['ids'])==len(set(proof['ids']))==32 and all(0<=n<50000 for n in proof['ids'])
    assert proof['training_file_order']==[f'data_batch_{i}' for i in range(1,6)]
    assert all(record['file']==f'data_batch_{record["dataset_index"]//10000+1}' and record['row_zero_based']==record['dataset_index']%10000 for record in proof['sample_identity'])
    pipeline=Images([(Image.new('RGB',(32,32)),0)],formal['data'],False)
    assert [type(t) for t in pipeline.transform.transforms]==[transforms.ToTensor,transforms.Normalize]
    normalizer=pipeline.transform.transforms[-1]
    assert list(normalizer.mean)==proof['parent_normalization']['normalize_mean']
    assert list(normalizer.std)==proof['parent_normalization']['normalize_std']
    s=statuses(formal,c,True,True);assert s['OBC']['status']=='READY_FOR_GPU_SMOKE' and s['OBC']['cuda_acceptance']=='NOT_RUN'
    assert formal['calibration']['approved'] is False and formal['execution']['approved'] is False

@pytest.mark.parametrize('change',['index_bytes','precision','preprocessing'])
def test_frozen_smoke_rejects_unauthorized_changes(tmp_path,change):
    formal,c=config()
    if change=='index_bytes':
        p=tmp_path/'ids.json';p.write_bytes((ROOT/c['calibration']['index_file']).read_bytes()+b'\n')
        c['calibration']['index_file']=str(p)
    elif change=='precision':c['precision']['amp']=True
    else:c['calibration']['preprocessing']['normalize_std']=[1.,1.,1.]
    with pytest.raises(ValueError):verify(c,formal)

def test_smoke_reload_detects_changed_optimizer_state(tmp_path,monkeypatch):
    formal,_=config();model=Tiny();masks={'middle.weight':magnitude_mask(model.middle.weight)};project(model,masks)
    original=torch.optim.SGD.load_state_dict
    def corrupt(optimizer,state):
        result=original(optimizer,state)
        next(iter(optimizer.state.values()))['momentum_buffer'].add_(1)
        return result
    monkeypatch.setattr(torch.optim.SGD,'load_state_dict',corrupt)
    with pytest.raises(ValueError,match='optimizer tensor reload'):
        micro_recovery(model,masks,batches(),formal['recovery'],2,tmp_path/'bad_reload.pt',{'purpose':'CPU_test'},{'stage':'synthetic'})
