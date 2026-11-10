import pytest,torch
from benchmark.pipeline import prune_obc
from benchmark.common import tensor_sha
from benchmark import artifacts
from test_pipeline import Tiny,batches

def test_layer_boundary_resume_retains_native_compensation_and_parent(tmp_path):
    dense=Tiny().eval();before=tensor_sha(dense.state_dict());identity={'source':'synthetic'};parent={'sha':'dense'}
    out,masks,receipts=prune_obc(dense,batches,tmp_path,identity,parent,parallel=2)
    assert before==tensor_sha(dense.state_dict()) and len(receipts)==1
    assert receipts[0]['samples_including_rounds']==12
    def forbidden():raise RuntimeError('Finished layer should not be recalibrated')
    resumed,new_masks,_=prune_obc(dense,forbidden,tmp_path,identity,parent,parallel=2,resume=True)
    assert tensor_sha(out.state_dict())==tensor_sha(resumed.state_dict())
    assert torch.equal(masks['middle.weight'],new_masks['middle.weight'])
    assert artifacts.load(tmp_path/'native_complete.pt')['stage']=='native_complete'
    with pytest.raises(ValueError,match='parent'):
        prune_obc(dense,batches,tmp_path,identity,{'sha':'different'},resume=True)
