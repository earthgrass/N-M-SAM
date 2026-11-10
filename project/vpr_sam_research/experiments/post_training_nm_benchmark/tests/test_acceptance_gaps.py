"""Only additional pre-execution acceptance coverage, all synthetic CPU."""
import copy,json,random
import numpy as np
import pytest,torch,yaml
from torch import nn
from benchmark import artifacts
from benchmark.calibration import approved_indices,index_sha
from benchmark.common import rng_state,restore_rng,tensor_sha,sha_file
from benchmark.evaluation import recalibrate_copy,evaluate
from benchmark.nm import magnitude_mask,validate_layer_mask
from benchmark.obc import load_official,prune_with_hessian
from benchmark.preflight import ROOT,gates,require_config_hash
from benchmark.recovery import recover
from test_pipeline import Tiny,batches,prepare

def test_native_permutation_hessian_inverse_and_objective_coordinate_correspondence():
    layer=nn.Conv2d(4,2,3,bias=False);x=torch.randn(36,44);h=x@x.T+10*torch.eye(36)
    cls=load_official();native=cls(layer);native.H=h.double()
    w,hp,hinv,_,perm=native.prepare(columnslast=True);inverse=perm.argsort()
    assert perm[:4].tolist()==[0,9,18,27]
    assert torch.equal(w[:,inverse],layer.weight.flatten(1))
    assert torch.equal(hp[ inverse][:,inverse],h)
    torch.testing.assert_close(hinv,torch.linalg.inv(h)[perm][:,perm])
    error=torch.randn(36,dtype=torch.float64)
    torch.testing.assert_close(error@h.double()@error,error[perm]@hp.double()@error[perm],rtol=1e-12,atol=1e-12)
    # A canonical legal Mask need not be legal in the native Conv convention.
    mask=magnitude_mask(torch.ones(1,36));mask[0,8:12]=torch.tensor([False,False,True,True])
    validate_layer_mask(torch.zeros(1,36),mask)
    assert int(mask[0,perm[:4]].sum())==1

def test_noncontiguous_conv_and_native_equal_score_tie_policy():
    layer=nn.Conv2d(4,2,3,bias=False)
    layer.weight=nn.Parameter(torch.randn(2,4,3,3).transpose(2,3));assert not layer.weight.is_contiguous()
    contiguous=copy.deepcopy(layer);contiguous.weight=nn.Parameter(layer.weight.detach().contiguous())
    x=torch.randn(36,50);h=(x@x.T+torch.eye(36)).double()
    out,mask,_=prune_with_hessian(layer,h,parallel=1)
    other,othermask,_=prune_with_hessian(contiguous,h,parallel=1)
    assert torch.equal(out,other) and torch.equal(mask,othermask)
    validate_layer_mask(out,mask)
    tied=nn.Linear(4,1,bias=False);tied.weight.data.fill_(1)
    _,tie_mask,_=prune_with_hessian(tied,torch.eye(4,dtype=torch.float64),parallel=1)
    assert tie_mask.tolist()==[[False,False,True,True]] # native argmin removes earliest tie

def test_bn_only_updates_exact_allowed_buffers_and_cumulative_statistics():
    model=Tiny().eval();model.register_buffer('sentinel',torch.tensor([7.,11.]))
    original=copy.deepcopy(model.state_dict());loader=list(batches())
    with torch.no_grad():
        inputs=[model.first(x) for x,_,_ in loader]
        mean=torch.stack([x.mean((0,2,3)) for x in inputs]).mean(0)
        variance=torch.stack([x.var((0,2,3),unbiased=True) for x in inputs]).mean(0)
    calibrated,_=recalibrate_copy(model,loader,'cpu')
    assert calibrated is not model and calibrated.bn.running_mean.data_ptr()!=model.bn.running_mean.data_ptr()
    torch.testing.assert_close(calibrated.bn.running_mean,mean)
    torch.testing.assert_close(calibrated.bn.running_var,variance)
    assert int(calibrated.bn.num_batches_tracked)==3
    for name,value in model.state_dict().items():assert torch.equal(value,original[name])
    allowed={'bn.running_mean','bn.running_var','bn.num_batches_tracked'}
    for name,value in calibrated.state_dict().items():
        if name not in allowed:assert torch.equal(value,original[name])
    before=tensor_sha(calibrated.state_dict());evaluate(calibrated,loader,'cpu')
    assert before==tensor_sha(calibrated.state_dict())

def test_every_sgd_step_keeps_mask_and_zeroes_and_best_uses_accuracy_earliest_tie(tmp_path,monkeypatch):
    model,masks=prepare();mask_digest=tensor_sha(masks);initial=model.middle.weight.detach().clone();observed=[]
    original_step=torch.optim.SGD.step
    def audited_step(optimizer,*args,**kwargs):
        result=original_step(optimizer,*args,**kwargs)
        assert tensor_sha(masks)==mask_digest
        assert torch.count_nonzero(model.middle.weight[~masks['middle.weight']])==0
        momentum=optimizer.state[model.middle.weight]['momentum_buffer']
        assert torch.count_nonzero(momentum[~masks['middle.weight']])==0
        observed.append(model.middle.weight.detach().clone());return result
    monkeypatch.setattr(torch.optim.SGD,'step',audited_step)
    values=iter([{'accuracy':.9,'loss':.2,'count':12},{'accuracy':.9,'loss':.1,'count':12},{'accuracy':.8,'loss':.3,'count':12}])
    monkeypatch.setattr('benchmark.recovery.evaluate',lambda *args:next(values))
    config={'purpose':'synthetic_test','device':'cpu','recovery':{'epochs':3,'learning_rate':.005,
        'momentum':.9,'weight_decay':.0005,'nesterov':False,'eta_min':0}}
    recover(model,masks,batches,batches,config,tmp_path,{'case':'best-tie'},{'stage':'native_complete'})
    assert len(observed)==9 and not torch.equal(initial[masks['middle.weight']],observed[0][masks['middle.weight']])
    best=artifacts.load(tmp_path/'best_test.pt');last=artifacts.load(tmp_path/'last.pt')
    assert best['epoch']==1 and best['best']['test_loss']==.2 and last['epoch']==3
    assert last['best']['epoch']==1 and last['steps']==9
    for epoch in (1,2,3):
        checkpoint=artifacts.load(tmp_path/f'epoch_{epoch:03d}.pt')
        fresh=Tiny();fresh.load_state_dict(checkpoint['model'])
        assert checkpoint['mask_sha256']==mask_digest and checkpoint['epoch']==epoch
        assert set(checkpoint['rng'])=={'python','numpy','cpu','cuda'}
        assert checkpoint['optimizer']['state'] and checkpoint['scheduler']['last_epoch']==epoch

def test_python_numpy_and_cpu_rng_restore_explicitly():
    state=rng_state();expected=random.random(),np.random.rand(),torch.rand(4)
    random.random();np.random.rand();torch.rand(20);restore_rng(state)
    assert random.random()==expected[0] and np.random.rand()==expected[1] and torch.equal(torch.rand(4),expected[2])
    if not torch.cuda.is_available():assert state['cuda'] is None

def approved_synthetic_config(tmp_path):
    c=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    p=tmp_path/'ids.json';p.write_text('[0, 1, 2, 3]',encoding='utf-8')
    c['calibration'].update(approved=True,samples=4,index_file=str(p),index_sha256=index_sha([0,1,2,3]),batch_size=2,
        rounds=1,forward_passes_per_layer=2,optimization_epochs=0,random_seed=0,augmentation='none',
        preprocessing={'to_tensor':True,'normalize_mean':c['data']['normalize_mean'],'normalize_std':c['data']['normalize_std']})
    return c

@pytest.mark.parametrize('field',['approved','index_file','rounds','augmentation','preprocessing','batch_size','forward_passes_per_layer'])
def test_each_missing_calibration_field_is_rejected(tmp_path,field):
    c=approved_synthetic_config(tmp_path);assert approved_indices(c)==[0,1,2,3]
    c['calibration'][field]=False if field=='approved' else None
    with pytest.raises(ValueError,match='pending'):approved_indices(c)

def test_source_grouping_input_and_config_hash_mismatch_fail_closed(tmp_path):
    c=approved_synthetic_config(tmp_path);c['pruning']['layout']='native_channel_strided'
    assert any('grouping' in e for e in gates(c))
    c['pruning']['layout']='oihw_output_row_contiguous_v1';c['nm_acceptance']['OBC']='BLOCKED'
    assert any('acceptance' in e for e in gates(c))
    p=tmp_path/'config.yaml';p.write_text('approved: false',encoding='utf-8')
    require_config_hash(p,sha_file(p))
    with pytest.raises(ValueError,match='SHA256'):require_config_hash(p,'wrong')
    from benchmark.audit_inputs import load_original
    # A wrong required input hash fails before even attempting to deserialize.
    with pytest.raises(ValueError,match='SHA256'):load_original(p,'wrong')
