import pytest,torch
from torch import nn
from benchmark.nm import magnitude_mask,validate_layer_mask
from benchmark.obc import accumulate,prune_with_hessian
from benchmark.probes import grouping_probe

def test_native_conv_grouping_is_not_canonical():
    proof=grouping_probe();assert proof['canonical_first_group']==[0,1,2,3]
    assert proof['obc_and_snows_first_group']==[0,9,18,27]

def test_stable_ties_and_dense_tail():
    w=torch.ones(2,10);mask=magnitude_mask(w)
    assert mask[0].tolist()==[True,True,False,False,True,True,False,False,True,True]
    validate_layer_mask(w*mask,mask)
    wrong=mask.clone();wrong[0,-1]=False
    with pytest.raises(ValueError):validate_layer_mask(w*wrong,wrong)

def test_official_obc_identity_hessian_and_selection_record():
    layer=nn.Linear(8,2,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.arange(1,17.).reshape(2,8))
    before=layer.weight.detach().clone()
    out,mask,receipt=prune_with_hessian(layer,torch.eye(8,dtype=torch.float64),parallel=1)
    assert torch.equal(mask,magnitude_mask(before)) and torch.equal(out,before*mask)
    assert torch.equal(layer.weight,before) and receipt['cpu_timing_fence_only']

def test_native_compensation_matches_conditional_optimum():
    layer=nn.Linear(8,2,bias=False);original=layer.weight.detach().clone()
    x=torch.randn(8,12);h=x@x.T+torch.eye(8)
    out,mask,_=prune_with_hessian(layer,h.double(),parallel=2)
    for row in range(2):
        keep=mask[row];drop=~keep
        optimum=original[row,keep]+torch.linalg.solve(h[keep][:,keep],h[keep][:,drop]@original[row,drop])
        assert torch.allclose(out[row,keep],optimum,rtol=3e-5,atol=3e-6)
        error=out[row]-original[row];raw=original[row]*mask[row]-original[row]
        assert error@h@error<=raw@h@raw+1e-6
    assert not torch.equal(out,original*mask)

def test_native_conv_hessian_matches_unfold_objective_and_linear_interface():
    layer=nn.Conv2d(4,2,3,padding=1,bias=False);images=torch.randn(3,4,5,5)
    h,count=accumulate(layer,[(images,layer(images))]);assert count==3
    x=nn.functional.unfold(images,3,padding=1).permute(1,0,2).flatten(1)
    expected=(x@x.T).double()*(2/3) # native matmul FP32, cast FP64, then normalize
    assert torch.equal(h,expected)
    proxy=nn.Linear(36,2,bias=False);proxy.weight.data.copy_(layer.weight.flatten(1))
    patches=x.T
    assert torch.allclose(proxy(patches).T,layer(images).permute(1,0,2,3).flatten(1),atol=1e-6)
    out,mask,_=prune_with_hessian(layer,h,parallel=1);validate_layer_mask(out,mask)
    # Same native kernel given the exact same canonical coordinates and Hessian.
    direct,direct_mask,_=prune_with_hessian(proxy,h,parallel=1)
    assert torch.equal(out.flatten(1),direct) and torch.equal(mask.flatten(1),direct_mask)

def test_obc_fails_nonfinite_and_unimplemented_partial_buckets():
    layer=nn.Linear(6,2,bias=False)
    with pytest.raises(ValueError):prune_with_hessian(layer,torch.eye(6))
    layer=nn.Linear(4,2,bias=False);h=torch.eye(4);h[0,0]=float('nan')
    with pytest.raises(FloatingPointError):prune_with_hessian(layer,h)

def test_zero_retained_weights_do_not_erase_explicit_mask():
    layer=nn.Linear(4,2,bias=False);layer.weight.data.zero_()
    out,mask,_=prune_with_hessian(layer,torch.eye(4,dtype=torch.float64),parallel=1)
    assert torch.count_nonzero(out)==0 and (mask.sum(1)==2).all()
