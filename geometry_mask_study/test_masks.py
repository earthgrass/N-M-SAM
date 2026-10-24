import copy
import torch
import pytest
from masks import three_masks,score_mask,mask_difference,deletion_metrics
from vpr_sam.pruning import nm_mask
from vpr_sam.optim.geometry import Geometry


def test_known_scores_ties_and_untouched_inputs():
    w=torch.tensor([[1.,2.,3.,4.,5.]])
    s=torch.tensor([[0.,2.,-2.,0.,9.]],dtype=torch.float64)
    original=w.clone(),s.clone()
    masks=three_masks(w,s)
    assert masks['magnitude'].tolist()==[[False,False,True,True,True]]
    assert masks['radius_only'].tolist()==[[True,False,True,False,True]]
    assert masks['geometry_guided'].tolist()==[[False,False,True,True,True]]
    assert torch.equal(w,original[0]) and torch.equal(s,original[1])
    assert score_mask(torch.ones(1,4,dtype=torch.float64),2,4).tolist()==[[True,True,False,False]]
    assert torch.equal(masks['magnitude'],nm_mask(w,2,4,'dense'))


def test_geometry_minimizes_declared_cost_not_loss():
    w=torch.tensor([[.1,.2,.3,.4]])
    s=torch.tensor([[-2.,0.,0.,2.]],dtype=torch.float64)
    masks=three_masks(w,s)
    assert masks['geometry_guided'].tolist()==[[True,False,True,False]]
    costs={k:deletion_metrics(w,s,v) for k,v in masks.items()}
    assert costs['geometry_guided']['geometric_deletion_cost']<costs['magnitude']['geometric_deletion_cost']
    assert costs['magnitude']['displacement_l2']<costs['geometry_guided']['displacement_l2']
    difference=mask_difference(masks['magnitude'],masks['geometry_guided'])
    assert difference['mask_flip_rate']==.5 and difference['different_block_fraction']==1.


@pytest.mark.parametrize('shape',[(3,19),(4,3,3,3)])
def test_full_block_legality_tail_and_original_magnitude(shape):
    torch.manual_seed(31)
    w=torch.randn(shape);s=torch.randn(shape,dtype=torch.float64)*.2
    for mask in three_masks(w,s).values():
        flat=mask.reshape(shape[0],-1);head=flat.shape[1]//4*4
        assert bool((flat[:,:head].reshape(shape[0],-1,4).sum(-1)==2).all())
        assert bool(flat[:,head:].all())
    assert torch.equal(three_masks(w,s)['magnitude'],nm_mask(w,2,4,'dense'))


def test_geometry_mapping_is_fail_closed():
    model=torch.nn.Linear(3,2)
    c=dict(dtype='float64',gamma=5e-8,warmup_steps=0,volume_atol=1e-10,volume_rtol=1e-12,static_std=.1)
    g=Geometry(model.named_parameters(),c,'fixed_l2');state=g.state_dict()
    wrong=copy.deepcopy(state);wrong['mapping'][0]=('different_name',(2,3))
    with pytest.raises(ValueError,match='mapping'):g.load_state_dict(wrong)
    with pytest.raises(ValueError):three_masks(model.weight,torch.zeros_like(model.weight))
    with pytest.raises(FloatingPointError):three_masks(model.weight,torch.full_like(model.weight,float('nan'),dtype=torch.float64))
