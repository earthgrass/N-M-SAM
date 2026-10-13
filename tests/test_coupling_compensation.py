import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.functional_model import EffectiveLoss
from nmsam.sparsity.registry import MaskRegistry
from nmsam.diagnostics.candidate_mask import joint_candidates
from nmsam.diagnostics.coupling import cross_block_coupling
from nmsam.diagnostics.compensation import quadratic_compensation_reference,steihaug_compensation,compare_compensation


def test_cross_block_identity_and_analytic_compensation():
    h=torch.tensor([[4.,1.,1.,0.],[1.,3.,0.,1.],[1.,0.,2.,.5],[0.,1.,.5,2.]],dtype=torch.float64)
    g=torch.tensor([.2,.1,-.4,.3],dtype=torch.float64);z=torch.zeros(4,dtype=torch.float64)
    loss=lambda x:g@x+.5*x@h@x
    qa=torch.tensor([1.,0.,0.,0.],dtype=torch.float64);qb=torch.tensor([0.,0.,2.,0.],dtype=torch.float64)
    coupling=cross_block_coupling(loss,z,qa,qb)
    assert coupling["identity_error"]<1e-12 and coupling["cross"]==2.
    support=torch.tensor([False,True,True,False]);q=qa+qb
    result=quadratic_compensation_reference(h,g,q,support,10.)
    expected=-torch.linalg.solve(h[support][:,support],(g+h@q)[support])
    torch.testing.assert_close(result["v"][support],expected)
    cg=steihaug_compensation(g+h@q,lambda x:h@x,support,10.,damping=0,max_hvps=4)
    torch.testing.assert_close(cg["v"],result["v"],atol=1e-10,rtol=1e-10)


def test_indefinite_trust_region_and_matched_recovery_interface():
    h=torch.diag(torch.tensor([-2.,1.,3.,4.],dtype=torch.float64));g=torch.ones(4,dtype=torch.float64)
    z=torch.zeros(4,dtype=torch.float64);q=torch.tensor([0.,0.,.2,0.],dtype=torch.float64)
    support=torch.tensor([True,True,False,False]);loss=lambda x:g@x+.5*x@h@x
    reference=quadratic_compensation_reference(h,g,q,support,.2)
    assert reference["v"].norm()<=.2+1e-10
    cg=steihaug_compensation(g,lambda x:h@x,support,.2,damping=0)
    assert cg["v"].norm()<=.2+1e-10 and not cg["global_min_proven"]
    result=compare_compensation(loss,z,q,support,.2,.1,max_hvps=2)
    assert result["quadratic"]["backward_calls"]==result["sgd_matched_quadratic"]["backward_calls"]
    assert result["robust"]["backward_calls"]==result["sgd_matched_robust"]["backward_calls"]
    for r in result.values():assert (r["v"][~support]==0).all()


def test_joint_candidates_budget_and_strict_legality():
    model=nn.Linear(8,2).double();registry=MaskRegistry(model)
    context=EffectiveLoss(model,StepBatch(torch.zeros(2,8,dtype=torch.float64),torch.tensor([0,1])),registry)
    candidates=list(joint_candidates(context,[("weight",0),("weight",1)]))
    assert len(candidates)==36
    for candidate in candidates:
        torch.testing.assert_close(context.z+candidate["q"],context.master*candidate["support"])
        weight=context.unpack(candidate["support"].to(context.z.dtype))["weight"]
        assert (weight.reshape(-1,4).sum(-1)==2).all()
    with pytest.raises(ValueError):list(joint_candidates(context,[("weight",0),("weight",1),("weight",2)]))
