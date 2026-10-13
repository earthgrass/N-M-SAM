import math
import pytest
import torch
from nmsam.diagnostics.quadratic_robust import quadratic_maximum
from nmsam.diagnostics.robust_probe import robust_proxy
from nmsam.diagnostics.objective_gap import evaluate_objective_gap
from nmsam.sparsity.mask import enumerate_masks


@pytest.mark.parametrize("diagonal",[[1.,2.,3.,4.],[0.,0.,1.,2.],[-4.,-2.,1.,3.],[-4.,-3.,-2.,-1.]])
def test_quadratic_global_supported_certificate_and_grid(diagonal):
    h=torch.diag(torch.tensor(diagonal,dtype=torch.float64));b=torch.tensor([.2,-.3,.4,-.1],dtype=torch.float64)
    for support in enumerate_masks():
        result=quadratic_maximum(h,b,support,.4);d=result["delta"]
        assert d.norm()<=.4+1e-10 and (d[~support]==0).all()
        assert result["stationarity_error"]<1e-8
        idx=support.nonzero().flatten();angle=torch.linspace(0,2*math.pi,2048,dtype=torch.float64)
        grid=torch.zeros(2048,4,dtype=torch.float64)
        grid[:,idx[0]]=.4*angle.cos();grid[:,idx[1]]=.4*angle.sin()
        values=grid@b+.5*(grid@h*grid).sum(-1)
        assert result["value"]>=float(values.max())-1e-10


def test_hard_case_and_one_step_counterexample():
    h=torch.diag(torch.tensor([1.,3.,-1.,-2.],dtype=torch.float64));b=torch.zeros(4,dtype=torch.float64)
    support=torch.tensor([True,True,False,False]);z=torch.zeros(4,dtype=torch.float64)
    reference=quadratic_maximum(h,b,support,.2)
    assert reference["value"]==pytest.approx(.5*3*.2**2)
    one=robust_proxy(lambda x:.5*x@h@x,z,support,.2)
    assert one["value"]==0 and not one["global_max_proven"]
    assert reference["value"]>one["value"] # mathematical counterexample, not experimental evidence


def test_support_changes_quadratic_candidate_objective_decomposition():
    h=torch.diag(torch.tensor([1.,2.,3.,4.],dtype=torch.float64));w=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    old=enumerate_masks()[0];z=w*old;loss=lambda x:.5*x@h@x
    candidates=[{"candidate_id":i,"support":m,"q":w*(m.int()-old.int())} for i,m in enumerate(enumerate_masks())]
    labels={"objective":"task_quadratic","derivative":"effective_true","state_mode":"eval_fixed_buffers","mask_id":0,"snapshot_id":"toy"}
    results=evaluate_objective_gap(loss,z,old,candidates,labels=labels)
    for row in results["rows"]:
        assert row["delta_F_proxy"]==pytest.approx(row["delta_L"]+row["delta_S_proxy"])
        assert row["Q_robust_proxy"]==pytest.approx(row["F_new_proxy"],abs=1e-10)
    new=enumerate_masks()[-1]
    proxy=robust_proxy(loss,w*new,new,.2,steps=3,restarts=2)
    assert (proxy["delta"][~new]==0).all() and proxy["delta"].norm()<=.2+1e-12
