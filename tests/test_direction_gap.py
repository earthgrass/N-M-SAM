import pytest
import torch
from nmsam.diagnostics.direction_gap import evaluate_direction_gap
from nmsam.diagnostics.gradient_path import gradient_path,subtract_l2_path
from nmsam.diagnostics.candidate_ranking import ranking_metrics,validate_labels
from nmsam.sparsity.mask import enumerate_masks

LABELS={"objective":"task_quadratic","derivative":"effective_true","state_mode":"eval_fixed_buffers","mask_id":0,"snapshot_id":"analytic"}


def test_constant_hessian_six_candidates_and_residual_identity():
    h=torch.diag(torch.tensor([1.,2.,4.,8.],dtype=torch.float64));w=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    old=enumerate_masks()[0];z=w*old
    loss=lambda x:.5*x@h@x
    candidates=[{"candidate_id":i,"support":m,"q":w*(m.int()-old.int())} for i,m in enumerate(enumerate_masks())]
    result=evaluate_direction_gap(loss,z,w,old,candidates,labels=LABELS)
    for row in result["rows"]:
        assert row["decomposition_error"]<1e-10 and row["path_average_error"]<1e-10
        assert row["second"]==pytest.approx(row["actual"],abs=1e-10)
        assert row["observed_plus_residual"]==pytest.approx(row["actual"],abs=1e-10)
    assert result["rankings"]["second"]["regret"]==0
    # Missing residual demonstrably matters in a known quadratic, not a network claim.
    assert result["rankings"]["observed_only"]["mae"]>0


def test_undefined_direction_and_regularizer_correction():
    z=torch.ones(4,dtype=torch.float64);support=torch.ones(4,dtype=torch.bool)
    loss=lambda x:x.square().sum()
    path=gradient_path(loss,z,support,0)
    assert path["status"]=="undefined_direction"
    eps=torch.tensor([.1,0.,.2,0.],dtype=torch.float64)
    torch.testing.assert_close(subtract_l2_path(2*eps+.3*eps,eps,.3,support),2*eps)
    metric=ranking_metrics([1,1],[2,2]);assert metric["spearman"] is None
    with pytest.raises(ValueError):validate_labels([{}])
