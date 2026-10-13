"""Phase-2A checkpoint/resume equivalence regression test."""
from nm_sam_cifar10.strong_baseline import resume_equivalence_sanity


def test_resume_equivalence_sanity():
    result = resume_equivalence_sanity()
    assert result["passed"]
    assert result["parameter_max_abs_diff"] == 0.0
    assert result["optimizer_state_equal"]
    assert result["scheduler_state_equal"]
