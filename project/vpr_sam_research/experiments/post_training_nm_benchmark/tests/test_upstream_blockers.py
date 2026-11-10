import torch
from benchmark.probes import snows_probe,cap_probe

def test_unmodified_snows_default_chunk_hvp_counterexample_is_detected():
    result=snows_probe()
    assert result['status']=='BLOCKED_NATIVE_HVP'
    assert result['whole_module_syntax_error'] is not None
    assert result['pinned_default_hvp'][0]['max_abs_hvp_error']==0
    assert result['pinned_default_hvp'][1]['max_abs_hvp_error']==1
    assert result['zero_rhs_cg_finite'] is False
    # Passing this diagnostic test means the candidate is BLOCKED, not validated.

def test_cap_official_nm_constructor_failures_are_not_silently_repaired():
    result=cap_probe();assert result['status']=='BLOCKED_OFFICIAL_NM_ENTRY'
    assert result['errors']['CAPHandle']['type']=='TypeError'
    assert result['errors']['CAPNMHandle']['type']=='AttributeError'
