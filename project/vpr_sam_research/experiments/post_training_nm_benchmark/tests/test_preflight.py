from pathlib import Path
import yaml
from benchmark.preflight import ROOT,gates,source_identity

def test_pending_calibration_and_no_gpu_authorization_are_independent_gates():
    c=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    errors=gates(c)
    assert any('pending' in e for e in errors) and any('not yet authorized' in e for e in errors)
    assert any('blocked' in e for e in gates(c,'SNOWS'))
    a=source_identity();assert 'configs/upstream_lock.json' in a['files']
    assert 'benchmark/pipeline.py' in a['files'] and a==source_identity()

def test_formal_gpu_gate_requires_cuda_acceptance_receipt_without_cpu_fallback(monkeypatch):
    import torch
    c=yaml.safe_load((ROOT/'configs/post_training_nm_benchmark.yaml').read_text(encoding='utf-8'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    errors=gates(c,gpu=True)
    assert any('CUDA smoke receipt' in e for e in errors)
    assert any('CPU substitution prohibited' in e for e in errors)
