import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.model import build_model
from benchmark.nm import eligible_layers, magnitude_mask, validate_layer_mask


def test_cifar100_factory_is_isolated_from_legacy_cifar10_factory():
    torch.manual_seed(7)
    cifar100 = build_cifar100_model()
    cifar10 = build_model()
    assert cifar100(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    assert cifar10(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    shapes100 = {name: value.shape for name, value in cifar100.state_dict().items()}
    shapes10 = {name: value.shape for name, value in cifar10.state_dict().items()}
    assert {name for name in shapes100 if shapes100[name] != shapes10[name]} == {"linear.weight", "linear.bias"}


def test_cifar100_forward_gradients_and_pruning_scope():
    model = build_cifar100_model()
    loss = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 99]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    scope = eligible_layers(model)
    assert "conv1" not in scope and "linear" not in scope
    assert all(isinstance(layer, (nn.Conv2d, nn.Linear)) for layer in scope.values())


def test_frozen_nm_tie_tail_and_bool_keep_mask():
    weight = torch.tensor([[1.0, 1.0, 1.0, 1.0, 8.0, 7.0]])
    mask = magnitude_mask(weight, n=2, m=4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False, False, True, True]]
    validate_layer_mask(weight, mask, n=2, m=4, check_zeros=False)


def test_sparsegpt_explicit_fp32_compute_path():
    weight = torch.tensor([[4.0, 3.0, 2.0, 1.0, 9.0, 8.0]])
    result = sparsegpt_matrix(weight, torch.eye(6), blocksize=4, compute_dtype=torch.float32)
    assert result.weight.dtype == torch.float32
    assert result.keep_mask.dtype == torch.bool
    assert result.keep_mask[:, :4].sum() == 2
    assert result.keep_mask[:, 4:].all()
    assert result.decomposition_seconds >= 0
    assert result.compensation_seconds >= 0
    adapter = SparseGPTAdapter(nn.Conv2d(1, 1, 3, padding=1, bias=False), compute_dtype=torch.float32)
    assert adapter.xtx.dtype == torch.float32


@pytest.mark.skipif(torch.cuda.is_available(), reason="This test only checks fail-closed behavior on a no-CUDA host")
def test_gpu_gate_refuses_cpu_substitution(tmp_path: Path):
    audit = tmp_path / "dataset.json"
    audit.write_text(json.dumps({"status": "DATA_IDENTITY_PASS"}), encoding="utf-8")
    model_audit = tmp_path / "model.json"
    model_audit.write_text(json.dumps({"status": "CIFAR100_MODEL_IDENTITY_PASS"}), encoding="utf-8")
    output = tmp_path / "gate.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sparsegpt_single_layer_gpu_gate.py"
    run = subprocess.run(
        [
            sys.executable, str(script), "--dataset-root", str(tmp_path),
            "--dataset-audit", str(audit), "--model-audit", str(model_audit),
            "--output", str(output),
        ],
        capture_output=True, text=True, check=False,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert run.returncode == 2
    assert result["status"] == "BLOCKED_NO_CUDA"
    assert result["statuses"]["SPARSEGPT_GPU_NUMERICAL_PASS"] is False
    assert result["statuses"]["SPARSEGPT_GPU_RESOURCE_PASS"] is False
