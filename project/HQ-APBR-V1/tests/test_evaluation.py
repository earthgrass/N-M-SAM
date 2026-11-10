"""Bounded CPU checks; never open official Test or run final evaluation."""
from pathlib import Path

import pytest
import torch

from hq_apbr_v1.data import calibration_ids, load_test, verify_archive
from hq_apbr_v1.evaluation import prepare_layer_endpoints, reconstruction_mse, tensor_hash
from hq_apbr_v1.vendor.fixedmask_obs import fixedmask_obs_matrix
from hq_apbr_v1.vendor.model import build_model, eligible_layers
from hq_apbr_v1.vendor.sparsegpt import sparsegpt_matrix


def test_vendored_c100_model_identity_and_exclusions():
    torch.set_num_threads(2)
    model = build_model()
    layers = eligible_layers(model)
    assert len(layers) == 19
    assert sum(layer.weight.numel() for layer in layers.values()) == 11_157_504
    assert sum(p.numel() for p in model.parameters()) == 11_220_132
    assert "conv1" not in layers and "linear" not in layers
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_vendored_native_fixed_mask_replay_same_path():
    torch.manual_seed(19)
    weight = torch.randn(3, 12)
    x = torch.randn(12, 40)
    h = 2 * x @ x.T / x.shape[1]
    native = sparsegpt_matrix(weight, h, blocksize=4, compute_dtype=torch.float32)
    fixed = fixedmask_obs_matrix(weight, h, native.keep_mask, blocksize=4, compute_dtype=torch.float32)
    assert torch.equal(native.keep_mask, fixed.keep_mask)
    assert torch.equal(native.weight, fixed.weight)
    assert torch.count_nonzero(fixed.weight[~fixed.keep_mask]) == 0


def test_four_endpoint_semantics_and_physical_reconstruction():
    torch.manual_seed(31)
    weight = torch.randn(3, 12)
    original = weight.clone()
    x = torch.randn(12, 40)
    h = x @ x.T / x.shape[1]
    endpoints, report = prepare_layer_endpoints(weight, h, blocksize=4)
    assert torch.equal(weight, original)
    raw, mask = endpoints["final_q_mask_only"]
    fixed, fixed_mask = endpoints["final_q_fixed_obs"]
    assert torch.equal(mask, fixed_mask)
    assert torch.equal(raw, original * mask)
    assert not torch.equal(raw, fixed)
    assert report["fixed_obs"]["survivor_changed_count"] > 0
    direct = ((fixed - weight) @ x).square().mean()
    assert reconstruction_mse(weight, fixed, h) == pytest.approx(float(direct), rel=1e-5)
    assert all(bool(torch.isfinite(value).all()) for value, _ in endpoints.values())
    assert tensor_hash({"w": mask}) == tensor_hash({"w": fixed_mask})


def test_train_calibration_identity_and_no_test_without_freeze(tmp_path):
    ids = calibration_ids()
    assert len(ids) == 1024 and len(set(ids)) == 1024
    assert min(ids) >= 0 and max(ids) < 50000
    invalid = tmp_path / "not_a_frozen_evaluation.json"
    invalid.write_text('{"status":"TRAINING","test_accessed":false}')
    with pytest.raises(ValueError, match="frozen"):
        load_test(tmp_path / "does_not_exist.tar.gz", frozen_manifest=invalid)
    bad_archive = tmp_path / "cifar-100-python.tar.gz"
    bad_archive.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_archive(bad_archive)
