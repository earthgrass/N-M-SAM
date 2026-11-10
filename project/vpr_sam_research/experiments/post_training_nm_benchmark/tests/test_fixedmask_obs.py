import torch
from torch import nn
from pathlib import Path

from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.fixedmask_obs import FixedMaskOBSAdapter, fixedmask_obs_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from scripts.run_apbr_seed0_fixedmask_obs import geometry_masks
from scripts.run_cifar100_50k_formal_training import tensor_sha


def _mask(rows: int, columns: int) -> torch.Tensor:
    base = torch.tensor([False, False, True, True], dtype=torch.bool)
    return base.repeat(rows, columns // 4)


def test_external_mask_is_immutable_and_survivors_are_compensated():
    torch.manual_seed(21)
    weight = torch.randn(3, 8, dtype=torch.float64)
    x = torch.randn(8, 64, dtype=torch.float64)
    hessian = 2 * x @ x.T / x.shape[1]
    keep = _mask(3, 8)
    first = fixedmask_obs_matrix(weight, hessian, keep, blocksize=4)
    second = fixedmask_obs_matrix(weight * 17, hessian, keep, blocksize=4)
    assert torch.equal(first.keep_mask, keep)
    assert torch.equal(second.keep_mask, keep)
    assert torch.count_nonzero(first.weight[~keep]) == 0
    assert bool(torch.isfinite(first.weight).all())
    assert first.survivor_changed_count > 0
    assert torch.count_nonzero(first.weight[keep] != weight[keep]) > 0


def test_no_compensation_is_exact_mask_only_projection():
    torch.manual_seed(22)
    weight = torch.randn(2, 8, dtype=torch.float64)
    hessian = torch.eye(8, dtype=torch.float64)
    keep = _mask(2, 8)
    result = fixedmask_obs_matrix(weight, hessian, keep, blocksize=4, compensate=False)
    torch.testing.assert_close(result.weight, weight * keep, rtol=0, atol=0)
    assert result.survivor_changed_count == 0


def test_fixed_support_matches_independent_quadratic_solution_when_pruned_columns_lead():
    torch.manual_seed(23)
    matrix = torch.randn(4, 4, dtype=torch.float64)
    hessian = matrix @ matrix.T + 0.5 * torch.eye(4, dtype=torch.float64)
    weight = torch.randn(1, 4, dtype=torch.float64)
    keep = _mask(1, 4)
    result = fixedmask_obs_matrix(weight, hessian, keep, blocksize=4, percdamp=0.0)
    pruned = torch.tensor([0, 1])
    retained = torch.tensor([2, 3])
    expected = weight.clone()
    expected[:, pruned] = 0
    expected[:, retained] = weight[:, retained] + torch.linalg.solve(
        hessian[retained][:, retained], hessian[retained][:, pruned] @ weight[:, pruned].T
    ).T
    torch.testing.assert_close(result.weight, expected, rtol=1e-10, atol=1e-10)


def test_local_reconstruction_error_improves_over_mask_only():
    torch.manual_seed(24)
    weight = torch.randn(4, 8, dtype=torch.float64)
    x = torch.randn(8, 256, dtype=torch.float64)
    hessian = 2 * x @ x.T / x.shape[1]
    keep = _mask(4, 8)
    raw = fixedmask_obs_matrix(weight, hessian, keep, blocksize=8, compensate=False).weight
    compensated = fixedmask_obs_matrix(weight, hessian, keep, blocksize=8, percdamp=0.0).weight
    raw_error = torch.linalg.vector_norm((weight - raw) @ x)
    compensated_error = torch.linalg.vector_norm((weight - compensated) @ x)
    assert compensated_error < raw_error


def test_conv_adapter_uses_unfold_hessian_and_preserves_fixed_mask():
    torch.manual_seed(25)
    layer = nn.Conv2d(2, 4, 2, bias=False).double()
    inputs = torch.randn(3, 2, 5, 5, dtype=torch.float64)
    keep = _mask(4, 8).reshape_as(layer.weight)
    adapter = FixedMaskOBSAdapter(layer)
    adapter.add_batch(inputs)
    patches = grouped_patch_matrices(layer, inputs)[0]
    torch.testing.assert_close(adapter.hessians()[0], 2 * patches @ patches.T / patches.shape[1])
    before = weight_to_matrix(layer.weight).clone()
    returned, results = adapter.prune(keep, blocksize=8)
    assert torch.equal(returned, keep)
    validate_keep_mask(returned.reshape(4, 8))
    assert torch.count_nonzero(layer.weight[~keep]) == 0
    assert sum(result.survivor_changed_count for result in results) > 0
    assert torch.count_nonzero(weight_to_matrix(layer.weight)[keep.reshape(4, 8)] != before[keep.reshape(4, 8)]) > 0


def test_seed0_apbr_geometry_mask_matches_frozen_historical_artifacts():
    root = Path(__file__).resolve().parents[1]
    seed0 = root / "artifacts" / "cifar100_50k10k_final_v1" / "seed0_remote_copy"
    parent_path = seed0 / "apbr_sam" / "final_epoch_200.pt"
    raw_path = seed0 / "methods" / "apbr_sam" / "raw_hard.pt"
    if not parent_path.exists() or not raw_path.exists():
        __import__("pytest").skip("Frozen Seed-0 local mirror is unavailable")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    historical = torch.load(raw_path, map_location="cpu", weights_only=False)
    model = build_cifar100_model()
    model.load_state_dict(parent["model"])
    masks = geometry_masks(model, parent)
    assert tensor_sha(masks) == "79d9c410c67e931101cf46112e732c68133a9935e4f99a41503a0de8ee7b40f4"
    assert tensor_sha(masks) == historical["mask_sha256"]
    assert len(masks) == 19
    assert sum(mask.numel() for mask in masks.values()) == 11_157_504
    assert sum(int((~mask).sum()) for mask in masks.values()) == 5_578_752
    for key, mask in masks.items():
        assert torch.equal(mask, historical["masks"][key])
        assert torch.equal(parent["model"][key] * mask, historical["model"][key])
