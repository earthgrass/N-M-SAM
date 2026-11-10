import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from baseline_adapters.common.conv2d import (
    conv2d_via_unfold,
    grouped_patch_matrices,
    matrix_to_weight,
    weight_to_matrix,
)
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.rose import ROSEAdapter, rose_permutation
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from baseline_adapters.wanda import WandaAdapter
from benchmark.model import BasicBlock, ResNet


@pytest.mark.parametrize(
    "kwargs,shape",
    [
        ({"in_channels": 3, "out_channels": 5, "kernel_size": 3, "padding": 1}, (2, 3, 7, 9)),
        ({"in_channels": 4, "out_channels": 6, "kernel_size": 3, "stride": 2, "padding": 2, "dilation": 2, "groups": 2}, (2, 4, 9, 8)),
        ({"in_channels": 4, "out_channels": 4, "kernel_size": 1, "groups": 4, "bias": False}, (3, 4, 5, 6)),
    ],
)
def test_conv2d_matrix_forward_and_gradients(kwargs, shape):
    torch.manual_seed(11)
    layer_native = nn.Conv2d(**kwargs).double()
    layer_matrix = copy.deepcopy(layer_native)
    x_native = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    # Make a non-contiguous NCHW tensor without changing its values/shape.
    x_matrix = x_native.detach().transpose(2, 3).contiguous().transpose(2, 3).requires_grad_(True)
    assert not x_matrix.is_contiguous()
    native = layer_native(x_native)
    mapped = conv2d_via_unfold(layer_matrix, x_matrix, max_vectors=7)
    torch.testing.assert_close(mapped, native, rtol=1e-11, atol=1e-11)
    probe = torch.randn_like(native)
    (native * probe).sum().backward()
    (mapped * probe).sum().backward()
    torch.testing.assert_close(x_matrix.grad, x_native.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(layer_matrix.weight.grad, layer_native.weight.grad, rtol=1e-10, atol=1e-10)
    if layer_native.bias is not None:
        torch.testing.assert_close(layer_matrix.bias.grad, layer_native.bias.grad, rtol=1e-10, atol=1e-10)


def test_oihw_coordinate_roundtrip_and_group_patch_order():
    weight = torch.arange(6 * 2 * 2 * 3).reshape(6, 2, 2, 3)
    matrix = weight_to_matrix(weight)
    assert torch.equal(matrix_to_weight(matrix, weight.shape), weight)
    layer = nn.Conv2d(4, 6, (2, 3), groups=2, bias=False)
    inputs = torch.arange(2 * 4 * 4 * 5, dtype=torch.float64).reshape(2, 4, 4, 5)
    grouped = grouped_patch_matrices(layer, inputs)
    full = F.unfold(inputs, layer.kernel_size)
    features = 2 * 2 * 3
    for group in range(2):
        expected = full[:, group * features : (group + 1) * features].permute(1, 0, 2).reshape(features, -1)
        assert torch.equal(grouped[group], expected)


def test_wanda_linear_matches_formula_and_preserves_survivors():
    layer = nn.Linear(8, 3, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(1, 25, dtype=torch.float64).reshape(3, 8))
    inputs = torch.tensor(
        [[[1.0, 2, 3, 4, 5, 6, 7, 8], [2.0, 1, 0, 3, 2, 4, 1, 5]]], dtype=torch.float64
    )
    adapter = WandaAdapter(layer)
    adapter.add_batch(inputs)
    expected = layer.weight.abs() * inputs.reshape(-1, 8).square().sum(0).sqrt()[None]
    torch.testing.assert_close(adapter.scores(), expected)
    before = layer.weight.detach().clone()
    mask = adapter.prune(keep_n=2, m=4).reshape(3, 8)
    validate_keep_mask(mask)
    assert torch.equal(layer.weight[mask], before[mask])
    assert torch.count_nonzero(layer.weight[~mask]) == 0


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_wanda_conv_statistics_and_mask(groups):
    torch.manual_seed(12)
    layer = nn.Conv2d(4, 8 if groups < 4 else 4, 3, padding=1, groups=groups, bias=False).double()
    inputs = torch.randn(2, 4, 5, 5, dtype=torch.float64)
    adapter = WandaAdapter(layer)
    adapter.add_batch(inputs)
    patches = grouped_patch_matrices(layer, inputs)
    rows = layer.out_channels // groups
    scores = adapter.scores()
    for group in range(groups):
        expected = weight_to_matrix(layer.weight)[group * rows : (group + 1) * rows].abs() * patches[group].square().sum(1).sqrt()[None]
        torch.testing.assert_close(scores[group * rows : (group + 1) * rows], expected)
    mask = adapter.prune().reshape(layer.out_channels, -1)
    validate_keep_mask(mask)


def test_sparsegpt_matrix_is_finite_deterministic_and_dense_tail():
    torch.manual_seed(13)
    weight = torch.randn(4, 10, dtype=torch.float64)
    x = torch.randn(10, 37, dtype=torch.float64)
    hessian = 2 * x @ x.T / x.shape[1]
    first = sparsegpt_matrix(weight, hessian, blocksize=8)
    second = sparsegpt_matrix(weight, hessian, blocksize=8)
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    assert torch.equal(first.keep_mask, second.keep_mask)
    validate_keep_mask(first.keep_mask)
    assert bool(first.keep_mask[:, 8:].all())
    assert torch.count_nonzero(first.weight[~first.keep_mask]) == 0
    assert bool(torch.isfinite(first.weight).all())
    assert first.hessian.shape == (10, 10)


def test_sparsegpt_conv_group_hessians_and_coordinate_writeback():
    torch.manual_seed(14)
    layer = nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=False).double()
    inputs = torch.randn(2, 4, 6, 6, dtype=torch.float64)
    adapter = SparseGPTAdapter(layer)
    adapter.add_batch(inputs)
    patches = grouped_patch_matrices(layer, inputs)
    hessians = adapter.hessians()
    for group in range(2):
        expected = 2 * patches[group] @ patches[group].T / patches[group].shape[1]
        torch.testing.assert_close(hessians[group], expected)
    mask = adapter.prune(blocksize=16).reshape(layer.out_channels, -1)
    validate_keep_mask(mask)
    assert torch.count_nonzero(weight_to_matrix(layer.weight)[~mask]) == 0


def test_rose_permutation_roundtrip_preserves_nm_feasible_groups():
    # Construct sharply different group losses so reordering is guaranteed.
    weight = torch.tensor(
        [[0.01, 0.02, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0], [0.03, 0.04, 7.0, 6.0, 1.5, 2.5, 3.5, 4.5]],
        dtype=torch.float64,
    )
    order = rose_permutation(weight, torch.ones(8, dtype=torch.float64), threshold=0.0)
    assert order.reordered
    assert torch.equal(weight[:, order.permutation][:, order.inverse], weight)
    assert torch.equal(order.permutation.sort().values, torch.arange(8))
    # Every reordered consecutive 4 columns originated from one original group.
    for start in range(0, 8, 4):
        assert torch.unique(order.permutation[start : start + 4] // 4).numel() == 1


def test_rose_linear_and_conv_restore_original_coordinates():
    torch.manual_seed(15)
    for layer, inputs in (
        (nn.Linear(8, 4, bias=False).double(), torch.randn(5, 8, dtype=torch.float64)),
        (nn.Conv2d(2, 4, 2, groups=2, bias=False).double(), torch.randn(3, 2, 4, 4, dtype=torch.float64)),
    ):
        adapter = ROSEAdapter(layer)
        adapter.add_batch(inputs)
        mask, orders = adapter.prune(blocksize=8, threshold=-1.0)
        validate_keep_mask(mask.reshape(mask.shape[0], -1))
        assert all(order.reordered for order in orders)
        assert torch.count_nonzero(layer.weight[~mask]) == 0


def test_official_rose_nm_branch_static_defect_is_pinned_in_source():
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "artifacts"
        / "upstream"
        / "ROSE"
        / "lib"
        / "prune_zoo"
        / "rose.py"
    )
    if not source.exists():
        pytest.skip("Ignored pinned source clone is not present")
    text = source.read_text(encoding="utf-8")
    nm_branch = text.index("if prune_n != 0:", text.index("def fasterprune"))
    else_branch = text.index("else:", nm_branch)
    use = text.index("reordered_indices = reorder_column[reorder_block]", else_branch)
    assert "reorder_column" not in text[nm_branch:else_branch]
    assert use > else_branch
    assert "/ torch.max(block_loss)" in text


def test_cifar100_resnet18_single_layer_wanda_technical_path():
    torch.manual_seed(16)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100).double().eval()
    images = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        block_input = F.relu(model.bn1(model.conv1(images)))
    target = model.layer1[0].conv1
    adapter = WandaAdapter(target)
    adapter.add_batch(block_input)
    mask = adapter.prune().reshape(target.out_channels, -1)
    validate_keep_mask(mask)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 100)
    assert bool(torch.isfinite(logits).all())
