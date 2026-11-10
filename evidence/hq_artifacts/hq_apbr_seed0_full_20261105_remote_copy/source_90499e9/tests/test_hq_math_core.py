"""CPU-only numerical gates A-G for the independent HQ-APBR reference."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hq_apbr_v1 import (
    GeometrySpec, final_obs_keep_mask, hq_global_l2_perturbation,
    historical_geometry_update, inverse_diagonal_cholesky,
    obs_singleton_cost, target_from_inverse_diagonals,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(20261105)
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()


def test_a_obs_cost_matches_constrained_spd_solution():
    generator = torch.Generator().manual_seed(81)
    matrix = torch.randn(7, 7, dtype=torch.float64, generator=generator)
    gram = matrix.T @ matrix + .3 * torch.eye(7, dtype=torch.float64)
    qdiag, damped, delta = inverse_diagonal_cholesky(gram, percdamp=.01)
    assert delta > 0
    q_direct = torch.linalg.inv(damped)
    assert torch.allclose(qdiag, q_direct.diag(), rtol=1e-12, atol=1e-12)
    weight = torch.randn(7, dtype=torch.float64, generator=generator)
    deleted = 2
    keep = [i for i in range(7) if i != deleted]
    dw = torch.empty_like(weight)
    dw[deleted] = -weight[deleted]
    dw[keep] = torch.linalg.solve(damped[keep][:, keep], damped[keep, deleted] * weight[deleted])
    direct_cost = .5 * (dw @ damped @ dw)
    formula_cost = obs_singleton_cost(weight[deleted:deleted + 1], qdiag[deleted:deleted + 1])[0]
    assert torch.allclose(direct_cost, formula_cost, rtol=1e-12, atol=1e-12)
    # This is not the cost under the original undamped reconstruction Gram.
    undamped_cost = .5 * (dw @ gram @ dw)
    assert abs(float(direct_cost - undamped_cost)) > 1e-5


def test_a_sparsegpt_dead_feature_policy_is_explicit():
    gram = torch.diag(torch.tensor([2., 0., 3.], dtype=torch.float64))
    q, repaired, damping = inverse_diagonal_cholesky(gram, dead_policy="sparsegpt")
    assert damping == pytest.approx(.02)
    assert repaired[1, 1] == pytest.approx(1.02)
    assert bool(torch.isfinite(q).all())
    with pytest.raises(torch._C._LinAlgError):
        inverse_diagonal_cholesky(gram, percdamp=0, dead_policy="reject")


def test_a_explicit_symmetrization_rejects_large_asymmetry_and_nonfinite_cost():
    almost_symmetric = torch.tensor([[1., .1], [.10000005, 1.]], dtype=torch.float64)
    q, effective_h, _ = inverse_diagonal_cholesky(almost_symmetric)
    assert torch.equal(effective_h, effective_h.T)
    assert torch.allclose(q, torch.linalg.inv(effective_h).diag(), atol=1e-14)
    with pytest.raises(ValueError, match="materially asymmetric"):
        inverse_diagonal_cholesky(torch.tensor([[1., .1], [.11, 1.]], dtype=torch.float64))
    with pytest.raises(FloatingPointError, match="overflow"):
        obs_singleton_cost(torch.tensor([1e308], dtype=torch.float64),
                           torch.tensor([5e-324], dtype=torch.float64))


def test_b_log_target_and_global_positive_scale():
    q = torch.tensor([[1., 4., 9., 16.]], dtype=torch.float64)
    spec = GeometrySpec("layer.weight", (2, 4, 1, 1), 1, q)
    target, center, count = target_from_inverse_diagonals([spec])
    assert count == 8
    assert torch.allclose(torch.exp(2 * (.5 * q.log())), q)
    assert torch.allclose(torch.exp(2 * target[spec.name][0, :, 0, 0]), q[0] * torch.exp(-2 * center))


def test_c_historical_update_equals_explicit_geometric_history():
    eta = .25
    diagonals = [torch.tensor([[1., 4.]]), torch.tensor([[9., 1.]]), torch.tensor([[4., 16.]])]
    state = {"a.weight": torch.zeros((2, 2, 1, 1), dtype=torch.float64)}
    targets = []
    for diagonal in diagonals:
        target, _, _ = target_from_inverse_diagonals([GeometrySpec("a.weight", (2, 2, 1, 1), 1, diagonal)])
        targets.append(target["a.weight"])
        state = historical_geometry_update(state, target, eta)
    explicit = sum(eta * (1 - eta) ** (2 - k) * t for k, t in enumerate(targets))
    assert torch.allclose(state["a.weight"], explicit, rtol=0, atol=1e-15)
    q_ratios = [d.to(torch.float64) / torch.exp(d.to(torch.float64).log().mean()) for d in diagonals]
    geometric = torch.ones((2, 2), dtype=torch.float64)
    for k, ratio in enumerate(q_ratios):
        geometric *= ratio.repeat_interleave(2, dim=0) ** (eta * (1 - eta) ** (2 - k))
    assert torch.allclose(torch.exp(2 * state["a.weight"].reshape(2, 2)), geometric, atol=1e-14)
    assert torch.equal(historical_geometry_update(state, state, 0)["a.weight"], state["a.weight"])
    assert torch.equal(historical_geometry_update(state, {"a.weight": targets[-1]}, 1)["a.weight"], targets[-1])


def test_c_rejects_uncentered_history():
    old = {"a.weight": torch.tensor([0., .1], dtype=torch.float64)}
    target = {"a.weight": torch.tensor([-.2, .2], dtype=torch.float64)}
    with pytest.raises(ValueError, match="zero-sum"):
        historical_geometry_update(old, target, .3)


def test_d_global_centering_weights_actual_coordinates_and_groups():
    a = GeometrySpec("a.weight", (2, 4, 1, 1), 1, torch.tensor([[1., 2., 3., 4.]]))
    b = GeometrySpec("b.weight", (6, 1, 1, 2), 2, torch.tensor([[5., 6.], [7., 8.]]))
    targets, center, count = target_from_inverse_diagonals([a, b])
    assert count == 20
    expanded = torch.cat([targets[a.name].flatten(), targets[b.name].flatten()])
    assert abs(float(expanded.sum())) < 1e-12
    mean = (2 * (.5 * a.q_diag.double().log()).sum() + 3 * (.5 * b.q_diag.double().log()).sum()) / 20
    assert center == pytest.approx(float(mean))
    assert torch.equal(targets[b.name][0], targets[b.name][2])
    assert torch.equal(targets[b.name][3], targets[b.name][5])
    ratio = torch.exp(2 * (targets[a.name][0, 1] - targets[a.name][0, 0]))
    assert ratio == pytest.approx(2.)


def test_e_fixed_global_l2_radius_includes_noneligible_and_handles_zeros():
    parameters = {"a.weight": torch.empty((2, 3)), "bias": torch.empty(2)}
    gradients = {"a.weight": torch.tensor([[1., 0., -2.], [3., 4., 0.]]), "bias": torch.tensor([5., -1.])}
    geometry = {"a.weight": torch.tensor([[0., .2, -.3], [.4, .5, .6]], dtype=torch.float64)}
    epsilon, norm, zero = hq_global_l2_perturbation(parameters, gradients, geometry, rho=.05)
    assert zero is False
    direct = torch.cat([(torch.exp(2 * geometry["a.weight"]) * gradients["a.weight"]).flatten(), gradients["bias"].double()])
    direct = .05 * direct / direct.norm()
    actual = torch.cat([epsilon["a.weight"].flatten(), epsilon["bias"]]).double()
    assert torch.allclose(actual, direct, rtol=2e-7, atol=1e-9)
    assert abs(norm - .05) / .05 < 2e-7
    empty, empty_norm, zero = hq_global_l2_perturbation(parameters, {}, geometry, rho=.05)
    assert zero and empty_norm == 0 and all(torch.count_nonzero(x) == 0 for x in empty.values())
    with pytest.raises(FloatingPointError):
        hq_global_l2_perturbation(parameters, gradients, {"a.weight": torch.full((2, 3), 1e308, dtype=torch.float64)}, .05)


def test_f_direct_q_equivalence_and_noneligible_scale_caveat():
    q = torch.tensor([[1., 4., 9., 16.]], dtype=torch.float64)
    targets, center, _ = target_from_inverse_diagonals([GeometrySpec("a.weight", (2, 4, 1, 1), 1, q)])
    parameters = {"a.weight": torch.empty((2, 4, 1, 1)), "bias": torch.empty(2)}
    g = torch.tensor([[[[1.]], [[2.]], [[3.]], [[4.]]]]).repeat(2, 1, 1, 1)
    gradients = {"a.weight": g, "bias": torch.zeros(2)}
    eps, _, _ = hq_global_l2_perturbation(parameters, gradients, targets, .05)
    direct = (q.repeat(2, 1).reshape_as(g) * g).double()
    direct = .05 * direct / direct.norm()
    assert torch.allclose(eps["a.weight"].double(), direct, atol=3e-9)
    gradients["bias"] = torch.tensor([1., 2.])
    eps, _, _ = hq_global_l2_perturbation(parameters, gradients, targets, .05)
    normalized_q = q * torch.exp(-2 * center)
    scaled = torch.cat([(normalized_q.repeat(2, 1).reshape_as(g) * g).flatten(), gradients["bias"].double()])
    expected = .05 * scaled / scaled.norm()
    actual = torch.cat([eps["a.weight"].flatten(), eps["bias"]]).double()
    assert torch.allclose(actual, expected, atol=3e-9)
    naive = torch.cat([(q.repeat(2, 1).reshape_as(g) * g).flatten(), gradients["bias"].double()])
    naive = .05 * naive / naive.norm()
    assert not torch.allclose(actual, naive, atol=1e-5)


def test_g_actual_resnet18_mapping_and_unfold_order():
    # Test-only read-only source import; the HQ implementation has no runtime
    # dependency on the historical project or its model factory.
    reference = Path(__file__).resolve().parents[2] / "APBR-post-training-benchmark/vpr_sam_research/experiments/post_training_nm_benchmark"
    assert reference.is_dir()
    sys.path.insert(0, str(reference))
    try:
        from baseline_adapters.cifar100_model import build_cifar100_model
        from benchmark.nm import eligible_layers
        model = build_cifar100_model().eval()
        layers = eligible_layers(model)
    finally:
        sys.path.remove(str(reference))
    assert len(layers) == 19
    specs = []
    deleted_total = 0
    for name, layer in layers.items():
        assert isinstance(layer, nn.Conv2d)
        features = layer.weight[0].numel()
        q = torch.linspace(1., 2., features, dtype=torch.float64).reshape(1, -1)
        specs.append(GeometrySpec(name + ".weight", tuple(layer.weight.shape), layer.groups, q))
        keep = final_obs_keep_mask(layer.weight.detach().reshape(layer.out_channels, -1), q, groups=layer.groups)
        assert keep.dtype == torch.bool and keep.shape == (layer.out_channels, features)
        complete = features // 4 * 4
        assert bool((keep[:, :complete].reshape(layer.out_channels, -1, 4).sum(-1) == 2).all())
        assert bool(keep[:, complete:].all())
        deleted_total += int((~keep).sum())
    targets, _, coordinates = target_from_inverse_diagonals(specs)
    assert coordinates == 11_157_504
    assert deleted_total == 5_578_752
    assert set(targets) == {name + ".weight" for name in layers}
    assert all(targets[name + ".weight"].shape == layer.weight.shape for name, layer in layers.items())
    for name in ("layer1.0.conv1", "layer4.0.conv2"):
        layer = layers[name]
        spatial = 8 if name.startswith("layer1") else 4
        x = torch.randn(1, layer.in_channels, spatial, spatial)
        patches = F.unfold(x, layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride)
        matrix = layer.weight.reshape(layer.out_channels, -1)
        out = (matrix @ patches).reshape_as(layer(x) - (0 if layer.bias is None else layer.bias[None, :, None, None]))
        if layer.bias is not None:
            out += layer.bias[None, :, None, None]
        assert torch.allclose(out, layer(x), atol=2e-5, rtol=2e-5)
    grouped = nn.Conv2d(4, 4, 3, padding=1, groups=2, bias=False).eval()
    x = torch.randn(1, 4, 5, 5)
    patches = F.unfold(x, 3, padding=1).reshape(1, 2, 18, 25)
    outputs = []
    for group in range(2):
        selected = patches[:, group].squeeze(0)
        outputs.append(grouped.weight[group * 2:(group + 1) * 2].reshape(2, -1) @ selected)
    grouped_out = torch.cat(outputs).reshape(4, 1, 5, 5).permute(1, 0, 2, 3)
    assert torch.allclose(grouped_out, grouped(x), atol=1e-5)
    mask = final_obs_keep_mask(torch.tensor([[1., 1., 1., 1., 7.]]), torch.ones(4 + 1))
    assert mask.dtype == torch.bool and mask.tolist() == [[False, False, True, True, True]]


def test_invalid_mappings_fail_explicitly():
    with pytest.raises(ValueError):
        target_from_inverse_diagonals([GeometrySpec("x.weight", (2, 3, 1, 1), 1, torch.ones(1, 2))])
    with pytest.raises(ValueError):
        final_obs_keep_mask(torch.ones(2, 4), torch.ones(3))
    with pytest.raises(ValueError):
        historical_geometry_update({"x": torch.zeros(2)}, {"y": torch.zeros(2)}, .5)
    extreme_mask = final_obs_keep_mask(
        torch.tensor([[1e308, 1e307, 1., 0.]], dtype=torch.float64),
        torch.tensor([5e-324, 5e-324, 1., 1.], dtype=torch.float64),
    )
    assert extreme_mask.tolist() == [[True, True, False, False]]
