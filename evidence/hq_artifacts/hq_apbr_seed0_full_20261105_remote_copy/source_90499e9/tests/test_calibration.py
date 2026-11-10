"""Bounded synthetic calibration tests; no CIFAR-100 Test access."""
import random
import weakref

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hq_apbr_v1.calibration import collect_grams, inverse_diagonal_timed, refresh, spearman
from hq_apbr_v1.math_core import inverse_diagonal_cholesky
from hq_apbr_v1.training import tensor_sha
import hq_apbr_v1.calibration as calibration


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(2, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.internal = nn.Conv2d(4, 4, 3, padding=1, groups=2, bias=False)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.internal(self.bn(self.stem(x))).mean((2, 3)))


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(412)
    model = SmallModel()
    inputs = torch.randn(4, 2, 5, 5)
    labels = torch.tensor([0, 1, 0, 1])
    ids = torch.arange(4)
    loader = DataLoader(TensorDataset(inputs, labels, ids), batch_size=2, shuffle=False)
    config = {"expected_layers": 1, "expected_coordinates": model.internal.weight.numel(),
              "expected_samples": 4, "percdamp": .01, "eta": .1, "gram_chunk_columns": 0}
    return model, inputs, loader, config


def test_collect_gram_matches_unfold_and_preserves_bn_parameters_modes_rng():
    model, inputs, loader, config = fixture()
    model.train()
    model.head.eval()  # restoration must preserve heterogeneous module modes
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    modes_before = [m.training for m in model.modules()]
    rng_before = torch.get_rng_state().clone()
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    grams, report = collect_grams(model, loader, config, "cpu")
    assert report["parameters_unchanged"] and report["bn_unchanged"] and report["rng_restored"]
    assert report["calibration_samples"] == 4 and report["layer_count"] == 1
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert random.getstate() == random_before
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_before))
    assert [m.training for m in model.modules()] == modes_before
    assert all(torch.equal(v, state_before[k]) for k, v in model.state_dict().items())
    model.eval()
    with torch.no_grad():
        activation = model.bn(model.stem(inputs))
        unfolded = F.unfold(activation, 3, padding=1).reshape(4, 2, 18, 25)
        for group in range(2):
            x = unfolded[:, group].permute(1, 0, 2).reshape(18, -1)
            expected = x @ x.T / x.shape[1]
            assert torch.allclose(grams["internal.weight"][group], expected, rtol=3e-6, atol=1e-7)


def test_timed_inverse_matches_reference_and_ordinary_dead_damping():
    torch.manual_seed(6)
    x = torch.randn(8, 16, dtype=torch.float64)
    h = x @ x.T / 16
    q, metrics = inverse_diagonal_timed(h, .01)
    reference, _, damping = inverse_diagonal_cholesky(h, percdamp=.01, dead_policy="reject")
    assert torch.equal(q, reference)
    assert metrics["cholesky_status"] == "PASS" and metrics["damping"] == damping
    assert metrics["cholesky_seconds"] >= 0 and metrics["inverse_diagonal_seconds"] >= 0
    dead_h = torch.diag(torch.tensor([0., 2., 4.], dtype=torch.float64))
    q, metrics = inverse_diagonal_timed(dead_h, .01)
    assert metrics["dead_coordinates"] == 1
    assert q[0] == pytest.approx(50.)
    with pytest.raises(FloatingPointError, match="damping"):
        inverse_diagonal_timed(torch.zeros(3, 3), .01)


def test_refresh_maps_grouped_input_features_and_spearman_is_stable():
    model, _, loader, config = fixture()
    config["reference_checks"] = True
    state = {"internal.weight": torch.zeros_like(model.internal.weight, dtype=torch.float64)}
    new, q, report = refresh(model, loader, state, config, "cpu")
    assert q["internal.weight"].shape == (2, 18)
    assert new["internal.weight"].shape == model.internal.weight.shape
    assert new["internal.weight"].dtype == torch.float64
    assert abs(float(new["internal.weight"].sum())) < 1e-12
    assert report["log_geometry_drift"]["rms"] > 0
    assert report["current_geometry_hash"] == tensor_sha(new)
    assert report["q_diagonal_hash"] == tensor_sha(q)
    references = report["layers"]["internal.weight"]["reference_checks"]
    assert len(references) == 2 and all(row["status"] == "PASS" for row in references)
    assert all(row["sample_shape"] == [18, 50] for row in references)
    assert torch.equal(new["internal.weight"][0], new["internal.weight"][1])
    assert torch.equal(new["internal.weight"][2], new["internal.weight"][3])
    again, q2, report2 = refresh(model, loader, new, config, "cpu", previous_q=q)
    assert torch.equal(q["internal.weight"], q2["internal.weight"])
    assert report2["spearman_stability"]["internal.weight"] == pytest.approx(1.)
    assert report2["current_geometry_hash"] != report["current_geometry_hash"]
    assert not any(v.requires_grad for v in again.values())
    assert spearman(torch.ones(4), torch.arange(4)) is None
    assert spearman(torch.tensor([1., 1., 2., 3.]), torch.tensor([3., 3., 2., 1.])) == pytest.approx(-1.)


def test_failure_removes_hooks_and_restores_modes_and_rng():
    model, _, loader, config = fixture()
    modes = [module.training for module in model.modules()]
    before = torch.get_rng_state().clone()
    config["expected_samples"] = 5
    with pytest.raises(ValueError, match="Calibration count"):
        collect_grams(model, loader, config, "cpu")
    assert torch.equal(torch.get_rng_state(), before)
    assert [module.training for module in model.modules()] == modes
    assert all(len(module._forward_pre_hooks) == 0 for module in model.modules())
    config["max_gram_bytes"] = 1
    with pytest.raises(MemoryError):
        collect_grams(model, loader, config, "cpu")


def test_multiple_layers_release_previous_gram_before_collecting_next(monkeypatch):
    model, inputs, loader, config = fixture()
    model.internal = nn.Sequential(model.internal, nn.Conv2d(4, 4, 1, bias=False))
    config["expected_layers"] = 2
    config["expected_coordinates"] = sum(layer.weight.numel() for layer in model.internal)
    state = {name + ".weight": torch.zeros_like(layer.weight, dtype=torch.float64)
             for name, layer in calibration.eligible_layers(model).items()}
    with pytest.raises(ValueError, match="Exactly one"):
        collect_grams(model, loader, config, "cpu")
    original = calibration.collect_layer_gram
    previous = []

    def checked(*args, **kwargs):
        if previous:
            assert previous[-1]() is None
        gram, report = original(*args, **kwargs)
        previous.append(weakref.ref(gram))
        return gram, report

    monkeypatch.setattr(calibration, "collect_layer_gram", checked)
    new, q, report = refresh(model, loader, state, config, "cpu")
    assert len(q) == 2 and report["model_forward_passes"] == 2
    assert report["total_forwarded_samples"] == 8
    assert report["calibration_samples"] == 4
    assert previous[-1]() is None
