"""Gate H: deterministic short synthetic CPU train/resume of HQ state."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from hq_apbr_v1 import (
    hq_global_l2_perturbation, historical_geometry_update,
    inverse_diagonal_cholesky,
)


CONFIG = {"rho": .05, "eta": .4, "lr": .03, "momentum": .9,
          "weight_decay": .01, "scheduler_t_max": 3, "calibration_size": 8}
CONFIG_SHA = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
DATA = torch.linspace(-1., 1., 48, dtype=torch.float32).reshape(12, 4)
LABELS = torch.tensor([0, 1] * 6)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def fresh():
    torch.manual_seed(337)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"],
                                momentum=CONFIG["momentum"],
                                weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["scheduler_t_max"])
    geometry = {"fc.weight": torch.zeros_like(model.fc.weight, dtype=torch.float64)}
    train_order = torch.Generator().manual_seed(37)
    calibration_order = torch.Generator().manual_seed(71)
    return model, optimizer, scheduler, geometry, train_order, calibration_order


def epoch(model, optimizer, scheduler, geometry, train_order, calibration_order):
    model.train()
    order = torch.randperm(12, generator=train_order).tolist()
    for start in range(0, 12, 4):
        indices = order[start:start + 4]
        inputs = DATA[indices] + .01 * torch.randn((4, 4))
        targets = LABELS[indices]
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets).backward()
        params = dict(model.named_parameters())
        first_grads = {name: p.grad for name, p in params.items()}
        epsilon, actual_norm, zero = hq_global_l2_perturbation(
            params, first_grads, geometry, CONFIG["rho"])
        assert not zero and abs(actual_norm / CONFIG["rho"] - 1) < 2e-5
        before = {name: p.detach().clone() for name, p in params.items()}
        with torch.no_grad():
            for name, p in params.items():
                p.add_(epsilon[name])
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets).backward()
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(before[name])
        optimizer.step()
    scheduler.step()
    model.eval()
    ids = torch.randperm(12, generator=calibration_order)[:CONFIG["calibration_size"]]
    calibration = DATA[ids].double()
    gram = calibration.T @ calibration / len(ids)
    q_diag, _, _ = inverse_diagonal_cholesky(gram, percdamp=.01)
    u = .5 * q_diag.log()
    target = u.repeat(2, 1)
    target -= target.mean()
    geometry = historical_geometry_update(geometry, {"fc.weight": target}, CONFIG["eta"])
    return geometry, order, ids.tolist()


def checkpoint(model, optimizer, scheduler, geometry, train_order, calibration_order, next_epoch):
    return {"schema": "hq_synthetic_cpu_resume_v1", "config_sha256": CONFIG_SHA,
            "next_epoch": next_epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "geometry": {k: v.clone() for k, v in geometry.items()},
            "torch_rng": torch.get_rng_state(),
            "train_order_rng": train_order.get_state(),
            "calibration_order_rng": calibration_order.get_state()}


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def test_h_short_synthetic_cpu_resume_bitwise(tmp_path):
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    full = fresh()
    full_history = []
    state = full[3]
    for _ in range(3):
        state, train_ids, calib_ids = epoch(full[0], full[1], full[2], state, full[4], full[5])
        full_history.append((train_ids, calib_ids))
    uninterrupted = checkpoint(full[0], full[1], full[2], state, full[4], full[5], 3)

    interrupted = fresh()
    state, train_ids, calib_ids = epoch(interrupted[0], interrupted[1], interrupted[2],
                                        interrupted[3], interrupted[4], interrupted[5])
    resumed_history = [(train_ids, calib_ids)]
    saved = checkpoint(interrupted[0], interrupted[1], interrupted[2], state,
                       interrupted[4], interrupted[5], 1)
    path = tmp_path / "hq_synthetic_short_checkpoint.pt"
    torch.save(saved, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["config_sha256"] == CONFIG_SHA
    assert restored["schema"] == "hq_synthetic_cpu_resume_v1"
    resumed = fresh()
    resumed[0].load_state_dict(restored["model"])
    resumed[1].load_state_dict(restored["optimizer"])
    resumed[2].load_state_dict(restored["scheduler"])
    resumed[4].set_state(restored["train_order_rng"])
    resumed[5].set_state(restored["calibration_order_rng"])
    torch.set_rng_state(restored["torch_rng"])
    state = restored["geometry"]
    for _ in range(restored["next_epoch"], 3):
        state, train_ids, calib_ids = epoch(resumed[0], resumed[1], resumed[2], state, resumed[4], resumed[5])
        resumed_history.append((train_ids, calib_ids))
    after_resume = checkpoint(resumed[0], resumed[1], resumed[2], state,
                              resumed[4], resumed[5], 3)
    assert resumed_history == full_history
    assert_nested_equal(after_resume, uninterrupted)
