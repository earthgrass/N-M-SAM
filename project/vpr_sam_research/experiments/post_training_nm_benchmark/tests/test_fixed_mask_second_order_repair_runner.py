from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_cifar100_50k_formal_training as train
import run_fixed_mask_second_order_repair as runner


def parent(seed: int = 1) -> dict:
    model = {"weight": torch.tensor([1.0])}
    identity = {"method": "apbr_sam", "seed": seed}
    return {
        "schema": 1, "status": "healthy", "protocol_id": runner.PROTOCOL,
        "epoch": 200, "steps": 78_200, "model": model,
        "model_sha256": train.tensor_sha(model), "stepper": {}, "scheduler": {},
        "history": [], "identity": identity, "rng": {}, "test_accessed": False,
    }


def test_parent_gate_accepts_only_matching_fixed_apbr_final() -> None:
    runner.validate_parent(parent(2), 2)
    for key, value in [("epoch", 199), ("steps", 78_199), ("test_accessed", True)]:
        item = parent()
        item[key] = value
        with pytest.raises(ValueError):
            runner.validate_parent(item, 1)
    wrong = parent()
    wrong["identity"] = {"method": "dense_sam", "seed": 1}
    with pytest.raises(ValueError, match="matching APBR-SAM"):
        runner.validate_parent(wrong, 1)


def test_calibration_ids_are_frozen_and_hashable() -> None:
    ids = runner.calibration_ids()
    assert len(ids) == 1024
    assert len(set(ids)) == 1024
    assert min(ids) >= 0 and max(ids) < 50_000
    assert runner.calibration_ids() == ids
    assert len(runner.ids_sha256(ids)) == 64
