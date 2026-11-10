from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_apbr_parent_magnitude_ablation as ablation
import run_cifar100_50k_formal_training as train


def checkpoint(seed: int = 1, method: str = "apbr_sam") -> dict:
    model = {"weight": torch.tensor([1.0])}
    identity = {"method": method, "seed": seed}
    return {
        "schema": 1,
        "status": "healthy",
        "protocol_id": ablation.PROTOCOL,
        "epoch": 200,
        "steps": 78_200,
        "model": model,
        "model_sha256": train.tensor_sha(model),
        "stepper": {},
        "scheduler": {},
        "history": [],
        "identity": identity,
        "rng": {},
        "test_accessed": False,
    }


def test_accepts_only_matching_apbr_epoch200_parent() -> None:
    ablation.validate_apbr_parent(checkpoint(seed=2), seed=2)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"epoch": 199}, "epoch-200"),
        ({"steps": 77_809}, "epoch-200"),
        ({"test_accessed": True}, "Test policy"),
    ],
)
def test_rejects_invalid_parent_fields(mutation: dict, expected: str) -> None:
    value = checkpoint()
    value.update(mutation)
    with pytest.raises(ValueError, match=expected):
        ablation.validate_apbr_parent(value, seed=1)


def test_rejects_dense_sam_or_wrong_seed_parent() -> None:
    with pytest.raises(ValueError, match="matching APBR-SAM"):
        ablation.validate_apbr_parent(checkpoint(method="dense_sam"), seed=1)
    with pytest.raises(ValueError, match="matching APBR-SAM"):
        ablation.validate_apbr_parent(checkpoint(seed=0), seed=1)
