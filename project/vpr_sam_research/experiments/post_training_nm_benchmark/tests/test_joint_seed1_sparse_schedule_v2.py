"""Frozen temporal guidance schedule and independent seed initialization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "joint_seed01_v2", ROOT / "scripts/run_apbr_gpt_joint_training_seed01_v2.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_full_and_epoch10_guidance_budgets() -> None:
    full = [epoch + 1 for epoch in range(200) if MODULE.guidance_due(epoch, "full")]
    sparse = [epoch + 1 for epoch in range(200) if MODULE.guidance_due(epoch, "epoch10")]
    assert full == list(range(6, 201))
    assert sparse == list(range(10, 201, 10))
    assert len(sparse) == 20
    with pytest.raises(ValueError):
        MODULE.guidance_due(200, "epoch10")


def test_seed1_initial_asset_is_independent() -> None:
    seed0 = ROOT / "artifacts/apbr_gpt_joint_seed0_rebased/server_bundle_v3/model_assets/initial_common_seed0.pt"
    seed1 = ROOT / "artifacts/apbr_gpt_joint_seed1_sparse_v2/local_seed1_asset/initial_common_seed1.pt"
    if not seed0.exists() or not seed1.exists():
        pytest.skip("Frozen local assets not present")
    a = torch.load(seed0, map_location="cpu", weights_only=False)
    b = torch.load(seed1, map_location="cpu", weights_only=False)
    assert a["seed"] == 0 and b["seed"] == 1
    assert a["model_tensor_sha256"] != b["model_tensor_sha256"]
    assert b["geometry_tensor_sha256"]
    assert b["random_initialization"] is True
