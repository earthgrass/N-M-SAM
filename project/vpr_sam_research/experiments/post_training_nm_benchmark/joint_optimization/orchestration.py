"""Auditable orchestration primitives for the isolated APBR-GPT experiment.

This module contains no dataset or CUDA side effects.  It centralises the
formal schedule, atomic checkpoint publication, mask auditing, and the guard
that prevents Final Test evaluation before every checkpoint is frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from baseline_adapters.common.nm import validate_keep_mask


@dataclass(frozen=True)
class FormalSchedule:
    epochs: int = 200
    warmup_epochs: int = 5
    hessian_refresh_epochs: int = 10

    def __post_init__(self) -> None:
        if self.epochs != 200 or self.warmup_epochs != 5 or self.hessian_refresh_epochs != 10:
            raise ValueError("The Phase-3 schedule is frozen to 200/5/10")

    def pruning_active(self, epoch: int) -> bool:
        if not 0 <= epoch < self.epochs:
            raise ValueError("Epoch outside formal schedule")
        return epoch >= self.warmup_epochs

    def refresh_due(self, epoch: int) -> bool:
        return self.pruning_active(epoch) and (epoch - self.warmup_epochs) % self.hessian_refresh_epochs == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class AtomicCheckpointManager:
    """Publish only completed optimizer steps using atomic replacement."""

    def __init__(self, output: Path) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, payload: Mapping) -> tuple[Path, str]:
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Checkpoint name must be a local .pt filename")
        target = self.output / name
        temporary = target.with_suffix(".pt.tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
        return target, sha256_file(target)

    def load(self, name: str, *, expected_sha256: str | None = None) -> dict:
        path = self.output / name
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise RuntimeError("Checkpoint SHA256 mismatch")
        return torch.load(path, map_location="cpu", weights_only=False)


def audit_nm_masks(
    model: nn.Module,
    layers: Mapping[str, nn.Conv2d],
    masks: Mapping[str, Tensor],
) -> dict:
    if set(masks) != set(layers):
        raise ValueError("Mask/layer identity mismatch")
    eligible = deleted = 0
    for name, layer in layers.items():
        mask = masks[name]
        if mask.dtype != torch.bool or mask.shape != layer.weight.shape:
            raise ValueError(f"Invalid bool mask identity: {name}")
        flat = mask.reshape(mask.shape[0], -1)
        validate_keep_mask(flat, keep_n=2, m=4)
        if torch.count_nonzero(layer.weight.detach()[~mask]):
            raise RuntimeError(f"Deleted weights are nonzero: {name}")
        eligible += mask.numel()
        deleted += int((~mask).sum())
    if eligible != 11_157_504 or deleted != 5_578_752:
        raise RuntimeError(f"Frozen ResNet-18 2:4 count mismatch: {eligible}/{deleted}")
    return {"eligible_coordinates": eligible, "deleted_coordinates": deleted,
            "eligible_layers": len(layers), "strict_2_4": True}


def write_manifest(path: Path, payload: Mapping) -> None:
    value = dict(payload)
    value.setdefault("final_test_accessed", False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class FinalEvaluationGuard:
    """Permit one evaluation only after a hash-locked freeze manifest exists."""

    def __init__(self, freeze_manifest: Path) -> None:
        self.freeze_manifest = Path(freeze_manifest)

    def run_once(self, evaluator: Callable[[], Mapping], output: Path) -> dict:
        if output.exists():
            raise RuntimeError("Final evaluation already exists")
        manifest = json.loads(self.freeze_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "CHECKPOINTS_FROZEN" or not manifest.get("checkpoint_sha256"):
            raise RuntimeError("Final evaluation requires frozen checkpoint hashes")
        result = dict(evaluator())
        result.update({"selection_used": False, "single_final_evaluation": True})
        write_manifest(output, result)
        return result


def formal_manifest() -> dict:
    schedule = FormalSchedule()
    return {
        "experiment_id": "apbr_gpt_joint_seed0_rebased",
        "schedule": asdict(schedule),
        "hessian_refresh_epochs_zero_based": [
            epoch for epoch in range(schedule.epochs) if schedule.refresh_due(epoch)
        ],
        "fixed_lambda_eta": True,
        "training_test_access": False,
        "final_test_policy": "one evaluation after checkpoint hash freeze",
        "automatic_retry": False,
    }
