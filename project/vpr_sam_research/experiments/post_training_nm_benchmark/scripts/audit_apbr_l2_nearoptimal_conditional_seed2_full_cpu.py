"""Read-only validation of frozen near-optimal conditional OBS checkpoints.

This script does not load CIFAR-100 Test or perform another evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.common.nm import validate_keep_mask
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_apbr_l2_nearoptimal_conditional_seed2_full_cpu import (
    METHODS, OUTPUT, PARENT_SHA256, L2_RUN, current_sources, write_new,
)


def main() -> None:
    preflight = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUTPUT / "pre_test_freeze.json").read_text(encoding="utf-8"))
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    numerical = json.loads((OUTPUT / "pretest_numerical_audit.json").read_text(encoding="utf-8"))
    formal_path = ROOT / "configs/cifar100_50k10k_apbr_fixed_l2_seed1to4_rtx5090_20261103.yaml"
    if (preflight["source_sha256"] != current_sources(formal_path)
            or preflight["source_sha256"] != freeze["source_sha256"]
            or results["source_sha256"] != freeze["source_sha256"]):
        raise ValueError("Source/config hash mismatch")
    if (freeze["status"] != "ALL_19_LAYER_CHECKPOINTS_FROZEN_BEFORE_TEST"
            or freeze["test_accessed"] is not False
            or numerical["status"] != "PASS"
            or results["status"] != "COMPLETE_CPU_EXPLORATORY_SEED2"
            or results["test_accessed_during_calibration_pruning"] is not False):
        raise ValueError("Freeze, numerical audit, or Test-isolation status mismatch")
    parent_path = L2_RUN / "final_epoch_200.pt"
    if file_sha(parent_path) != PARENT_SHA256:
        raise ValueError("Parent SHA256 mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent["test_accessed"] is not False:
        raise ValueError("Parent Test isolation mismatch")
    checkpoints = {}
    masks = {}
    for method in METHODS:
        path = OUTPUT / f"{method}_compensated.pt"
        if file_sha(path) != freeze["checkpoint_sha256"][method] or file_sha(path) != results["checkpoint_sha256"][method]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["parent_sha256"] != PARENT_SHA256 or payload["method"] != method
                or payload["status"] != "FROZEN_BEFORE_FINAL_TEST"
                or payload["test_accessed_during_calibration_pruning"] is not False
                or payload["recovery_epochs"] != 0 or payload["bn_only_calibration"] is not False):
            raise ValueError(f"Checkpoint provenance mismatch: {method}")
        if len(payload["masks"]) != 19 or set(payload["model"]) != set(parent["model"]):
            raise ValueError(f"Layer/state key mismatch: {method}")
        if tensor_sha(payload["masks"]) != payload["mask_sha256"] or payload["mask_sha256"] != freeze["mask_sha256"][method]:
            raise ValueError(f"Mask SHA256 mismatch: {method}")
        deleted = 0
        for key, value in payload["model"].items():
            if key in payload["masks"]:
                keep = payload["masks"][key]
                if keep.dtype != torch.bool or keep.shape != value.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Invalid tensor: {method}/{key}")
                validate_keep_mask(keep.reshape(keep.shape[0], -1), 2, 4)
                if int(torch.count_nonzero(value[~keep])):
                    raise ValueError(f"Pruned weight regrowth: {method}/{key}")
                deleted += int((~keep).sum())
            elif not torch.equal(value, parent["model"][key]):
                raise ValueError(f"Excluded weight or BN changed: {method}/{key}")
        if deleted != 5578752 or results["raw_hard_test"][method]["test_count"] != 10000:
            raise ValueError(f"Deletion count or Test count mismatch: {method}")
        checkpoints[method] = {"checkpoint_sha256": file_sha(path),
                               "mask_sha256": payload["mask_sha256"], "deleted": deleted}
        masks[method] = payload["masks"]
    keys = sorted(masks["true"])
    def difference(a: str, b: str) -> dict:
        positions = groups = 0
        for key in keys:
            x = masks[a][key].reshape(-1, 4)
            y = masks[b][key].reshape(-1, 4)
            positions += int((x != y).sum())
            groups += int((x != y).any(dim=1).sum())
        return {"positions": positions, "groups": groups}
    audit = {
        "status": "INDEPENDENT_ARTIFACT_AUDIT_PASS", "test_file_opened_by_audit": False,
        "parent_sha256": PARENT_SHA256,
        "source_sha256": preflight["source_sha256"],
        "checkpoint": checkpoints,
        "true_vs_unit": difference("true", "unit"),
        "true_vs_shuffled": difference("true", "shuffled"),
        "unit_vs_physical": difference("unit", "physical"),
        "physical_replay_bitwise_19_layers": "not_applicable_no_prior_l2_conditional_checkpoint",
        "test_count_per_method": 10000,
    }
    write_new(OUTPUT / "independent_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
