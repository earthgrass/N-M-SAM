"""Independent checkpoint and mask audit for the new SAM Seed 2 CPU result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.common.nm import validate_keep_mask
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_sam_conditional_seed2_newfinal_cpu import SYNC, sources, write_new


def main(seed: int) -> None:
    output = ROOT / "artifacts/sam_conditional_seed2_newfinal_cpu_20261104_v1"
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    freeze = json.loads((output / "pre_test_freeze.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if (preflight["source_sha256"] != sources()
            or freeze["source_sha256"] != sources()
            or results["source_sha256"] != sources()
            or freeze["status"] != "ALL_19_LAYER_CHECKPOINTS_FROZEN_BEFORE_TEST"
            or freeze["test_accessed"] is not False
            or results["status"] != f"COMPLETE_CPU_EXPLORATORY_SEED{seed}"
            or results["test_accessed_during_calibration_pruning"] is not False
            or results["test"]["dense"]["test_count"] != 10000
            or results["test"]["physical"]["test_count"] != 10000):
        raise ValueError("Source, freeze, Test, or result identity mismatch")
    local_dir = SYNC / f"seed{seed}/dense_sam"
    proof = json.loads((local_dir / "LOCAL_SHA256.json").read_text(encoding="utf-8"))
    parent_path = local_dir / "final_epoch_200.pt"
    parent_sha = proof["checkpoint_sha256"]
    if (file_sha(parent_path) != parent_sha or parent_sha != preflight["parent_sha256"]
            or parent_sha != freeze["parent_sha256"]
            or parent_sha != results["parent_sha256"]):
        raise ValueError("Parent checkpoint SHA256 mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    checkpoint_path = output / "physical_compensated.pt"
    checkpoint_sha = file_sha(checkpoint_path)
    if checkpoint_sha != freeze["checkpoint_sha256"] or checkpoint_sha != results["checkpoint_sha256"]:
        raise ValueError("Compensated checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload["status"] != "FROZEN_BEFORE_FINAL_TEST" or payload["seed"] != seed
            or payload["parent_sha256"] != parent_sha
            or payload["recovery_epochs"] != 0
            or payload["bn_only_calibration"] is not False
            or payload["test_accessed_during_pruning"] is not False
            or len(payload["masks"]) != 19
            or set(payload["model"]) != set(parent["model"])
            or tensor_sha(payload["masks"]) != payload["mask_sha256"]
            or payload["mask_sha256"] != freeze["mask_sha256"]):
        raise ValueError("Compensated checkpoint or Mask provenance mismatch")
    deleted = 0
    for key, value in payload["model"].items():
        if key in payload["masks"]:
            keep = payload["masks"][key]
            if keep.dtype != torch.bool or keep.shape != value.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid tensor: {key}")
            validate_keep_mask(keep.reshape(keep.shape[0], -1), 2, 4)
            if int(torch.count_nonzero(value[~keep])):
                raise ValueError(f"Deleted weight regrowth: {key}")
            deleted += int((~keep).sum())
        elif not torch.equal(value, parent["model"][key]):
            raise ValueError(f"Excluded/BN state changed: {key}")
    if deleted != 5578752 or results["deletions"] != 5578752:
        raise ValueError("2:4 deletion count mismatch")
    write_new(output / "independent_audit.json", {
        "status": "INDEPENDENT_ARTIFACT_AUDIT_PASS", "seed": seed,
        "test_file_opened_by_audit": False, "parent_sha256": parent_sha,
        "checkpoint_sha256": checkpoint_sha, "mask_sha256": payload["mask_sha256"],
        "layer_count": 19, "deleted": deleted, "source_sha256": sources(),
        "test_count_per_endpoint": 10000})
    print(f"Seed {seed} independent audit PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2,), required=True)
    main(parser.parse_args().seed)
