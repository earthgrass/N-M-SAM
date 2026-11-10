"""Independent frozen-artifact audit; never reevaluates CIFAR-100 Test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.common.nm import validate_keep_mask
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA
from scripts.run_sam_apbr_conditional_seed1_cpu import COHORT, EXPECTED, OUT, source_sha, write_new


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pre = read(OUT / "preflight.json")
    freeze = read(OUT / "pre_test_freeze.json")
    result = read(OUT / "results.json")
    if (pre["source_sha256"] != source_sha(ROOT / "configs/cifar100_50k10k_final_v1.yaml")
            or pre["source_sha256"] != result["source_sha256"]
            or freeze["source_sha256"] != result["source_sha256"]
            or freeze["test_accessed"] is not False
            or result["test_accessed_during_pruning"] is not False
            or result["status"] != "COMPLETE_CPU_EXPLORATORY_SEED1"):
        raise ValueError("Source, freeze or Test isolation mismatch")
    test_path = DATA / "extracted/cifar-100-python/test"
    if file_sha(test_path) != result["test_pickle_sha256"]:
        raise ValueError("Official Test pickle hash mismatch")
    output = {}
    for method, (kind, parent_sha, _mode) in EXPECTED.items():
        parent_path = COHORT / kind / "final_epoch_200.pt"
        checkpoint_path = OUT / f"{method}.pt"
        if file_sha(parent_path) != parent_sha or file_sha(checkpoint_path) != freeze["checkpoint_sha256"][method]:
            raise ValueError(f"Checkpoint hash mismatch: {method}")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (ckpt["parent_sha256"] != parent_sha or ckpt["seed"] != 1
                or ckpt["status"] != "FROZEN_BEFORE_TEST"
                or ckpt["test_accessed_during_pruning"] is not False
                or ckpt["recovery_epochs"] != 0
                or ckpt["calibration_ids_sha256"] != pre["calibration_ids_sha256"]
                or tensor_sha(ckpt["masks"]) != freeze["mask_sha256"][method]
                or len(ckpt["masks"]) != 19 or set(ckpt["model"]) != set(parent["model"])):
            raise ValueError(f"Provenance/Mask mismatch: {method}")
        deletion_count = 0
        for key, weight in ckpt["model"].items():
            if key in ckpt["masks"]:
                keep = ckpt["masks"][key]
                if keep.dtype != torch.bool or keep.shape != weight.shape or not bool(torch.isfinite(weight).all()):
                    raise ValueError(f"Tensor invalid: {method}/{key}")
                validate_keep_mask(keep.reshape(keep.shape[0], -1), 2, 4)
                if int(torch.count_nonzero(weight[~keep])):
                    raise ValueError(f"Weight regrowth: {method}/{key}")
                deletion_count += int((~keep).sum())
            elif not torch.equal(weight, parent["model"][key]):
                raise ValueError(f"Excluded weight or BN changed: {method}/{key}")
        if deletion_count != 5578752 or result["test"][method]["test_count"] != 10000:
            raise ValueError(f"Deletion/Test count mismatch: {method}")
        output[method] = {"parent_sha256": parent_sha,
                          "checkpoint_sha256": file_sha(checkpoint_path),
                          "mask_sha256": tensor_sha(ckpt["masks"]),
                          "deletions": deletion_count,
                          "test_accuracy": result["test"][method]["test_accuracy"]}
    audit = {"status": "INDEPENDENT_ARTIFACT_AUDIT_PASS", "seed": 1,
             "test_reevaluated": False, "test_accessed_during_pruning": False,
             "methods": output}
    write_new(OUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
