"""Prove historical SAM2 result and new transferred Final share model tensors."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/sam_seed2_historical_final_verified_20261104/last.pt"
NEW = ROOT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy/seed2/dense_sam/final_epoch_200.pt"
PROOF = NEW.parent / "LOCAL_SHA256.json"
RESULT = ROOT / "artifacts/sam_conditional_seed2_cpu_20261104_v1/results.json"
AUDIT = ROOT / "artifacts/sam_conditional_seed2_cpu_20261104_v1/independent_audit.json"
OUTPUT = NEW.parent / "DENSE_IDENTITY_AND_RESULT_ORIGIN.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old_sha, new_sha = sha(OLD), sha(NEW)
    transfer = json.loads(PROOF.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    old = torch.load(OLD, map_location="cpu", weights_only=False)
    new = torch.load(NEW, map_location="cpu", weights_only=False)
    if (old_sha != result["parent_sha256"]
            or new_sha != transfer["checkpoint_sha256"]
            or result["status"] != "COMPLETE_CPU_EXPLORATORY_SEED2"
            or audit["status"] != "INDEPENDENT_ARTIFACT_AUDIT_PASS"
            or old["identity"]["seed"] != new["identity"]["seed"] != 2
            or old["identity"]["method"] != new["identity"]["method"] != "dense_sam"
            or old["epoch"] != new["epoch"] != 200
            or old["steps"] != new["steps"] != 78200
            or old["test_accessed"] is not False or new["test_accessed"] is not False):
        raise ValueError("SAM2 handoff identity mismatch")
    old_state, new_state = old["model"], new["model"]
    if old_state.keys() != new_state.keys():
        raise ValueError("SAM2 state_dict key mismatch")
    mismatches = [key for key in old_state if not torch.equal(old_state[key], new_state[key])]
    if mismatches or old["model_sha256"] != new["model_sha256"]:
        raise ValueError(f"SAM2 model tensor mismatch: {mismatches[:4]}")
    payload = {
        "status": "SAM2_DENSE_STATE_BITWISE_EQUAL",
        "historical_checkpoint_sha256": old_sha,
        "new_transferred_checkpoint_sha256": new_sha,
        "historical_config_sha256": old["identity"]["config_sha256"],
        "new_config_sha256": new["identity"]["config_sha256"],
        "model_state_sha256": old["model_sha256"],
        "state_dict_tensor_count": len(old_state),
        "all_model_parameters_and_buffers_bitwise_equal": True,
        "historical_conditional_checkpoint_sha256": result["checkpoint_sha256"],
        "historical_conditional_mask_sha256": result["mask_sha256"],
        "historical_conditional_test_count": result["test"]["physical"]["test_count"],
        "historical_conditional_accuracy": result["test"]["physical"]["test_accuracy"],
        "historical_conditional_ce": result["test"]["physical"]["test_ce"],
        "new_pruning_or_test_executed": False,
        "finding": "The previous fixed calibration/pruning/Test result starts from a bitwise-identical Dense model state; checkpoint metadata and config files differ.",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(OUTPUT)
    print(json.dumps({"status": payload["status"], "tensor_count": len(old_state),
                      "accuracy": payload["historical_conditional_accuracy"]}))


if __name__ == "__main__":
    main()
