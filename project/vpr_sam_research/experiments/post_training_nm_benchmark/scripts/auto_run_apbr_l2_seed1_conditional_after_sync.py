"""Wait for verified Seed 1 Dense Final, then run one CPU comparison once."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy"
OUT = ROOT / "artifacts/apbr_l2_conditional_seed1_autorun_20261103_v2"
RUN = ROOT / "scripts/run_apbr_l2_nearoptimal_conditional_seed1_full_cpu.py"
AUDIT = ROOT / "scripts/audit_apbr_l2_nearoptimal_conditional_seed1_full_cpu.py"
COMPARE = ROOT / "scripts/compare_sam_apbr_apbr_l2_conditional_seed01.py"
RESULTS = ROOT / "artifacts/apbr_l2_nearoptimal_conditional_seed1_full_cpu_20261103_v1"
SOURCES = [RUN, AUDIT, COMPARE,
           ROOT / "baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py",
           ROOT / "baseline_adapters/sparsegpt/adapter.py",
           ROOT / "configs/cifar100_50k10k_apbr_fixed_l2_seed1to4_rtx5090_20261103.yaml",
           ROOT / "baseline_adapters/cifar100_model.py",
           ROOT / "benchmark/nm.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def status(value: dict) -> None:
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = OUT / "status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def execute(script: Path, log_name: str) -> None:
    with (OUT / log_name).open("wb") as log:
        result = subprocess.run([sys.executable, "-u", str(script)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{script.name} failed with code {result.returncode}; see {OUT / log_name}")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Independent one-shot watcher output already exists: {OUT}")
    OUT.mkdir(parents=True)
    source_hash = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    state = {"status": "WAITING_FOR_VERIFIED_L2_SEED1_FINAL",
             "source_sha256": source_hash,
             "automatic_retry": False, "remote_queue_untouched": True}
    status(state)
    while True:
        sync_file = SYNC / "sync_status.json"
        if sync_file.exists():
            sync = json.loads(sync_file.read_text(encoding="utf-8"))
            expected = sync.get("completed_local_checkpoints", {}).get("seed1/apbr_fixed_l2")
            if expected:
                parent = SYNC / "seed1/apbr_fixed_l2/final_epoch_200.pt"
                local_proof = SYNC / "seed1/apbr_fixed_l2/LOCAL_SHA256.json"
                if parent.is_file() and local_proof.is_file():
                    proof = json.loads(local_proof.read_text(encoding="utf-8"))
                    if (proof["job"]["checkpoint_sha256"] != expected
                            or proof["file_sha256"]["final_epoch_200.pt"] != expected
                            or sha(parent) != expected):
                        raise ValueError("Verified local Seed 1 Dense Final identity mismatch")
                    break
            if sync["remote_status"] == "failed":
                raise RuntimeError("Remote queue failed before Seed 1 APBR-L2 completion")
        time.sleep(60)
    if {str(p.relative_to(ROOT)): sha(p) for p in SOURCES} != source_hash:
        raise RuntimeError("Source/config changed while waiting; refusing CPU pruning and Test")
    if RESULTS.exists():
        raise FileExistsError(f"Experiment output already exists: {RESULTS}")
    state["status"] = "RUNNING_CPU_CONDITIONAL_OBS"
    state["parent_sha256"] = expected
    status(state)
    execute(RUN, "run.log")
    state["status"] = "RUNNING_INDEPENDENT_AUDIT"
    status(state)
    execute(AUDIT, "audit.log")
    state["status"] = "BUILDING_TWO_SEED_COMPARISON"
    status(state)
    execute(COMPARE, "comparison.log")
    if {str(p.relative_to(ROOT)): sha(p) for p in SOURCES} != source_hash:
        raise RuntimeError("Source/config changed during execution")
    state["status"] = "COMPLETE"
    state["results_json"] = str(RESULTS / "results.json")
    state["comparison_json"] = str(ROOT / "docs/research/sam_apbr_l2_conditional_seed01_20261103/comparison.json")
    status(state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        status({"status": "FAILED", "automatic_retry": False,
                "traceback": traceback.format_exc()})
        raise
