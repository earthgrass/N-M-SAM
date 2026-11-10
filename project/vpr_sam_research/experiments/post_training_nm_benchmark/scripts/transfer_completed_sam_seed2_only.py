"""Transfer the completed, user-approved SAM Seed 2 Final without queue changes.

The serial controller is paused while its current Seed 2 training child finishes.
This transfer checks the independent remote SHA256, local SHA256, and checkpoint
identity. The controller is stopped separately after the transfer passes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sync_dense_sam_apbr_l2_seed1to4_local import (  # noqa: E402
    OUT, REMOTE_RUN, FILES, connect, copy_resumable, read_json, sha,
)

REMOTE_DIR = REMOTE_RUN + "/seed2/dense_sam"
LOCAL_DIR = OUT / "seed2/dense_sam"
EXPECTED_INIT = "ca8bb0ef702d07857572203eceb8acb56190a7a0b9324055555bae0c056e440e"


def remote_sha(client, path: str) -> str:
    stdin, stdout, stderr = client.exec_command("sha256sum " + path)
    del stdin
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote sha256sum failed ({code}): {stderr.read().decode()}")
    value = stdout.read().decode().split()[0]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Malformed remote SHA256")
    return value


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    client = connect()
    try:
        sftp = client.open_sftp()
        status = read_json(sftp, REMOTE_DIR + "/status.json")
        publication = read_json(sftp, REMOTE_DIR + "/checkpoint_publication.json")
        identity = read_json(sftp, REMOTE_DIR + "/identity.json")
        config = ROOT / "configs/cifar100_50k10k_final_v1_sam_seed245_rtx5090_20261103.yaml"
        if (status["status"] != "completed" or status["completed_epochs"] != 200
                or status["steps"] != 78200 or status["test_accessed"] is not False
                or status["selection_mode"] != "fixed_final_epoch"
                or publication["epoch"] != 200 or publication["test_accessed"] is not False
                or identity["seed"] != 2 or identity["method"] != "dense_sam"
                or identity["initial_model_sha256"] != EXPECTED_INIT
                or identity["config_sha256"] != sha(config)
                or identity["train_size"] != 50000 or identity["validation_size"] != 0
                or identity["test_accessed"] is not False):
            raise ValueError("Remote SAM Seed 2 Final identity not complete")
        remote_checkpoint_sha = remote_sha(client, REMOTE_DIR + "/final_epoch_200.pt")
        if (remote_checkpoint_sha != publication["sha256"]
                or remote_checkpoint_sha != status["final_checkpoint_sha256"]):
            raise ValueError("Remote checkpoint differs from publication SHA256")
        hashes = {}
        for name in FILES:
            expected = remote_checkpoint_sha if name == "final_epoch_200.pt" else None
            hashes[name] = copy_resumable(sftp, REMOTE_DIR + "/" + name,
                                          LOCAL_DIR / name, expected)
        log_name = "seed2_dense_sam.log"
        hashes[log_name] = copy_resumable(sftp, REMOTE_RUN + "/" + log_name,
                                          OUT / "logs" / log_name)
        remote_log_sha = remote_sha(client, REMOTE_RUN + "/" + log_name)
        if hashes[log_name] != remote_log_sha:
            raise ValueError("Remote/local SAM Seed 2 log SHA256 mismatch")
        parent = torch.load(LOCAL_DIR / "final_epoch_200.pt", map_location="cpu", weights_only=False)
        if (parent["epoch"] != 200 or parent["steps"] != 78200
                or parent["protocol_id"] != "cifar100_50k10k_final_v1"
                or parent["test_accessed"] is not False or parent["status"] != "healthy"
                or parent["identity"]["seed"] != 2
                or parent["identity"]["method"] != "dense_sam"
                or parent["identity"]["initial_model_sha256"] != EXPECTED_INIT):
            raise ValueError("Local SAM Seed 2 checkpoint contract failed")
        l2proof = json.loads((OUT / "seed2/apbr_fixed_l2/LOCAL_SHA256.json").read_text(encoding="utf-8"))
        if l2proof["job"]["initial_model_sha256"] != EXPECTED_INIT:
            raise ValueError("SAM/APBR-L2 paired initialization mismatch")
        queue = read_json(sftp, REMOTE_RUN + "/queue_status.json")
        write_new(LOCAL_DIR / "queue_status_at_transfer.json", queue)
        proof = {"status": "TRANSFER_VERIFIED", "seed": 2, "method": "dense_sam",
                 "checkpoint_sha256": remote_checkpoint_sha,
                 "initial_model_sha256": EXPECTED_INIT, "file_sha256": hashes,
                 "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                 "remote_files_deleted": False, "test_accessed": False}
        write_new(LOCAL_DIR / "LOCAL_SHA256.json", proof)
        print(json.dumps({"status": proof["status"], "checkpoint_sha256": remote_checkpoint_sha,
                          "file_count": len(hashes)}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
