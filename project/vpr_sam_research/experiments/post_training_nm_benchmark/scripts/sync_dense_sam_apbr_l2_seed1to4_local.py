"""Resumable, read-only remote-to-local Dense Final synchronization.

Polls the remote serial queue, copies each completed job immediately, and
verifies its Final checkpoint SHA256. Does not alter or restart remote work.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy"
REMOTE = "/root/autodl-tmp/apbr_l2_dense_seed1to4_20261103"
REMOTE_RUN = REMOTE + "/artifacts/run_v1"
HOST = "connect.weste.seetacloud.com"
PORT = 32555
KEY = Path.home() / ".ssh/codex_apbr_seed245_nopass_20261103"
KNOWN = Path.home() / ".ssh/known_hosts"
FILES = ("final_epoch_200.pt", "identity.json", "resolved_config.json",
         "status.json", "checkpoint_publication.json", "epoch_metrics.csv",
         "epoch_metrics.jsonl")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def connect():
    client = paramiko.SSHClient()
    client.load_host_keys(str(KNOWN))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(HOST, port=PORT, username="root", key_filename=str(KEY),
                   look_for_keys=False, allow_agent=False, timeout=20)
    return client


def read_json(sftp, remote: str):
    with sftp.open(remote, "rb") as stream:
        return json.loads(stream.read().decode("utf-8"))


def copy_resumable(sftp, remote: str, local: Path, expected_sha: str | None = None) -> str:
    local.parent.mkdir(parents=True, exist_ok=True)
    size = sftp.stat(remote).st_size
    if local.exists():
        if local.stat().st_size == size and (expected_sha is None or sha(local) == expected_sha):
            return sha(local)
        raise ValueError(f"Existing local file differs; refusing overwrite: {local}")
    part = local.with_name(local.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Partial file exceeds remote size: {part}")
    with sftp.open(remote, "rb") as source, part.open("ab") as target:
        source.seek(offset)
        while offset < size:
            chunk = source.read(min(4 * 1024 * 1024, size - offset))
            if not chunk:
                raise OSError(f"Short remote read: {remote}")
            target.write(chunk)
            offset += len(chunk)
        target.flush()
    if part.stat().st_size != size:
        raise OSError(f"Transfer size mismatch: {remote}")
    digest = sha(part)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA256 mismatch: {remote}; preserved {part}")
    part.replace(local)
    return digest


def once() -> bool:
    client = connect()
    try:
        sftp = client.open_sftp()
        queue = read_json(sftp, REMOTE_RUN + "/queue_status.json")
        atomic_json(OUT / "remote_queue_status.json", queue)
        completed = {}
        for job in queue["jobs"]:
            if job["state"] != "completed":
                continue
            name = f"seed{job['seed']}/{job['method']}"
            remote_dir = REMOTE_RUN + "/" + name
            local_dir = OUT / name
            hashes = {}
            for filename in FILES:
                remote_file = remote_dir + "/" + filename
                expected = job["checkpoint_sha256"] if filename == "final_epoch_200.pt" else None
                hashes[filename] = copy_resumable(sftp, remote_file, local_dir / filename, expected)
            log_name = f"seed{job['seed']}_{job['method']}.log"
            hashes[log_name] = copy_resumable(sftp, REMOTE_RUN + "/" + log_name, OUT / "logs" / log_name)
            atomic_json(local_dir / "LOCAL_SHA256.json", {"job": job, "file_sha256": hashes,
                        "verified_at_utc": datetime.now(timezone.utc).isoformat()})
            completed[name] = job["checkpoint_sha256"]
        state = {"remote_status": queue["status"], "completed_local_checkpoints": completed,
                 "expected_jobs": len(queue["jobs"]), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "remote_files_deleted": False}
        atomic_json(OUT / "sync_status.json", state)
        if queue["status"] in ("failed", "completed"):
            copy_resumable(sftp, REMOTE + "/controller.log", OUT / "controller.log")
            if queue["status"] == "failed":
                return True
            if len(completed) != len(queue["jobs"]):
                raise RuntimeError("Queue complete but local copies incomplete")
            state["status"] = "COMPLETE_ALL_SHA256_VERIFIED"
            atomic_json(OUT / "sync_status.json", state)
            return True
        return False
    finally:
        client.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            if once():
                break
        except Exception as exc:
            with (OUT / "sync_errors.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{datetime.now(timezone.utc).isoformat()} {type(exc).__name__}: {exc}\n")
        time.sleep(60)


if __name__ == "__main__":
    main()
