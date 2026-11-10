"""Capture read-only repository and accelerator evidence for the gate audit."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def command(args: list[str], cwd: Path | None = None) -> dict[str, object]:
    try:
        run = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        return {
            "command": args,
            "returncode": run.returncode,
            "stdout": run.stdout.strip(),
            "stderr": run.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root_result = command(["git", "rev-parse", "--show-toplevel"], args.repo)
    git_root = Path(str(root_result.get("stdout", args.repo))).resolve()
    cuda_devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            cuda_devices.append({
                "index": index, "name": props.name,
                "properties_total_memory_bytes": props.total_memory,
                "free_memory_bytes": free, "total_memory_bytes": total,
                "compute_capability": [props.major, props.minor],
            })
    result = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {
            "requested_path": str(args.repo.resolve()),
            "root": root_result,
            "branch": command(["git", "branch", "--show-current"], git_root),
            "head": command(["git", "rev-parse", "HEAD"], git_root),
            "status_porcelain_v1_untracked_all": command(["git", "status", "--porcelain=v1", "--untracked-files=all"], git_root),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": cuda_devices,
            "nvidia_smi": command([
                "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "git_root": str(git_root), "head": result["repository"]["head"].get("stdout"),
        "cuda_available": torch.cuda.is_available(), "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
