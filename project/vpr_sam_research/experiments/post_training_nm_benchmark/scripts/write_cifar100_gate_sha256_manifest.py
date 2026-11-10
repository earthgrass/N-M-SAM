"""Write a reproducible SHA256 inventory for the CIFAR-100 gate deliverables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite manifest: {output}")
    deliverable = root / "docs" / "research" / "cifar100_gpu_gate_20261026"
    files = [path for path in deliverable.rglob("*") if path.is_file() and path.resolve() != output]
    code = [
        root / "baseline_adapters" / "cifar100_model.py",
        root / "baseline_adapters" / "sparsegpt" / "adapter.py",
        root / "scripts" / "audit_cifar100_identity.py",
        root / "scripts" / "audit_cifar100_model_identity.py",
        root / "scripts" / "capture_cifar100_gate_environment.py",
        root / "scripts" / "run_sparsegpt_single_layer_gpu_gate.py",
        root / "tests" / "test_cifar100_identity_gpu_gate.py",
    ]
    inventory = []
    for path in sorted(set(files + code)):
        inventory.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"algorithm": "SHA256", "files": inventory}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "file_count": len(inventory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
