"""Named random streams, provenance, atomic IO and path confinement."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def seed_for(seed: int, *parts) -> int:
    value = json.dumps([seed, *parts], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def seed_all(seed: int, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA for exact continuation")
        torch.cuda.set_rng_state_all(state["cuda"])


def project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path must be a child of project root: {relative}")
    return path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(root: Path):
    # Hash declared inputs, not editable-install metadata or bytecode caches.
    files = [p for folder, suffixes in (("src/vpr_sam", {".py"}),
                                      ("configs", {".json"}), ("scripts", {".py", ".sh"}))
             for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts]
    files += [root / name for name in ("pyproject.toml", "requirements-research.txt")
              if (root / name).is_file()]
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p) for p in sorted(files)}


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "torchvision", "numpy", "matplotlib", "Pillow")},
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capabilities": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def finite(tensor, label: str):
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Non-finite {label}")
