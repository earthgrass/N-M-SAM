"""Remote read-only data/GPU/source preflight for Dense-only queue."""
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

BASE = Path("/root/autodl-tmp/apbr_l2_dense_seed1to4_20261103")
ROOT = BASE / "project/vpr_sam_research"
PROJECT = ROOT / "experiments/post_training_nm_benchmark"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "src"))

from baseline_adapters.cifar100_model import build_cifar100_model
from scripts.run_cifar100_apbr_fixed_l2_seed1to4_training import seed_all, tensor_sha
from vpr_sam.optim import Stepper


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bad = []
    for row in (ROOT / "SOURCE_SHA256SUMS.txt").read_text().splitlines():
        expected, rel = row.split("  ", 1)
        if sha(ROOT / rel) != expected:
            bad.append(rel)
    if bad:
        raise ValueError(f"Source bundle hashes mismatch: {bad}")
    archive = BASE / "data/cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437
            or sha(archive) != "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
            or hashlib.md5(archive.read_bytes()).hexdigest() != "eb9058c3a382ffc7106e4002c42a8d85"):
        raise ValueError("Official CIFAR-100 archive mismatch")
    with (BASE / "data/extracted/cifar-100-python/train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"])
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if (data.shape != (50000, 3072) or data.dtype != np.uint8
            or labels.shape != (50000,)
            or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 500))):
        raise ValueError("Official Train identity mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU required")
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    if uuid != "GPU-1b337844-dce7-1dd0-b5c0-4f4436da910a":
        raise ValueError("GPU UUID mismatch")
    x = torch.randn(64, 64, device="cuda")
    if not bool(torch.isfinite(x @ x).all()):
        raise FloatingPointError("CUDA matmul smoke failed")
    configs = [PROJECT / "configs/cifar100_50k10k_final_v1_sam_seed245_rtx5090_20261103.yaml",
               PROJECT / "configs/cifar100_50k10k_apbr_fixed_l2_seed1to4_rtx5090_20261103.yaml"]
    for path in configs:
        config = yaml.safe_load(path.read_text())
        if (config["dataset"]["train_size"] != 50000
                or config["dataset"]["validation_size"] != 0
                or config["training"]["dense_epochs"] != 200
                or config["gpu_uuid"] != uuid):
            raise ValueError(f"Config mismatch: {path}")
    seed_all(1)
    model = build_cifar100_model().to("cuda")
    init_sha = tensor_sha(model.state_dict())
    reuse = json.loads((PROJECT / "configs/seed1_sam_reused_identity_20261103.json").read_text())
    if init_sha != reuse["initial_model_sha256"]:
        raise ValueError("Seed1 SAM/APBR-L2 initialization mismatch")
    _ = Stepper(model, {"seed": 1, "optimizer": {"method": "fixed_l2", "lr": .1,
            "momentum": .9, "weight_decay": .0005, "nesterov": False, "rho": .05},
            "geometry": yaml.safe_load(configs[1].read_text())["apbr_sam"]["geometry"]})
    torch.cuda.synchronize()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    evidence = {"status": "PREFLIGHT_PASS", "source_files_verified": 34,
                "archive_sha256": sha(archive), "train_pickle_sha256": sha(BASE / "data/extracted/cifar-100-python/train"),
                "train_count": 50000, "class_count": 100, "per_class": 500,
                "seed1_initial_model_sha256": init_sha,
                "gpu": gpu, "torch": torch.__version__, "torchvision": torchvision.__version__,
                "torch_cuda": torch.version.cuda, "cuda_available": True,
                "test_accessed": False}
    path = BASE / "incoming/preflight.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
