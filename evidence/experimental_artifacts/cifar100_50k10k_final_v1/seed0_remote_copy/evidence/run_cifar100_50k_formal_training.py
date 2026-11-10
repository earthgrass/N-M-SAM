"""Formal CIFAR-100 50k training with fixed epoch-200 selection.

This module opens only the official training pickle.  It has no validation or
test dataset construction, metric, selection, or early-stopping path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import CIFAR100_MODEL_ID, build_cifar100_model

PROTOCOL_ID = "cifar100_50k10k_final_v1"
ARCHITECTURE_SHA256 = "95367b739c42cca14cb51d1a776e5dbdbe71f9ed4a1afbd2c5c0b1bb994827ea"


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(values) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def validate_checkpoint(value, identity=None) -> None:
    required = {"schema", "status", "protocol_id", "epoch", "model", "model_sha256",
                "stepper", "scheduler", "history", "identity", "rng"}
    if not required.issubset(value) or value["schema"] != 1 or value["status"] != "healthy":
        raise ValueError("Checkpoint schema or health mismatch")
    if value["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Checkpoint belongs to another data/selection protocol")
    if identity is not None and value["identity"] != identity:
        raise ValueError("Checkpoint identity mismatch")
    if tensor_sha(value["model"]) != value["model_sha256"]:
        raise ValueError("Checkpoint tensor SHA256 mismatch")
    if not all(bool(torch.isfinite(item).all()) for item in value["model"].values()):
        raise FloatingPointError("Non-finite checkpoint model")


def atomic_checkpoint(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    validate_checkpoint(torch.load(temporary, map_location="cpu", weights_only=False), value["identity"])
    temporary.replace(path)
    validate_checkpoint(torch.load(path, map_location="cpu", weights_only=False), value["identity"])
    return digest_file(path)


def seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


class OfficialTrain(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform) -> None:
        self.data, self.labels, self.transform = data, labels, transform

    def __len__(self) -> int:
        return 50_000

    def __getitem__(self, sample_id: int):
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def worker_seed(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def training_loader(dataset: Dataset, batch_size: int, seed: int, epoch: int, workers: int = 2) -> DataLoader:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch * 97 + 1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=workers,
                      pin_memory=True, persistent_workers=False, worker_init_fn=worker_seed,
                      generator=generator)


def source_hashes(config_path: Path, vpr_src: Path):
    paths = [Path(__file__), config_path, ROOT / "baseline_adapters" / "cifar100_model.py",
             ROOT / "benchmark" / "model.py", vpr_src / "vpr_sam" / "optim" / "step.py",
             vpr_src / "vpr_sam" / "optim" / "geometry.py", vpr_src / "vpr_sam" / "repro.py"]
    return {str(path.resolve()): digest_file(path) for path in paths}


def gpu_identity(expected_uuid: str):
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = str(properties.uuid)
    if not torch_uuid.startswith("GPU-"):
        torch_uuid = "GPU-" + torch_uuid
    fields = [item.strip() for item in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
         "--format=csv,noheader,nounits"], text=True).strip().split(",")]
    if len(fields) != 5 or fields[1] != expected_uuid or torch_uuid != expected_uuid:
        raise RuntimeError("Frozen GPU identity mismatch")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        raise RuntimeError(f"GPU already has a compute process: {processes}")
    return {"name": fields[0], "uuid": fields[1], "memory_total_mib": int(fields[2]),
            "memory_free_mib": int(fields[3]), "driver": fields[4],
            "compute_capability": [properties.major, properties.minor]}


def save_history(output: Path, history) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    atomic_text(output / "epoch_metrics.csv", stream.getvalue())
    atomic_text(output / "epoch_metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in history))


def payload(model, stepper, scheduler, epoch, history, identity):
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {"schema": 1, "status": "healthy", "protocol_id": PROTOCOL_ID,
            "selection_mode": "fixed_final_epoch", "primary_checkpoint": "final_epoch_200.pt",
            "epoch": epoch, "steps": stepper.steps, "model": model_state,
            "model_sha256": tensor_sha(model_state), "stepper": stepper.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history, "identity": identity,
            "rng": rng_state(), "test_accessed": False,
            "parameter_location": "unperturbed_after_sgd"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("sam", "vpr"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL_ID or config["dataset"]["train_size"] != 50_000:
        raise ValueError("Wrong formal protocol")
    if config["dataset"]["validation_size"] != 0 or config["training"]["selection_mode"] != "fixed_final_epoch":
        raise ValueError("Validation selection is forbidden")
    if config["training"]["dense_epochs"] != 200 or config["dataset"]["drop_last"]:
        raise ValueError("Frozen epoch/drop_last contract failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device is required; CPU fallback forbidden")
    if str(args.vpr_src) not in sys.path:
        sys.path.insert(0, str(args.vpr_src))
    from vpr_sam.optim import Stepper

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty formal output")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires this protocol's last.pt")

    gpu = gpu_identity(config["gpu_uuid"])
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169_001_437 or digest_file(archive, "md5") != config["dataset"]["archive_md5"] or digest_file(archive) != config["dataset"]["archive_sha256"]:
        raise ValueError("Official archive identity mismatch")
    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,) or set(labels.tolist()) != set(range(100)):
        raise ValueError("Official train data contract failed")
    counts = np.bincount(labels, minlength=100)
    if not bool(np.all(counts == 500)):
        raise ValueError("Official train class distribution mismatch")

    seed_all(args.seed)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    dataset = OfficialTrain(data, labels, transform)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    method_config = {
        "seed": args.seed,
        "optimizer": {"method": args.method, "lr": 0.1, "momentum": 0.9,
                      "weight_decay": 0.0005, "nesterov": False, "rho": 0.05},
        "geometry": dict(config["apbr_sam"]["geometry"]),
    }
    stepper = Stepper(model, method_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base, T_max=200, eta_min=0.0)
    frozen_sources = source_hashes(config_path, args.vpr_src)
    identity = {
        "protocol_id": PROTOCOL_ID, "method": "dense_sam" if args.method == "sam" else "apbr_sam",
        "seed": args.seed, "model_id": CIFAR100_MODEL_ID, "architecture_sha256": ARCHITECTURE_SHA256,
        "random_initialization": True, "external_pretraining": False,
        "initial_model_sha256": tensor_sha(model.state_dict()), "config_sha256": digest_file(config_path),
        "source_sha256": frozen_sources, "archive_sha256": digest_file(archive),
        "train_pickle_sha256": digest_file(train_file), "train_size": 50_000,
        "validation_size": 0, "test_loader_constructed": False, "test_accessed": False,
        "gpu": gpu, "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "started_at_unix": time.time(),
    }
    start_epoch, history = 0, []
    if args.resume:
        saved_identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, saved_identity)
        if saved_identity != identity:
            # Start time is part of the original identity and is never regenerated on resume.
            comparable = dict(saved_identity)
            comparable["gpu"] = identity["gpu"]
            comparable["python"] = identity["python"]
            comparable["torch"] = identity["torch"]
            comparable["torch_cuda_build"] = identity["torch_cuda_build"]
            comparable["started_at_unix"] = identity["started_at_unix"]
            if comparable != identity:
                raise ValueError("Resume config/data/source/model identity mismatch")
        identity = saved_identity
        model.load_state_dict(checkpoint["model"])
        stepper.load_state_dict(checkpoint["stepper"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
    else:
        atomic_json(output / "identity.json", identity)
        atomic_json(output / "resolved_config.json", {"protocol": config, "method_config": method_config,
                    "selection": "fixed epoch 200; no validation/test selection",
                    "checkpoint_policy": "atomic rolling last.pt after every epoch plus final_epoch_200.pt"})

    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.monotonic()
    final_payload = None
    for epoch_index in range(start_epoch, 200):
        if source_hashes(config_path, args.vpr_src) != frozen_sources:
            raise RuntimeError("Frozen source/config changed during training")
        epoch_started = time.monotonic()
        loss_sum = adversarial_sum = 0.0
        correct = count = batches = 0
        order = hashlib.sha256()
        eps_min, eps_max = float("inf"), 0.0
        for inputs, targets, sample_ids in training_loader(dataset, 128, args.seed, epoch_index):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            result = stepper.step(inputs, targets)
            loss_sum += result["center_loss"] * result["count"]
            adversarial_sum += result["adversarial_loss"] * result["count"]
            correct += result["correct"]
            count += result["count"]
            batches += 1
            order.update(sample_ids.numpy().astype("<i8").tobytes())
            eps_min = min(eps_min, result["eps_norm"])
            eps_max = max(eps_max, result["eps_norm"])
        if count != 50_000 or batches != 391:
            raise ValueError(f"Full-train coverage failed: samples={count}, batches={batches}")
        used_lr = stepper.base.param_groups[0]["lr"]
        scheduler.step()
        torch.cuda.synchronize(device)
        diagnostics = stepper.geometry.diagnostics()
        row = {"epoch": epoch_index + 1, "steps": stepper.steps, "batches": batches,
               "train_samples": count, "unique_dataset_size": 50_000,
               "train_order_sha256": order.hexdigest(), "learning_rate_used": used_lr,
               "learning_rate_next": stepper.base.param_groups[0]["lr"],
               "train_loss": loss_sum / count, "adversarial_train_loss": adversarial_sum / count,
               "train_accuracy": correct / count, "eps_norm_min": eps_min, "eps_norm_max": eps_max,
               "geometry_updates": diagnostics["geometry_updates"], "log_volume": diagnostics["log_volume"],
               "epoch_seconds": time.monotonic() - epoch_started,
               "elapsed_seconds": time.monotonic() - run_started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "test_accessed": False}
        history.append(row)
        final_payload = payload(model, stepper, scheduler, epoch_index + 1, history, identity)
        last_sha = atomic_checkpoint(output / "last.pt", final_payload)
        save_history(output, history)
        atomic_json(output / "checkpoint_publication.json", {"epoch": epoch_index + 1,
                    "rolling_checkpoint": "last.pt", "sha256": last_sha,
                    "selection_used": False, "test_accessed": False})
        atomic_json(output / "status.json", {"status": "training", "protocol_id": PROTOCOL_ID,
                    "completed_epochs": epoch_index + 1, "steps": stepper.steps,
                    "primary_checkpoint": "pending_final_epoch_200.pt", "test_accessed": False})
        print(json.dumps(row, allow_nan=False), flush=True)
    if final_payload is None or final_payload["epoch"] != 200 or final_payload["steps"] != 78_200:
        raise RuntimeError("Final fixed training budget mismatch")
    final_sha = atomic_checkpoint(output / "final_epoch_200.pt", final_payload)
    atomic_json(output / "status.json", {"status": "completed", "protocol_id": PROTOCOL_ID,
                "completed_epochs": 200, "steps": 78_200, "selection_mode": "fixed_final_epoch",
                "primary_checkpoint": "final_epoch_200.pt", "final_checkpoint_sha256": final_sha,
                "test_accessed": False, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            atomic_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(), "automatic_retry": False})
            atomic_json(directory / "status.json", {"status": "failed", "message": str(error)})
        raise
    raise SystemExit(code)
