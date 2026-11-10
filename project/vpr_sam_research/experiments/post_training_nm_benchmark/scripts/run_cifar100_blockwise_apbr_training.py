"""Independent CIFAR-100 training queue worker for two blockwise APBR ideas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cifar100_50k_formal_training as formal
from baseline_adapters.cifar100_model import CIFAR100_MODEL_ID, build_cifar100_model
from benchmark.nm import eligible_layers

EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028_v2"
PROTOCOL = formal.PROTOCOL_ID


def source_hashes(config: Path, vpr_src: Path) -> dict[str, str]:
    paths = [Path(__file__), config, ROOT / "blockwise_apbr/geometry.py", ROOT / "blockwise_apbr/step.py",
             ROOT / "baseline_adapters/cifar100_model.py", ROOT / "benchmark/model.py", ROOT / "benchmark/nm.py",
             vpr_src / "vpr_sam/optim/step.py", vpr_src / "vpr_sam/optim/geometry.py", vpr_src / "vpr_sam/repro.py"]
    return {str(path.resolve()): formal.digest_file(path) for path in paths}


def save_history(output: Path, history: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader(); writer.writerows(history)
    formal.atomic_text(output / "epoch_metrics.csv", stream.getvalue())
    formal.atomic_text(output / "epoch_metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in history))


def identity_compatible(saved: dict, current: dict) -> bool:
    comparable = dict(saved)
    for key in ("gpu", "python", "torch", "torch_cuda_build", "started_at_unix"):
        comparable[key] = current[key]
    return comparable == current


def aggregate(rows: list[dict], key: str, mode: str = "max"):
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def run_smoke(model, stepper, dataset, seed: int, output: Path, scheme: str) -> None:
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs, targets, sample_ids = next(iter(formal.training_loader(dataset, 128, seed, 0, workers=0)))
    inputs, targets = inputs.cuda(), targets.cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = stepper.step(inputs, targets)
    torch.cuda.synchronize()
    if stepper.active or stepper.failed or stepper.steps != 1:
        raise RuntimeError("Smoke Stepper transaction did not complete cleanly")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Smoke produced non-finite parameters")
    evidence = {
        "status": "PASS", "experiment_id": EXPERIMENT, "scheme": scheme, "seed": seed,
        "sample_ids_sha256": hashlib.sha256(sample_ids.numpy().astype("<i8").tobytes()).hexdigest(),
        "step_result": result, "parameter_state_changed_by_sgd": any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()),
        "parameter_location": "unperturbed_after_sgd", "test_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if scheme == "global_block_feedback" and result.get("perturbation_matches_original") is not True:
        raise RuntimeError("Scheme 4 did not report exact original perturbation")
    if scheme == "equal_block_energy" and result.get("active_block_norm_relative_spread", 1.0) > 2e-6:
        raise RuntimeError("Scheme 1 block norms are not equal")
    formal.atomic_json(output / "smoke_evidence.json", evidence)
    print(json.dumps(evidence, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vpr-src", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme", choices=("global_block_feedback", "equal_block_energy"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT or config.get("status") != "FROZEN":
        raise ValueError("Wrong or unfrozen blockwise experiment config")
    if args.scheme not in config["schemes"] or args.seed not in config["seeds"]:
        raise ValueError("Scheme/seed outside frozen matrix")
    if config["training"]["epochs"] != 200 or config["evaluation"]["recovery_epochs"] != 0:
        raise ValueError("Frozen budget/no-Recovery contract failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device is required; CPU fallback forbidden")
    if str(args.vpr_src) not in sys.path:
        sys.path.insert(0, str(args.vpr_src))
    from blockwise_apbr import BlockwiseStepper

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite nonempty output")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("Resume requires last.pt")

    gpu = formal.gpu_identity(config["gpu_uuid"])
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169_001_437
            or formal.digest_file(archive, "md5") != config["dataset"]["archive_md5"]
            or formal.digest_file(archive) != config["dataset"]["archive_sha256"]):
        raise ValueError("Official CIFAR-100 archive identity mismatch")
    train_file = args.dataset_root / "extracted/cifar-100-python/train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data, labels = np.asarray(raw[b"data"], dtype=np.uint8), np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,) or not bool(np.all(np.bincount(labels, minlength=100) == 500)):
        raise ValueError("Official CIFAR-100 Train contract failed")

    formal.seed_all(args.seed)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    dataset = formal.OfficialTrain(data, labels, transform)
    model = build_cifar100_model().cuda()
    eligible_names = {name + ".weight" for name in eligible_layers(model)}
    if len(eligible_names) != 19:
        raise ValueError("Expected 19 eligible Conv2d parameters")
    method_config = {
        "seed": args.seed,
        "optimizer": {"method": "vpr", "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": False, "rho": config["apbr"]["rho"]},
        "geometry": {"gamma": config["apbr"]["gamma"], "dtype": config["apbr"]["geometry_dtype"],
                     "warmup_steps": config["apbr"]["warmup_steps"], "volume_atol": config["apbr"]["volume_atol"],
                     "volume_rtol": config["apbr"]["volume_rtol"], "static_std": 0.1,
                     "block_degenerate_threshold": config["blockwise"]["degenerate_threshold"]},
    }
    stepper = BlockwiseStepper(model, method_config, args.scheme, eligible_names)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base, T_max=200, eta_min=0.0)
    if args.smoke:
        run_smoke(model, stepper, dataset, args.seed, output, args.scheme)
        return 0

    frozen_sources = source_hashes(config_path, args.vpr_src)
    identity = {
        "protocol_id": PROTOCOL, "experiment_id": EXPERIMENT, "method": args.scheme, "seed": args.seed,
        "model_id": CIFAR100_MODEL_ID, "architecture_sha256": formal.ARCHITECTURE_SHA256,
        "random_initialization": True, "external_pretraining": False,
        "initial_model_sha256": formal.tensor_sha(model.state_dict()), "config_sha256": formal.digest_file(config_path),
        "source_sha256": frozen_sources, "archive_sha256": formal.digest_file(archive),
        "train_pickle_sha256": formal.digest_file(train_file), "train_size": 50_000, "validation_size": 0,
        "test_loader_constructed": False, "test_accessed": False, "gpu": gpu,
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "started_at_unix": time.time(),
    }
    start_epoch, history = 0, []
    if args.resume:
        saved_identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        formal.validate_checkpoint(checkpoint, saved_identity)
        if not identity_compatible(saved_identity, identity):
            raise ValueError("Resume identity mismatch")
        identity = saved_identity
        model.load_state_dict(checkpoint["model"]); stepper.load_state_dict(checkpoint["stepper"])
        scheduler.load_state_dict(checkpoint["scheduler"]); formal.restore_rng(checkpoint["rng"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
    else:
        formal.atomic_json(output / "identity.json", identity)
        formal.atomic_json(output / "resolved_config.json", {"experiment": config, "method_config": method_config,
                           "eligible_names": sorted(eligible_names), "test_accessed": False})

    torch.cuda.reset_peak_memory_stats()
    run_started = time.monotonic()
    final_payload = None
    for epoch_index in range(start_epoch, 200):
        if source_hashes(config_path, args.vpr_src) != frozen_sources:
            raise RuntimeError("Frozen source/config changed during training")
        epoch_started = time.monotonic(); step_rows = []
        loss_sum = adv_sum = 0.0; correct = count = batches = 0; order = hashlib.sha256()
        for inputs, targets, sample_ids in formal.training_loader(dataset, 128, args.seed, epoch_index):
            result = stepper.step(inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True))
            step_rows.append(result); loss_sum += result["center_loss"] * result["count"]
            adv_sum += result["adversarial_loss"] * result["count"]; correct += result["correct"]
            count += result["count"]; batches += 1; order.update(sample_ids.numpy().astype("<i8").tobytes())
        if count != 50_000 or batches != 391:
            raise ValueError("Full Train coverage failed")
        used_lr = stepper.base.param_groups[0]["lr"]; scheduler.step(); torch.cuda.synchronize()
        diagnostics = stepper.geometry.diagnostics()
        row = {
            "epoch": epoch_index + 1, "steps": stepper.steps, "batches": batches, "train_samples": count,
            "train_order_sha256": order.hexdigest(), "learning_rate_used": used_lr,
            "learning_rate_next": stepper.base.param_groups[0]["lr"], "train_loss": loss_sum / count,
            "adversarial_train_loss": adv_sum / count, "train_accuracy": correct / count,
            "original_eps_l2_min": aggregate(step_rows, "original_eps_l2", "min"),
            "original_eps_l2_max": aggregate(step_rows, "original_eps_l2"),
            "applied_eps_l2_min": aggregate(step_rows, "applied_eps_l2", "min"),
            "applied_eps_l2_max": aggregate(step_rows, "applied_eps_l2"),
            "block_norm_spread_max": aggregate(step_rows, "block_norm_relative_spread"),
            "active_block_norm_spread_max": aggregate(step_rows, "active_block_norm_relative_spread"),
            "block_scale_min": aggregate(step_rows, "block_scale_min", "min"),
            "block_scale_max": aggregate(step_rows, "block_scale_max"),
            "active_blocks_min": aggregate(step_rows, "active_blocks", "min"),
            "degenerate_blocks_max": aggregate(step_rows, "degenerate_blocks"),
            "block_delta_sum_error_max": aggregate(step_rows, "block_delta_sum_max"),
            "residual_delta_sum_abs_max": max(abs(row.get("residual_delta_sum", 0.0)) for row in step_rows),
            "geometry_updates": diagnostics["geometry_updates"], "log_volume": diagnostics["log_volume"],
            "log_radius_min": diagnostics["log_radius_min"], "log_radius_max": diagnostics["log_radius_max"],
            "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - run_started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "test_accessed": False,
        }
        history.append(row)
        final_payload = formal.payload(model, stepper, scheduler, epoch_index + 1, history, identity)
        final_payload["experiment_id"] = EXPERIMENT; final_payload["scheme"] = args.scheme
        last_sha = formal.atomic_checkpoint(output / "last.pt", final_payload)
        save_history(output, history)
        formal.atomic_json(output / "checkpoint_publication.json", {"epoch": epoch_index + 1, "rolling_checkpoint": "last.pt", "sha256": last_sha, "selection_used": False, "test_accessed": False})
        formal.atomic_json(output / "status.json", {"status": "training", "experiment_id": EXPERIMENT, "scheme": args.scheme, "seed": args.seed, "completed_epochs": epoch_index + 1, "steps": stepper.steps, "test_accessed": False})
        print(json.dumps(row, allow_nan=False), flush=True)
    if final_payload is None or final_payload["epoch"] != 200 or final_payload["steps"] != 78_200:
        raise RuntimeError("Final fixed training budget mismatch")
    final_sha = formal.atomic_checkpoint(output / "final_epoch_200.pt", final_payload)
    formal.atomic_json(output / "status.json", {"status": "completed", "experiment_id": EXPERIMENT, "scheme": args.scheme, "seed": args.seed, "completed_epochs": 200, "steps": 78_200, "primary_checkpoint": "final_epoch_200.pt", "final_checkpoint_sha256": final_sha, "test_accessed": False, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output-dir") + 1]); directory.mkdir(parents=True, exist_ok=True)
            formal.atomic_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "automatic_retry": False})
            formal.atomic_json(directory / "status.json", {"status": "failed", "message": str(error)})
        raise
