"""Seed-0 spherical SAM with the frozen groupwise second-order training objective."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]; PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT / "src")]
import run_cifar100_50k_formal_training as base
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.orchestration import AtomicCheckpointManager, FormalSchedule, write_manifest
from joint_optimization.sam_joint_stepper import SAMJointStepper
from run_apbr_gpt_joint_training_seed0 import refresh_cache
from vpr_sam.optim.geometry import Geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SAM-GPT formal runner requires exactly one CUDA device")
    schedule = FormalSchedule(); device = torch.device("cuda:0")
    protocol = base.yaml.safe_load(args.protocol_config.read_text(encoding="utf-8"))
    joint = json.loads(args.joint_config.read_text(encoding="utf-8"))
    lambda_record = json.loads((args.calibration_dir / "lambda_calibration.json").read_text())
    identity_record = json.loads((args.calibration_dir / "calibration_identity.json").read_text())
    if identity_record["status"] != "REAL_CALIBRATION_PASS" or identity_record["final_test_accessed"]:
        raise RuntimeError("Frozen train-only calibration gate failed")
    initial = torch.load(args.initial_asset, map_location="cpu", weights_only=False)
    if initial.get("cohort") != "apbr_gpt_joint_seed0_rebased": raise ValueError("Wrong initial cohort")
    train_file = args.data_root / "extracted/cifar-100-python/train"
    with train_file.open("rb") as stream: raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8); labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,): raise RuntimeError("Train identity failed")
    base.seed_all(0)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(.5), transforms.ToTensor(), transforms.Normalize(
            protocol["dataset"]["normalize_mean"], protocol["dataset"]["normalize_std"])])
    dataset = base.OfficialTrain(data, labels, transform)
    fixed_ids = np.asarray(identity_record["batch_ids"], dtype=np.int64)
    fixed = torch.from_numpy(data[fixed_ids].copy()).reshape(-1, 3, 32, 32).float().div_(255)
    mean = torch.tensor(protocol["dataset"]["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(protocol["dataset"]["normalize_std"]).view(1, 3, 1, 1)
    fixed = ((fixed - mean) / std).to(device)

    model = build_cifar100_model().to(device); model.load_state_dict(initial["model"])
    geometry = Geometry(model.named_parameters(), protocol["apbr_sam"]["geometry"], "sam", 0)
    optimizer = torch.optim.SGD(geometry.params, lr=.1, momentum=.9, weight_decay=5e-4)
    layer_map = eligible_layers(model); eligible = tuple(name + ".weight" for name in layer_map)
    stepper = SAMJointStepper(model, geometry, optimizer, eligible_parameter_names=eligible,
        lambda_p=lambda_record["lambda_p"], temperature=joint["joint_objective"]["temperature"],
        rho=joint["formal_training"]["rho"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("Refusing nonempty formal output")
    manager = AtomicCheckpointManager(output)
    references = {name + ".weight": float(value) for name, value in identity_record["references"].items()}
    manifest = {"status": "training_prepared", "experiment_id": "sam_gpt_joint_seed0_rebased",
        "protocol_id": "cifar100_50k10k_final_v1", "seed": 0,
        "schedule": {"epochs": 200, "warmup_epochs": 5, "hessian_refresh_epochs": 10},
        "initial_checkpoint_sha256": base.digest_file(args.initial_asset),
        "initial_model_sha256": initial["model_tensor_sha256"], "lambda_p": stepper.lambda_p,
        "eta_s": None, "perturbation": "standard_global_l2_sam",
        "pruning_probe": "detached_standard_sam_epsilon", "test_loader_constructed": False,
        "test_accessed": False}
    write_manifest(output / "manifest.json", manifest)
    history, refresh_rows, cache_name, cache_sha = [], [], None, None
    torch.cuda.reset_peak_memory_stats(device); run_started = time.monotonic(); payload = None
    for epoch in range(200):
        active = schedule.pruning_active(epoch)
        if schedule.refresh_due(epoch) or (active and not stepper.factors):
            cache, ref, rows = refresh_cache(model, layer_map, fixed, epoch, references)
            stepper.refresh(cache, ref); refresh_rows.extend(rows)
            cache_name = f"hessian_cache_epoch_{epoch + 1:03d}.pt"
            _path, cache_sha = manager.save(cache_name, {"factors": cache, "references": ref,
                "refresh_epoch": epoch + 1, "test_accessed": False})
        started = time.monotonic(); count = batches = correct = 0
        joint_sum = regularizer_sum = center_sum = adversarial_sum = 0.0
        for inputs, targets, _ids in base.training_loader(dataset, 128, 0, epoch):
            result = stepper.step(inputs.to(device), targets.to(device), pruning_active=active)
            count += result["count"]; batches += 1; correct += result["correct"]
            center_sum += result["center_loss"] * result["count"]
            adversarial_sum += result["adversarial_loss"] * result["count"]
            joint_sum += result["joint_loss"] * result["count"]
            regularizer_sum += result["pruning_regularizer"] * result["count"]
        if count != 50_000 or batches != 391: raise RuntimeError("Full train coverage failed")
        scheduler.step(); torch.cuda.synchronize(device)
        row = {"epoch": epoch + 1, "steps": stepper.steps, "samples": count, "batches": batches,
            "train_accuracy": correct / count, "center_loss": center_sum / count,
            "adversarial_loss": adversarial_sum / count, "joint_loss": joint_sum / count,
            "pruning_regularizer": regularizer_sum / count, "pruning_active": active,
            "epoch_seconds": time.monotonic() - started, "elapsed_seconds": time.monotonic() - run_started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "test_accessed": False}
        history.append(row)
        payload = {"schema": 1, "status": "healthy", "protocol_id": "cifar100_50k10k_final_v1",
            "test_accessed": False, "identity": {"method": "sam_gpt_joint", "seed": 0,
                "model_id": protocol["model"]["id"], "initial_model_sha256": initial["model_tensor_sha256"],
                "initial_checkpoint_sha256": base.digest_file(args.initial_asset)},
            "model": model.state_dict(), "stepper": stepper.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": base.rng_state(), "epoch": epoch + 1, "history": history,
            "refresh_rows": refresh_rows, "manifest": manifest,
            "cache_name": cache_name, "cache_sha256": cache_sha}
        manager.save("last.pt", payload)
        write_manifest(output / "status.json", {"status": "training", "protocol_id": "cifar100_50k10k_final_v1",
            "completed_epochs": epoch + 1, "steps": stepper.steps, "pruning_active": active,
            "test_accessed": False})
    if payload is None or stepper.steps != 78_200: raise RuntimeError("Formal budget mismatch")
    final_path, final_sha = manager.save("final_epoch_200.pt", payload)
    write_manifest(output / "checkpoint_freeze.json", {"status": "CHECKPOINTS_FROZEN",
        "checkpoint_sha256": {final_path.name: final_sha}, "epoch": 200, "steps": 78_200,
        "test_accessed": False})
    write_manifest(output / "status.json", {"status": "completed", "protocol_id": "cifar100_50k10k_final_v1",
        "completed_epochs": 200, "steps": 78_200, "final_checkpoint_sha256": final_sha,
        "test_accessed": False})
    with (output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    return 0


if __name__ == "__main__": raise SystemExit(main())
