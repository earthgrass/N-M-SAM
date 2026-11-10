"""Isolated Seed 0/1 APBR-Joint runner with frozen full or epoch-10 guidance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT / "src")]

import run_cifar100_50k_formal_training as base
from baseline_adapters.apbr_gpt_joint import candidate_costs
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.hessian_cache import build_training_factors, capture_layer_inputs
from joint_optimization.joint_stepper import JointStepper
from joint_optimization.orchestration import AtomicCheckpointManager, FormalSchedule, formal_manifest, write_manifest
from vpr_sam.optim.geometry import Geometry


def guidance_due(epoch_zero_based: int, mode: str) -> bool:
    if not 0 <= epoch_zero_based < 200:
        raise ValueError("Epoch outside frozen budget")
    if mode == "full":
        return epoch_zero_based >= 5
    if mode == "epoch10":
        return (epoch_zero_based + 1) % 10 == 0
    raise ValueError("Unknown guidance mode")


def refresh_cache(model, layers, fixed_inputs, epoch, references):
    device = next(model.parameters()).device
    captured = capture_layer_inputs(model, layers, fixed_inputs, storage_device=device)
    factors, rows = {}, []
    for index, (name, layer) in enumerate(layers.items()):
        seed_bytes = f"apbr-gpt-hessian:{epoch}:{index}:{name}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "little") % (2**63 - 1)
        item = build_training_factors(layer, captured.pop(name), sample_seed=seed,
            max_patches=8192, percdamp=0.01, compute_dtype=torch.float32)
        factors[name + ".weight"] = item.factors
        rows.append({"epoch": epoch + 1, "layer": name, "sampled_patches": item.sampled_patches,
            "patch_ids_sha256": item.sampled_patch_ids_sha256, "damp": item.factors.damp,
            "condition_max": item.factors.condition_max, "gram_seconds": item.gram_seconds,
            "factor_seconds": item.factor_seconds})
    return factors, {name: references[name] for name in factors}, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--joint-config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--guidance-mode", choices=("full", "epoch10"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Formal runner requires exactly one available CUDA device")
    schedule = FormalSchedule(); device = torch.device("cuda:0")
    protocol = base.yaml.safe_load(args.protocol_config.read_text(encoding="utf-8"))
    joint = json.loads(args.joint_config.read_text(encoding="utf-8"))
    lambda_record = json.loads((args.calibration_dir / "lambda_calibration.json").read_text())
    eta_record = json.loads((args.calibration_dir / "eta_s_calibration.json").read_text())
    identity_record = json.loads((args.calibration_dir / "calibration_identity.json").read_text())
    if identity_record["status"] != "REAL_CALIBRATION_PASS" or identity_record["final_test_accessed"]:
        raise RuntimeError("Real train-only calibration gate has not passed")
    initial = torch.load(args.initial_asset, map_location="cpu", weights_only=False)
    expected_cohort = "apbr_gpt_joint_seed0_rebased" if args.seed == 0 else "apbr_gpt_joint_seed1_v2"
    if initial.get("cohort") != expected_cohort or initial.get("seed") != args.seed:
        raise ValueError("Wrong initial cohort or seed")

    train_file = args.data_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8); labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,):
        raise RuntimeError("Official train identity failed")
    base.seed_all(args.seed)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
        transforms.Normalize(protocol["dataset"]["normalize_mean"], protocol["dataset"]["normalize_std"])])
    dataset = base.OfficialTrain(data, labels, transform)
    fixed_ids = np.asarray(identity_record["batch_ids"], dtype=np.int64)
    fixed = torch.from_numpy(data[fixed_ids].copy()).reshape(-1, 3, 32, 32).float().div_(255)
    mean = torch.tensor(protocol["dataset"]["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(protocol["dataset"]["normalize_std"]).view(1, 3, 1, 1)
    fixed = ((fixed - mean) / std).to(device)

    model = build_cifar100_model().to(device); model.load_state_dict(initial["model"])
    geometry = Geometry(model.named_parameters(), protocol["apbr_sam"]["geometry"], "vpr", args.seed)
    geometry.load_state_dict(initial["apbr_geometry"])
    optimizer = torch.optim.SGD(geometry.params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    layer_map = eligible_layers(model); eligible = tuple(name + ".weight" for name in layer_map)
    stepper = JointStepper(model, geometry, optimizer, eligible_parameter_names=eligible,
        lambda_p=lambda_record["lambda_p"], eta_s=eta_record["eta_s"],
        temperature=joint["joint_objective"]["temperature"], rho=joint["formal_training"]["rho"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    output = args.output_dir.resolve(); manager = AtomicCheckpointManager(output)
    history, refresh_rows, start_epoch = [], [], 0
    cache_name = cache_sha = None
    manifest = formal_manifest(); manifest.update({"status": "training_prepared", "seed": args.seed,
        "experiment_id": f"apbr_joint_seed{args.seed}_{args.guidance_mode}_20261101_v2",
        "guidance_mode": args.guidance_mode,
        "guidance_epochs_one_based": list(range(10, 201, 10)) if args.guidance_mode == "epoch10" else list(range(6, 201)),
        "reference_source": "seed0_train_side_calibration_frozen",
        "initial_checkpoint_sha256": base.digest_file(args.initial_asset),
        "lambda_p": stepper.lambda_p, "eta_s": stepper.eta_s,
        "train_pickle_sha256": base.digest_file(train_file), "test_loader_constructed": False})
    if args.resume:
        saved = manager.load("last.pt")
        model.load_state_dict(saved["model"]); stepper.load_state_dict(saved["stepper"])
        scheduler.load_state_dict(saved["scheduler"]); base.restore_rng(saved["rng"])
        history, refresh_rows, start_epoch = saved["history"], saved["refresh_rows"], saved["epoch"]
        cache_name, cache_sha = saved.get("cache_name"), saved.get("cache_sha256")
        if cache_name:
            if not cache_sha:
                raise RuntimeError("Resume checkpoint lacks frozen Hessian cache identity")
            cached = manager.load(cache_name, expected_sha256=cache_sha)
            stepper.refresh(cached["factors"], cached["references"])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing to overwrite formal output")
        output.mkdir(parents=True, exist_ok=True); write_manifest(output / "manifest.json", manifest)

    references = {name: float(value) for name, value in identity_record["references"].items()}
    references = {name + ".weight": value for name, value in references.items()}
    torch.cuda.reset_peak_memory_stats(device); run_started = time.monotonic()
    for epoch in range(start_epoch, schedule.epochs):
        active = guidance_due(epoch, args.guidance_mode)
        due = schedule.refresh_due(epoch) if args.guidance_mode == "full" else active
        previous_cache = cache_name
        if due or (active and not stepper.factors):
            refresh_epoch = epoch if due else (
                schedule.warmup_epochs + (epoch - schedule.warmup_epochs) // schedule.hessian_refresh_epochs
                * schedule.hessian_refresh_epochs
            )
            cache, ref, rows = refresh_cache(model, layer_map, fixed, refresh_epoch, references)
            stepper.refresh(cache, ref); refresh_rows.extend(rows)
            cache_name = f"hessian_cache_epoch_{refresh_epoch + 1:03d}.pt"
            _cache_path, cache_sha = manager.save(cache_name, {"factors": cache, "references": ref,
                "refresh_epoch": refresh_epoch + 1, "test_accessed": False})
        if active and not stepper.factors:
            raise RuntimeError("Active epoch has no Hessian cache")
        started = time.monotonic(); count = batches = 0; loss_sum = regularizer_sum = 0.0
        for inputs, targets, _ids in base.training_loader(dataset, 128, args.seed, epoch):
            result = stepper.step(inputs.to(device), targets.to(device), pruning_active=active)
            count += result["count"]; batches += 1
            loss_sum += result["joint_loss"] * result["count"]
            regularizer_sum += result["pruning_regularizer"] * result["count"]
        if count != 50_000 or batches != 391:
            raise RuntimeError("Full train coverage failed")
        scheduler.step(); torch.cuda.synchronize(device)
        history.append({"epoch": epoch + 1, "steps": stepper.steps, "samples": count,
            "batches": batches, "pruning_active": active, "joint_loss": loss_sum / count,
            "pruning_regularizer": regularizer_sum / count, "epoch_seconds": time.monotonic() - started,
            "elapsed_seconds": time.monotonic() - run_started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "test_accessed": False})
        payload = {"schema": 1, "status": "healthy",
            "protocol_id": "cifar100_50k10k_final_v1", "test_accessed": False,
            "identity": {"method": f"joint_apbr_gpt_{args.guidance_mode}", "seed": args.seed,
                "model_id": protocol["model"]["id"],
                "initial_model_sha256": initial["model_tensor_sha256"],
                "initial_checkpoint_sha256": base.digest_file(args.initial_asset)},
            "model": model.state_dict(), "stepper": stepper.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": base.rng_state(), "epoch": epoch + 1,
            "history": history, "refresh_rows": refresh_rows, "manifest": manifest,
            "cache_name": cache_name, "cache_sha256": cache_sha}
        manager.save("last.pt", payload)
        if previous_cache and previous_cache != cache_name:
            (output / previous_cache).unlink()
        write_manifest(output / "status.json", {"status": "training",
            "protocol_id": "cifar100_50k10k_final_v1", "completed_epochs": epoch + 1,
            "steps": stepper.steps, "pruning_active": active,
            "test_accessed": False})
    if stepper.steps != 78_200:
        raise RuntimeError("Formal step budget mismatch")
    final_path, final_sha = manager.save("final_epoch_200.pt", payload)
    write_manifest(output / "checkpoint_freeze.json", {"status": "CHECKPOINTS_FROZEN",
        "checkpoint_sha256": {final_path.name: final_sha}, "epoch": 200, "steps": 78_200,
        "test_accessed": False})
    write_manifest(output / "status.json", {"status": "completed",
        "protocol_id": "cifar100_50k10k_final_v1", "completed_epochs": 200,
        "steps": 78_200, "final_checkpoint_sha256": final_sha,
        "test_accessed": False})
    with (output / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    with (output / "hessian_refresh.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=refresh_rows[0]); writer.writeheader(); writer.writerows(refresh_rows)
    (output / "last.pt").unlink()
    if cache_name:
        (output / cache_name).unlink()
    write_manifest(output / "rolling_cache_retention.json", {"status": "FINAL_CHECKPOINT_FROZEN",
        "removed_after_completion": ["last.pt", cache_name], "final_sha256": final_sha})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
