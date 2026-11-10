"""Method-specific CIFAR-100 2:4 endpoint execution.

Baselines stop after their native Raw-Hard endpoint. APBR-SAM alone performs
the preregistered fixed-mask 20-epoch recovery. The legacy pruning functions
remain the single implementation of scoring, reordering, and compensation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_cifar100_50k_formal_training as train
import run_cifar100_50k_postprocess as legacy
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import project, validate_masks

BASE_PROTOCOL = "cifar100_50k10k_final_v1"
COMPARISON_PROTOCOL = "cifar100_method_specific_endpoint_v1"
BASELINES = ("magnitude", "wanda", "sparsegpt", "rose")
METHODS = BASELINES + ("apbr_sam",)


def sha(path: Path) -> str:
    return train.digest_file(path)


def execution_identity(base_config: Path, endpoint_config: Path) -> dict:
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_cifar100_50k_postprocess.py",
        ROOT / "baseline_adapters" / "sparsegpt" / "adapter.py",
        ROOT / "baseline_adapters" / "rose" / "adapter.py",
        ROOT / "baseline_adapters" / "wanda" / "adapter.py",
        ROOT / "benchmark" / "nm.py",
    ]
    return {
        "comparison_protocol_id": COMPARISON_PROTOCOL,
        "base_config_sha256": sha(base_config),
        "endpoint_config_sha256": sha(endpoint_config),
        "source_sha256": {str(path): sha(path) for path in sources},
    }


def check_config(base: dict, endpoint: dict, method: str) -> None:
    if base.get("protocol_id") != BASE_PROTOCOL:
        raise ValueError("Wrong base training protocol")
    if endpoint.get("comparison_protocol_id") != COMPARISON_PROTOCOL or endpoint.get("status") != "FROZEN":
        raise ValueError("Wrong endpoint protocol")
    if endpoint.get("execution", {}).get("mode") == "reporting_overlay_on_existing_full_recovery_queue":
        raise RuntimeError("Reporting-only protocol does not authorize alternate endpoint execution")
    rule = endpoint["methods"][method]
    expected = 20 if method == "apbr_sam" else 0
    if rule["recovery_epochs"] != expected:
        raise ValueError("Method recovery contract mismatch")


def bn_only(model, data, labels, ids, transform, device):
    calibrated = copy.deepcopy(model)
    before = train.tensor_sha(dict(calibrated.named_parameters()))
    for module in calibrated.modules():
        module.eval()
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
            module.train()
    with torch.no_grad():
        for inputs, _, _ in legacy.loader(legacy.Subset(data, labels, ids, transform)):
            calibrated(inputs.to(device, non_blocking=True))
    if train.tensor_sha(dict(calibrated.named_parameters())) != before:
        raise ValueError("BN-only changed parameters")
    return calibrated


def write_complete_baseline(out: Path, method: str, raw_sha: str, bn_sha: str, raw_payload: dict, started: float):
    train.atomic_json(out / "status.json", {
        "status": "completed", "protocol_id": BASE_PROTOCOL,
        "comparison_protocol_id": COMPARISON_PROTOCOL, "method": method,
        "seed": raw_payload["seed"], "primary_endpoint": "raw_hard",
        "raw_hard_sha256": raw_sha, "bn_only_sha256": bn_sha,
        "mask_sha256": raw_payload["mask_sha256"], "recovery_started": False,
        "recovery_epochs": 0, "test_accessed": False,
        "calibration_seconds": raw_payload["pruning"]["calibration_seconds"],
        "pruning_seconds": raw_payload["pruning"]["pruning_seconds"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.perf_counter() - started,
    })


def recover_apbr(out: Path, model, masks, raw_payload: dict, raw_sha: str, data, labels, augment, seed: int):
    full = train.OfficialTrain(data, labels, augment)
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005, nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
    history, final, started = [], None, time.perf_counter()
    for epoch in range(20):
        model.train(); loss_sum = 0.; correct = count = batches = 0
        order = hashlib.sha256(); lr = optimizer.param_groups[0]["lr"]; begun = time.perf_counter()
        for inputs, targets, ids in train.training_loader(full, 128, seed, epoch):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True); logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Recovery loss")
            loss.backward(); project(model, masks, optimizer); optimizer.step(); project(model, masks, optimizer)
            loss_sum += float(loss) * len(targets); correct += int((logits.argmax(1) == targets).sum())
            count += len(targets); batches += 1; order.update(ids.numpy().astype("<i8").tobytes())
        if count != 50_000 or batches != 391:
            raise ValueError("Recovery coverage")
        validate_masks(model, masks, n=2, m=4, complete=True); scheduler.step()
        row = {"epoch": epoch + 1, "steps": (epoch + 1) * 391, "train_samples": count,
               "batches": batches, "train_order_sha256": order.hexdigest(), "learning_rate": lr,
               "train_loss": loss_sum / count, "train_accuracy": correct / count,
               "epoch_seconds": time.perf_counter() - begun, "test_accessed": False}
        history.append(row)
        final = {"schema": 1, "status": "healthy", "protocol_id": BASE_PROTOCOL,
                 "comparison_protocol_id": COMPARISON_PROTOCOL, "stage": "recovery",
                 "selection_mode": "fixed_final_epoch", "method": "apbr_sam", "seed": seed,
                 "epoch": epoch + 1, "steps": (epoch + 1) * 391, "model": legacy.model_state(model),
                 "masks": raw_payload["masks"], "mask_sha256": raw_payload["mask_sha256"],
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "history": history, "parent": {"path": str(out / "raw_hard.pt"), "sha256": raw_sha},
                 "test_accessed": False}
        last_sha = legacy.atomic_save(out / "recovery" / "last.pt", final)
        train.atomic_json(out / "status.json", {"status": "recovery", "comparison_protocol_id": COMPARISON_PROTOCOL,
                          "method": "apbr_sam", "epoch": epoch + 1, "steps": final["steps"],
                          "last_sha256": last_sha, "test_accessed": False})
        print(json.dumps({"method": "apbr_sam", **row}), flush=True)
    if final is None or final["epoch"] != 20 or final["steps"] != 7820:
        raise RuntimeError("Recovery budget")
    return legacy.atomic_save(out / "recovery" / "final_epoch_020.pt", final), time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--endpoint-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Refusing nonempty method output")
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    endpoint = yaml.safe_load(args.endpoint_config.read_text(encoding="utf-8"))
    check_config(base, endpoint, args.method)
    identity = execution_identity(args.base_config, args.endpoint_config)
    if identity["base_config_sha256"] != endpoint["base_training_config_sha256"]:
        raise ValueError("Frozen base config hash mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    expected = "apbr_sam" if args.method == "apbr_sam" else "dense_sam"
    if parent.get("protocol_id") != BASE_PROTOCOL or parent.get("epoch") != 200:
        raise ValueError("Final parent budget/protocol mismatch")
    if parent["identity"]["method"] != expected or parent["identity"]["seed"] != args.seed:
        raise ValueError("Final parent identity mismatch")
    if parent.get("test_accessed") is not False:
        raise ValueError("Parent Test policy")
    train.seed_all(args.seed); device = torch.device("cuda:0")
    with (args.dataset_root / "extracted/cifar-100-python/train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8); labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(base["dataset"]["normalize_mean"], base["dataset"]["normalize_std"])])
    augment = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(.5), transforms.ToTensor(),
                                  transforms.Normalize(base["dataset"]["normalize_mean"], base["dataset"]["normalize_std"])])
    calibration_ids = np.random.default_rng(base["formal_calibration"]["selection_seed"]).permutation(50_000)[:1024].tolist()
    bn_ids = np.random.default_rng(base["bn_only"]["selection_seed"]).permutation(50_000)[:4096].tolist()
    model = build_cifar100_model().to(device); model.load_state_dict(parent["model"])
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    masks, report = legacy.prune(model, args.method, legacy.loader(legacy.Subset(data, labels, calibration_ids, plain)), parent, device)
    raw_payload = {"schema": 1, "status": "healthy", "protocol_id": BASE_PROTOCOL,
                   "comparison_protocol_id": COMPARISON_PROTOCOL, "stage": "raw_hard",
                   "method": args.method, "seed": args.seed, "epoch": 0,
                   "model": legacy.model_state(model), "masks": {key: value.cpu() for key, value in masks.items()},
                   "mask_sha256": train.tensor_sha(masks),
                   "parent": {"path": str(args.parent), "sha256": sha(args.parent)},
                   "pruning": report, "calibration_ids": calibration_ids,
                   "execution_identity": identity, "test_accessed": False}
    raw_sha = legacy.atomic_save(out / "raw_hard.pt", raw_payload)
    calibrated = bn_only(model, data, labels, bn_ids, plain, device)
    bn_payload = dict(raw_payload); bn_payload.update(stage="bn_only", model=legacy.model_state(calibrated),
                                                       bn_ids=bn_ids, raw_parent_sha256=raw_sha)
    bn_sha = legacy.atomic_save(out / "bn_only.pt", bn_payload); del calibrated
    if args.method in BASELINES:
        write_complete_baseline(out, args.method, raw_sha, bn_sha, raw_payload, started)
        return
    recovery_sha, recovery_seconds = recover_apbr(out, model, masks, raw_payload, raw_sha, data, labels, augment, args.seed)
    train.atomic_json(out / "status.json", {
        "status": "completed", "protocol_id": BASE_PROTOCOL,
        "comparison_protocol_id": COMPARISON_PROTOCOL, "method": "apbr_sam", "seed": args.seed,
        "primary_endpoint": "recovery_final_epoch_20", "raw_hard_sha256": raw_sha,
        "bn_only_sha256": bn_sha, "recovery_final_sha256": recovery_sha,
        "mask_sha256": raw_payload["mask_sha256"], "recovery_started": True,
        "recovery_epochs": 20, "recovery_steps": 7820, "test_accessed": False,
        "calibration_seconds": report["calibration_seconds"], "pruning_seconds": report["pruning_seconds"],
        "recovery_seconds": recovery_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            train.atomic_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc(), "automatic_retry": False})
            train.atomic_json(output / "status.json", {"status": "failed", "message": str(error)})
        raise
