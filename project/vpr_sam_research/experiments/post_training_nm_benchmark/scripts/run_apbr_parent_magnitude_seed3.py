"""APBR-dense-parent magnitude 2:4 ablation.

This intentionally reuses the frozen formal post-processing primitives while
changing only the accepted dense parent identity from ``dense_sam`` to
``apbr_sam``.  Outputs must live outside the formal matrix directory.
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
import run_cifar100_50k_postprocess as formal
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import project, validate_masks

PROTOCOL = "cifar100_50k10k_final_v1"
ABLATION_ID = "apbr_seed3_geometry_vs_magnitude_20261030"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_apbr_parent(parent: dict, seed: int) -> None:
    if parent.get("protocol_id") != PROTOCOL:
        raise ValueError("Parent protocol mismatch")
    if parent.get("epoch") != 200 or parent.get("steps") != 78_200:
        raise ValueError("Parent is not the fixed epoch-200 checkpoint")
    identity = parent.get("identity", {})
    if identity.get("method") != "apbr_sam" or identity.get("seed") != seed:
        raise ValueError("Parent is not the matching APBR-SAM dense checkpoint")
    if parent.get("test_accessed") is not False:
        raise ValueError("Parent Test policy mismatch")
    train.validate_checkpoint(parent, identity)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--geometry-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3,), required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing nonempty ablation output")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL:
        raise ValueError("Frozen formal config mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    validate_apbr_parent(parent, args.seed)
    geometry_reference = torch.load(
        args.geometry_reference, map_location="cpu", weights_only=False
    )
    if (
        geometry_reference.get("protocol_id") != PROTOCOL
        or geometry_reference.get("stage") != "raw_hard"
        or geometry_reference.get("method") != "apbr_sam"
        or geometry_reference.get("seed") != args.seed
        or geometry_reference.get("parent", {}).get("sha256")
        != train.digest_file(args.parent)
    ):
        raise ValueError("Geometry reference is not bound to this APBR parent")

    train.seed_all(args.seed)
    device = torch.device("cuda:0")
    train_file = args.dataset_root / "extracted/cifar-100-python/train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,):
        raise ValueError("Official CIFAR-100 train identity mismatch")

    plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                config["dataset"]["normalize_mean"],
                config["dataset"]["normalize_std"],
            ),
        ]
    )
    augment = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                config["dataset"]["normalize_mean"],
                config["dataset"]["normalize_std"],
            ),
        ]
    )
    calibration_ids = np.random.default_rng(
        config["formal_calibration"]["selection_seed"]
    ).permutation(50_000)[:1024].tolist()
    bn_ids = np.random.default_rng(config["bn_only"]["selection_seed"]).permutation(
        50_000
    )[:4096].tolist()

    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"])
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()

    masks, pruning_report = formal.prune(
        model,
        "magnitude",
        formal.loader(formal.Subset(data, labels, calibration_ids, plain)),
        parent,
        device,
    )
    parent_sha = train.digest_file(args.parent)
    raw_payload = {
        "schema": 1,
        "status": "healthy",
        "protocol_id": PROTOCOL,
        "ablation_id": ABLATION_ID,
        "stage": "raw_hard",
        "method": "magnitude",
        "dense_parent_method": "apbr_sam",
        "seed": args.seed,
        "epoch": 0,
        "model": formal.model_state(model),
        "masks": {name: mask.cpu() for name, mask in masks.items()},
        "mask_sha256": train.tensor_sha(masks),
        "parent": {"path": str(args.parent.resolve()), "sha256": parent_sha},
        "geometry_reference": {
            "path": str(args.geometry_reference.resolve()),
            "sha256": train.digest_file(args.geometry_reference),
            "mask_sha256": geometry_reference["mask_sha256"],
        },
        "pruning": pruning_report,
        "calibration_ids": calibration_ids,
        "test_accessed": False,
    }
    raw_path = output / "raw_hard.pt"
    raw_sha = formal.atomic_save(raw_path, raw_payload)

    calibrated = copy.deepcopy(model)
    before = train.tensor_sha(dict(calibrated.named_parameters()))
    for module in calibrated.modules():
        module.eval()
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
            module.train()
    with torch.no_grad():
        for inputs, _, _ in formal.loader(
            formal.Subset(data, labels, bn_ids, plain)
        ):
            calibrated(inputs.to(device, non_blocking=True))
    if train.tensor_sha(dict(calibrated.named_parameters())) != before:
        raise ValueError("BN-only changed parameters")
    bn_payload = dict(raw_payload)
    bn_payload.update(
        stage="bn_only",
        model=formal.model_state(calibrated),
        bn_ids=bn_ids,
        raw_parent_sha256=raw_sha,
    )
    bn_sha = formal.atomic_save(output / "bn_only.pt", bn_payload)
    del calibrated

    full_train = train.OfficialTrain(data, labels, augment)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=0.0
    )
    history: list[dict] = []
    recovery_started = time.perf_counter()
    final = None
    for epoch in range(20):
        model.train()
        loss_sum = 0.0
        correct = count = batches = 0
        order = hashlib.sha256()
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_started = time.perf_counter()
        for inputs, targets, sample_ids in train.training_loader(
            full_train, 128, args.seed, epoch
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Recovery loss")
            loss.backward()
            project(model, masks, optimizer)
            optimizer.step()
            project(model, masks, optimizer)
            loss_sum += float(loss) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
            batches += 1
            order.update(sample_ids.numpy().astype("<i8").tobytes())
        if count != 50_000 or batches != 391:
            raise ValueError("Recovery coverage")
        validate_masks(model, masks, n=2, m=4, complete=True)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "steps": (epoch + 1) * 391,
            "train_samples": count,
            "batches": batches,
            "train_order_sha256": order.hexdigest(),
            "learning_rate": learning_rate,
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "test_accessed": False,
        }
        history.append(row)
        final = {
            "schema": 1,
            "status": "healthy",
            "protocol_id": PROTOCOL,
            "ablation_id": ABLATION_ID,
            "stage": "recovery",
            "selection_mode": "fixed_final_epoch",
            "method": "magnitude",
            "dense_parent_method": "apbr_sam",
            "seed": args.seed,
            "epoch": epoch + 1,
            "steps": (epoch + 1) * 391,
            "model": formal.model_state(model),
            "masks": raw_payload["masks"],
            "mask_sha256": raw_payload["mask_sha256"],
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "parent": {"path": str(raw_path), "sha256": raw_sha},
            "test_accessed": False,
        }
        last_sha = formal.atomic_save(output / "recovery/last.pt", final)
        atomic_json(
            output / "status.json",
            {
                "status": "recovery",
                "ablation_id": ABLATION_ID,
                "epoch": epoch + 1,
                "steps": final["steps"],
                "last_sha256": last_sha,
                "test_accessed": False,
            },
        )
        print(json.dumps({"seed": args.seed, **row}, allow_nan=False), flush=True)

    if final is None or final["epoch"] != 20 or final["steps"] != 7820:
        raise RuntimeError("Recovery budget")
    final_sha = formal.atomic_save(output / "recovery/final_epoch_020.pt", final)
    torch.cuda.synchronize(device)
    atomic_json(
        output / "status.json",
        {
            "status": "completed",
            "protocol_id": PROTOCOL,
            "ablation_id": ABLATION_ID,
            "method": "magnitude",
            "dense_parent_method": "apbr_sam",
            "seed": args.seed,
            "raw_hard_sha256": raw_sha,
            "bn_only_sha256": bn_sha,
            "recovery_final_sha256": final_sha,
            "mask_sha256": raw_payload["mask_sha256"],
            "geometry_reference_mask_sha256": geometry_reference["mask_sha256"],
            "recovery_epochs": 20,
            "recovery_steps": 7820,
            "test_accessed": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "recovery_elapsed_seconds": time.perf_counter() - recovery_started,
            "total_elapsed_seconds": time.perf_counter() - run_started,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            failed_output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            atomic_json(
                failed_output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                },
            )
            atomic_json(
                failed_output / "status.json",
                {"status": "failed", "message": str(error)},
            )
        raise
