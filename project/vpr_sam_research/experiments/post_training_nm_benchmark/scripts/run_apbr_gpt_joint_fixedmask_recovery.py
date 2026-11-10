"""Fixed-mask Recovery-20 from one frozen APBR-GPT matrix Raw/Native endpoint."""
from __future__ import annotations

import argparse
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
import run_cifar100_50k_formal_training as base
import run_cifar100_50k_postprocess as legacy
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import project, validate_masks

PROTOCOL = "cifar100_50k10k_final_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL or config["recovery"]["epochs"] != 20:
        raise ValueError("Recovery protocol mismatch")
    raw_sha = base.digest_file(args.raw_checkpoint)
    raw = torch.load(args.raw_checkpoint, map_location="cpu", weights_only=False)
    if raw.get("protocol_id") != PROTOCOL or raw.get("stage") != "raw_hard" or raw.get("test_accessed") is not False:
        raise ValueError("Raw/Native parent identity mismatch")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError("Refusing nonempty recovery output")
    with (args.dataset_root / "extracted/cifar-100-python/train").open("rb") as stream:
        record = pickle.load(stream, encoding="bytes")
    data = np.asarray(record[b"data"], dtype=np.uint8); labels = np.asarray(record[b"fine_labels"], dtype=np.int64)
    augment = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(.5), transforms.ToTensor(), transforms.Normalize(
            config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"])])
    base.seed_all(0); device = torch.device("cuda:0")
    model = build_cifar100_model().to(device); model.load_state_dict(raw["model"])
    masks = {key: value.to(device=device, dtype=torch.bool) for key, value in raw["masks"].items()}
    validate_masks(model, masks, n=2, m=4, complete=True); project(model, masks)
    dataset = base.OfficialTrain(data, labels, augment)
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005, nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
    history = []; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    final = None
    for epoch in range(20):
        model.train(); loss_sum = 0.; correct = count = batches = 0; order = hashlib.sha256()
        epoch_started = time.perf_counter(); learning_rate = optimizer.param_groups[0]["lr"]
        for inputs, targets, ids in base.training_loader(dataset, 128, 0, epoch):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); logits = model(inputs); loss = nn.functional.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite Recovery loss")
            loss.backward(); project(model, masks, optimizer); optimizer.step(); project(model, masks, optimizer)
            loss_sum += float(loss) * len(targets); correct += int((logits.argmax(1) == targets).sum())
            count += len(targets); batches += 1; order.update(ids.numpy().astype("<i8").tobytes())
        if count != 50_000 or batches != 391: raise RuntimeError("Recovery coverage mismatch")
        validate_masks(model, masks, n=2, m=4, complete=True); scheduler.step(); torch.cuda.synchronize(device)
        row = {"epoch": epoch + 1, "steps": (epoch + 1) * 391, "train_samples": count,
            "batches": batches, "train_order_sha256": order.hexdigest(), "learning_rate": learning_rate,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "epoch_seconds": time.perf_counter() - epoch_started, "test_accessed": False}
        history.append(row)
        final = {"schema": 1, "status": "healthy", "protocol_id": PROTOCOL, "stage": "recovery",
            "selection_mode": "fixed_final_epoch", "method": raw["method"], "seed": 0,
            "epoch": epoch + 1, "steps": (epoch + 1) * 391, "model": legacy.model_state(model),
            "masks": {key: value.cpu() for key, value in masks.items()}, "mask_sha256": raw["mask_sha256"],
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
            "parent": {"path": str(args.raw_checkpoint), "sha256": raw_sha}, "test_accessed": False}
        legacy.atomic_save(output / "last.pt", final)
        base.atomic_json(output / "status.json", {"status": "recovery", "epoch": epoch + 1,
            "steps": final["steps"], "test_accessed": False})
        print(json.dumps(row), flush=True)
    final_sha = legacy.atomic_save(output / "final_epoch_020.pt", final)
    base.atomic_json(output / "status.json", {"status": "completed", "protocol_id": PROTOCOL,
        "method": raw["method"], "raw_parent_sha256": raw_sha, "recovery_final_sha256": final_sha,
        "mask_sha256": raw["mask_sha256"], "recovery_epochs": 20, "recovery_steps": 7820,
        "test_accessed": False, "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1]); target.mkdir(parents=True, exist_ok=True)
            base.atomic_json(target / "failure.json", {"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False, "test_accessed": False})
        raise
