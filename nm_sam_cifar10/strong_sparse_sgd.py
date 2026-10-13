"""Frozen Strong 2:4-SGD-Magnitude experiment runner.

This runner is intentionally isolated from ``strong_sam.py``.  It composes
the existing SGD step, Phase-1 MaskManager, N:M parametrization, CIFAR-10
augmentation, model, and optimizer factory without exposing any SAM option.
Formal 200-epoch training requires ``--allow-full-training``; smoke runs are
hard-limited to 20--50 optimizer steps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nm_sam_cifar10.data.cifar10 import build_cifar10
    from nm_sam_cifar10.data.strong_baseline_transforms import build_strong_baseline_transforms
    from nm_sam_cifar10.metrics.sanity import check_block_sums
    from nm_sam_cifar10.metrics.sparsity_metrics import collect_sparsity_metrics
    from nm_sam_cifar10.models import build_model
    from nm_sam_cifar10.optim.optimizer_factory import build_optimizer
    from nm_sam_cifar10.sparsity import SparsityConfig
    from nm_sam_cifar10.sparsity.mask_manager import MaskManager
    from nm_sam_cifar10.trainers.sgd_step import SGDStepResult, sgd_step
    from nm_sam_cifar10.utils.seed import set_seed
else:
    from .data.cifar10 import build_cifar10
    from .data.strong_baseline_transforms import build_strong_baseline_transforms
    from .metrics.sanity import check_block_sums
    from .metrics.sparsity_metrics import collect_sparsity_metrics
    from .models import build_model
    from .optim.optimizer_factory import build_optimizer
    from .sparsity import SparsityConfig
    from .sparsity.mask_manager import MaskManager
    from .trainers.sgd_step import SGDStepResult, sgd_step
    from .utils.seed import set_seed


METHOD = "strong_2to4_sgd_magnitude"
DISPLAY_NAME = "2:4-SGD-Magnitude"


@dataclass(frozen=True)
class StrongSparseSGDConfig:
    method: str
    output_dir: str
    data_root: str = "./data/cifar10"
    model: str = "resnet18_cifar"
    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False
    scheduler: str = "CosineAnnealingLR"
    scheduler_t_max: int = 200
    warmup_epochs: int = 0
    precision: str = "fp32"
    n: int = 2
    m: int = 4
    sparse_scope: str = "phase1"
    mask_score: str = "magnitude"
    mask_update_interval: int = 20
    device: str = "cpu"
    num_workers: int = 0
    subset_train: Optional[int] = None
    subset_test: Optional[int] = None
    max_steps: Optional[int] = None


@dataclass(frozen=True)
class SparseSGDStepResult:
    train: SGDStepResult
    mask_updated: bool
    mask_change_ratio: float
    gradients_finite: bool
    mask_score_reference: str


def load_config(path: Path) -> StrongSparseSGDConfig:
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Config must be JSON-compatible YAML when PyYAML is unavailable") from exc
        raw = yaml.safe_load(text)
    known = {field.name for field in fields(StrongSparseSGDConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")
    config = StrongSparseSGDConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: StrongSparseSGDConfig) -> None:
    frozen = {
        "method": METHOD, "model": "resnet18_cifar", "seed": 0,
        "epochs": 200, "batch_size": 128, "lr": 0.1,
        "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False,
        "scheduler": "CosineAnnealingLR", "scheduler_t_max": 200,
        "warmup_epochs": 0, "precision": "fp32", "n": 2, "m": 4,
        "sparse_scope": "phase1", "mask_score": "magnitude",
        "mask_update_interval": 20,
    }
    drift = {
        name: (getattr(config, name), expected)
        for name, expected in frozen.items()
        if getattr(config, name) != expected
    }
    if drift:
        raise ValueError(f"Frozen 2:4-SGD-Magnitude protocol drift: {drift}")
    if config.max_steps is not None and not 20 <= config.max_steps <= 50:
        raise ValueError("CPU smoke max_steps must be in [20, 50]")


def _capture_rng() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu().to(dtype=torch.uint8))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [item.detach().cpu().to(dtype=torch.uint8) for item in state["torch_cuda"]]
        )


def _setup(config: StrongSparseSGDConfig):
    model = build_model(config.model, num_classes=10)
    manager = MaskManager(
        model,
        SparsityConfig(n=config.n, m=config.m, enabled=True, score=config.mask_score),
        scope=config.sparse_scope,
    )
    model = model.to(torch.device(config.device))

    class _Args:
        lr = config.lr
        momentum = config.momentum
        weight_decay = config.weight_decay
        nesterov = config.nesterov

    optimizer = build_optimizer(model, "sgd", _Args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_t_max
    )
    return model, manager, optimizer, scheduler


def _mask_clone(model: nn.Module):
    return [wrapper.mask.detach().clone() for wrapper in getattr(model, "_nm_wrappers", [])]


def _mask_diff(before, after) -> Tuple[int, int]:
    changed = sum(int((left != right).sum().item()) for left, right in zip(before, after))
    total = sum(left.numel() for left in before)
    return changed, total


def sparse_sgd_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    manager: MaskManager,
    step_number: int,
    mask_update_interval: int = 20,
) -> SparseSGDStepResult:
    """Run SGD with ``M_(t+1) = Top2(|W_t|)`` at update boundaries.

    The score is captured from the pre-update master weights, while the new
    mask is installed only after the ordinary SGD update.  The forward and
    backward therefore continue to use the current mask ``M_t``.
    """
    update = mask_update_interval > 0 and step_number % mask_update_interval == 0
    if update:
        pre_update_weights = [
            wrapper.weight_master.detach().clone() for wrapper in manager.wrappers
        ]
        mask_scores = [weight.abs() for weight in pre_update_weights]
        del pre_update_weights
    else:
        mask_scores = []
    result = sgd_step(model, optimizer, criterion, inputs, targets, device)
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    change = manager.recompute_from_scores(mask_scores) if update else 0.0
    return SparseSGDStepResult(
        result,
        update,
        float(change),
        gradients_finite,
        "pre_update" if update else "none",
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    manager: MaskManager,
    epoch: int,
    global_step: int,
    loader_generator: torch.Generator,
    mask_update_interval: int = 20,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "mask_manager": manager.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "next_mask_update_step": ((global_step // mask_update_interval) + 1) * mask_update_interval,
        "rng_state": _capture_rng(),
        "train_generator_state": loader_generator.get_state(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, manager, loader_generator, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    manager.load_state_dict(payload["mask_manager"])
    _restore_rng(payload["rng_state"])
    loader_generator.set_state(
        payload["train_generator_state"].detach().cpu().to(dtype=torch.uint8)
    )
    return payload


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(criterion(logits, targets).item()) * inputs.size(0)
        correct += int((logits.argmax(1) == targets).sum().item())
        count += inputs.size(0)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def run(config: StrongSparseSGDConfig, allow_full_training: bool = False) -> Dict[str, Any]:
    validate_config(config)
    if config.max_steps is None and not allow_full_training:
        raise RuntimeError("Formal 200-epoch run requires explicit --allow-full-training")
    is_smoke = config.max_steps is not None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )

    set_seed(config.seed, deterministic=True)
    device = torch.device(config.device)
    train_transform, eval_transform = build_strong_baseline_transforms()
    train_set = build_cifar10(config.data_root, True, train_transform)
    clean_train_set = build_cifar10(config.data_root, True, eval_transform)
    test_set = build_cifar10(config.data_root, False, eval_transform)
    # ``build_cifar10`` retains a legacy FakeData fallback.  This experiment
    # explicitly forbids it, including for smoke runs.
    if len(train_set) != 50_000 or len(clean_train_set) != 50_000 or len(test_set) != 10_000:
        raise RuntimeError(
            "Real CIFAR-10 is required; FakeData/fallback cardinality detected: "
            f"train={len(train_set)}, clean_train={len(clean_train_set)}, test={len(test_set)}"
        )
    if config.subset_train is not None:
        indices = list(range(min(config.subset_train, len(train_set))))
        train_set = Subset(train_set, indices)
        clean_train_set = Subset(clean_train_set, indices)
    if config.subset_test is not None:
        test_set = Subset(test_set, list(range(min(config.subset_test, len(test_set)))))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, generator=generator, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    clean_train_loader = DataLoader(
        clean_train_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    if not is_smoke and (len(train_set) != 50_000 or len(test_set) != 10_000):
        raise RuntimeError(
            f"Formal CIFAR-10 cardinality mismatch: train={len(train_set)}, test={len(test_set)}"
        )

    model, manager, optimizer, scheduler = _setup(config)
    manager.write_report(output_dir / "SPARSE_LAYER_REPORT.txt")
    criterion = nn.CrossEntropyLoss()
    wrappers = list(getattr(model, "_nm_wrappers", []))
    initial_sparsity = collect_sparsity_metrics(model, config.n, config.m)
    if (
        len(wrappers) != 17
        or initial_sparsity.total_blocks != 2_747_648
        or initial_sparsity.bad_block_count != 0
    ):
        raise RuntimeError(
            "Phase-1 sparse scope mismatch: "
            f"wrappers={len(wrappers)}, blocks={initial_sparsity.total_blocks}, "
            f"bad_blocks={initial_sparsity.bad_block_count}"
        )

    global_step = 0
    epoch = 0
    best_test_accuracy = float("-inf")
    mask_history = []
    epoch_rows = []
    traces = []
    nan_seen = inf_seen = False
    elapsed_before_resume = 0.0
    checkpoint_last = output_dir / "checkpoint_last.pt"
    if not is_smoke and checkpoint_last.exists():
        payload = load_checkpoint(
            checkpoint_last, model, optimizer, scheduler, manager, generator, device
        )
        epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        extra = payload.get("extra", {})
        best_test_accuracy = float(extra.get("best_test_accuracy", best_test_accuracy))
        mask_history = list(extra.get("mask_history", []))
        epoch_rows = list(extra.get("epoch_rows", []))
        elapsed_before_resume = float(extra.get("total_training_time", 0.0))

    mask_timing = {
        "step19_captured": False,
        "step20_used_old_mask": False,
        "score_captured_before_optimizer_step20": False,
        "update_after_optimizer_step20": False,
        "step21_uses_new_mask": False,
        "mask_generation_after_step20": 0,
        "mask_changed_elements": 0,
        "mask_change_ratio": 0.0,
    }
    step19_mask = step20_new_mask = None
    stop = False
    formal_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while epoch < config.epochs and not stop:
        epoch_started = time.perf_counter()
        epoch_loss_sum = epoch_correct = epoch_examples = epoch_batches = 0
        epoch_changes = []
        for inputs, targets in train_loader:
            step_number = global_step + 1
            pre_step_mask = _mask_clone(model)
            if is_smoke and step_number == 19:
                step19_mask = pre_step_mask
                mask_timing["step19_captured"] = True
            if is_smoke and step_number == 20:
                mask_timing["step20_used_old_mask"] = bool(
                    step19_mask is not None
                    and all(torch.equal(a, b) for a, b in zip(step19_mask, pre_step_mask))
                )
            if is_smoke and step_number == 21 and step20_new_mask is not None:
                mask_timing["step21_uses_new_mask"] = all(
                    torch.equal(a, b) for a, b in zip(step20_new_mask, pre_step_mask)
                )

            result = sparse_sgd_step(
                model, optimizer, criterion, inputs, targets, device, manager,
                step_number, config.mask_update_interval,
            )
            global_step = step_number
            loss = result.train.loss
            nan_seen = nan_seen or math.isnan(loss)
            inf_seen = inf_seen or math.isinf(loss)
            if nan_seen or inf_seen or not result.gradients_finite:
                raise RuntimeError(
                    f"Non-finite loss/gradient at optimizer step {global_step}"
                )

            changed = total = 0
            if result.mask_updated:
                new_mask = _mask_clone(model)
                changed, total = _mask_diff(pre_step_mask, new_mask)
                sparsity_now = collect_sparsity_metrics(model, config.n, config.m)
                if sparsity_now.bad_block_count != 0:
                    raise RuntimeError(f"N:M legality failure at step {global_step}")
                event = {
                    "step": global_step,
                    "global_change_ratio": result.mask_change_ratio,
                    "layer_change_ratio": manager.layer_change_ratios(),
                }
                mask_history.append(event)
                epoch_changes.append(result.mask_change_ratio)
                if is_smoke and global_step == 20:
                    step20_new_mask = new_mask
                    mask_timing.update({
                        "score_captured_before_optimizer_step20": (
                            result.mask_score_reference == "pre_update"
                        ),
                        "update_after_optimizer_step20": True,
                        "mask_generation_after_step20": min(w.generation for w in wrappers),
                        "mask_changed_elements": changed,
                        "mask_change_ratio": changed / max(total, 1),
                    })

            batch_size = result.train.batch_size
            epoch_loss_sum += loss * batch_size
            epoch_correct += int(round(result.train.accuracy * batch_size))
            epoch_examples += batch_size
            epoch_batches += 1
            if is_smoke:
                traces.append({
                    "step": global_step,
                    "loss": loss,
                    "loss_finite": math.isfinite(loss),
                    "gradients_finite": result.gradients_finite,
                    "mask_updated": result.mask_updated,
                    "mask_change_ratio": result.mask_change_ratio,
                    "mask_score_reference": result.mask_score_reference,
                })
            if config.max_steps is not None and global_step >= config.max_steps:
                stop = True
                break

        if stop:
            break
        scheduler.step()
        epoch += 1
        epoch_time = time.perf_counter() - epoch_started
        test_metrics = _evaluate(model, test_loader, criterion, device)
        best_test_accuracy = max(best_test_accuracy, test_metrics["accuracy"])
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": epoch_loss_sum / max(epoch_examples, 1),
            "train_accuracy": epoch_correct / max(epoch_examples, 1),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "mask_change_mean": float(np.mean(epoch_changes)) if epoch_changes else 0.0,
            "epoch_time": epoch_time,
            "images_per_second": epoch_examples / max(epoch_time, 1e-12),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        epoch_rows.append(row)
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(epoch_rows)
        (output_dir / "mask_change_history.json").write_text(
            json.dumps(mask_history, indent=2, sort_keys=True), encoding="utf-8"
        )
        total_elapsed = elapsed_before_resume + (time.perf_counter() - formal_start)
        save_checkpoint(
            checkpoint_last, model, optimizer, scheduler, manager, epoch, global_step,
            generator, config.mask_update_interval,
            extra={
                "best_test_accuracy": best_test_accuracy,
                "mask_history": mask_history,
                "epoch_rows": epoch_rows,
                "total_training_time": total_elapsed,
            },
        )

    sparsity = collect_sparsity_metrics(model, config.n, config.m)
    legality = bool(
        sparsity.bad_block_count == 0 and check_block_sums(model, config.n, config.m)
    )
    checkpoint = output_dir / ("checkpoint_smoke.pt" if is_smoke else "checkpoint_last.pt")
    if is_smoke:
        save_checkpoint(
            checkpoint, model, optimizer, scheduler, manager, epoch, global_step,
            generator, config.mask_update_interval,
        )

    saved_masks = _mask_clone(model)
    saved_params = [parameter.detach().clone() for parameter in model.parameters()]
    saved_optimizer = optimizer.state_dict()
    restored_model, restored_manager, restored_optimizer, restored_scheduler = _setup(config)
    restored_generator = torch.Generator()
    payload = load_checkpoint(
        checkpoint, restored_model, restored_optimizer, restored_scheduler,
        restored_manager, restored_generator, device,
    )
    checkpoint_ok = bool(
        payload["global_step"] == global_step
        and payload["epoch"] == epoch
        and all(torch.equal(a, b) for a, b in zip(saved_masks, _mask_clone(restored_model)))
        and all(torch.equal(a, b) for a, b in zip(saved_params, restored_model.parameters()))
        and _nested_equal(saved_optimizer, restored_optimizer.state_dict())
        and _nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
        and manager.state_dict() == restored_manager.state_dict()
        and payload["next_mask_update_step"]
        == ((global_step // config.mask_update_interval) + 1) * config.mask_update_interval
    )

    if is_smoke:
        result = {
            "method": config.method,
            "display_name": DISPLAY_NAME,
            "dataset": "CIFAR-10 (real)",
            "device": str(device),
            "global_step": global_step,
            "loss_finite": all(item["loss_finite"] for item in traces),
            "gradients_finite": all(item["gradients_finite"] for item in traces),
            "steps": traces,
            "bad_block_count": int(sparsity.bad_block_count),
            "nm_legality": legality,
            "sparse_wrapper_count": len(wrappers),
            "total_complete_blocks": int(sparsity.total_blocks),
            "mask_timing": mask_timing,
            "checkpoint_resume": checkpoint_ok,
            "nan": nan_seen,
            "inf": inf_seen,
            "sam_used": False,
            "full_training_started": False,
        }
        (output_dir / "smoke_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    final_test = _evaluate(model, test_loader, criterion, device)
    clean_train = _evaluate(model, clean_train_loader, criterion, device)
    change_values = [item["global_change_ratio"] for item in mask_history]
    total_training_time = elapsed_before_resume + (time.perf_counter() - formal_start)
    formal_result = {
        "method": config.method,
        "display_name": DISPLAY_NAME,
        "optimizer": "SGD",
        "sam_used": False,
        "mask_score": "magnitude",
        "epochs": epoch,
        "global_step": global_step,
        "final_test_accuracy": final_test["accuracy"],
        "final_test_loss": final_test["loss"],
        "clean_train_accuracy": clean_train["accuracy"],
        "clean_train_loss": clean_train["loss"],
        "generalization_gap": clean_train["accuracy"] - final_test["accuracy"],
        "best_test_accuracy": best_test_accuracy,
        "total_training_time": total_training_time,
        "mean_epoch_time": float(np.mean([row["epoch_time"] for row in epoch_rows])),
        "mean_images_per_second": float(np.mean([row["images_per_second"] for row in epoch_rows])),
        "peak_cuda_memory_bytes": max(row["peak_cuda_memory_bytes"] for row in epoch_rows),
        "train_loss": epoch_rows[-1]["train_loss"],
        "nan": nan_seen,
        "inf": inf_seen,
        "actual_sparsity": sparsity.actual_sparsity,
        "sparse_wrapper_count": len(wrappers),
        "total_complete_blocks": sparsity.total_blocks,
        "bad_block_count": int(sparsity.bad_block_count),
        "mask_change_mean": float(np.mean(change_values)) if change_values else 0.0,
        "mask_change_final": change_values[-1] if change_values else 0.0,
        "checkpoint_resume": checkpoint_ok,
        "full_training_completed": epoch == config.epochs,
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(formal_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return formal_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-full-training", action="store_true")
    parser.add_argument("--max-steps-override", type=int)
    parser.add_argument("--output-dir-override", type=str)
    parser.add_argument("--device-override", type=str)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = {}
    if args.max_steps_override is not None:
        overrides["max_steps"] = args.max_steps_override
    if args.output_dir_override is not None:
        overrides["output_dir"] = args.output_dir_override
    if args.device_override is not None:
        overrides["device"] = args.device_override
    if overrides:
        config = replace(config, **overrides)
    result = run(config, allow_full_training=args.allow_full_training)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
