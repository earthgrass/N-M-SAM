"""Frozen Strong four-method SAM experiment runner.

Formal configurations require ``--allow-full-training``.  CPU smoke configs
carry a finite ``max_steps`` and therefore cannot accidentally start the
200-epoch experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    from nm_sam_cifar10.optim.optimizer_factory import base_optimizer, build_optimizer
    from nm_sam_cifar10.sparsity import SparsityConfig
    from nm_sam_cifar10.sparsity.mask_manager import MaskManager
    from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn
    from nm_sam_cifar10.trainers.sam_step import sam_step
    from nm_sam_cifar10.utils.seed import set_seed
else:
    from .data.cifar10 import build_cifar10
    from .data.strong_baseline_transforms import build_strong_baseline_transforms
    from .metrics.sanity import check_block_sums
    from .metrics.sparsity_metrics import collect_sparsity_metrics
    from .models import build_model
    from .optim.optimizer_factory import base_optimizer, build_optimizer
    from .sparsity import SparsityConfig
    from .sparsity.mask_manager import MaskManager
    from .sparsity.parametrization import make_param_mask_fn
    from .trainers.sam_step import sam_step
    from .utils.seed import set_seed


@dataclass(frozen=True)
class StrongSAMConfig:
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
    rho: float = 0.05
    adaptive: bool = False
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

    @property
    def sparse(self) -> bool:
        return self.method != "strong_dense_sam"


def load_config(path: Path) -> StrongSAMConfig:
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Config must be JSON-compatible YAML when PyYAML is unavailable") from exc
        raw = yaml.safe_load(text)
    known = {field.name for field in fields(StrongSAMConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")
    config = StrongSAMConfig(**raw)
    validate_config(config)
    return config


METHOD_SCORES = {
    "strong_dense_sam": None,
    "strong_2to4_sam_magnitude": "magnitude",
    "strong_2to4_sam_taylor_g1": "taylor_g1",
    "strong_2to4_sam_taylor_g2": "taylor_g2",
}


def validate_config(config: StrongSAMConfig) -> None:
    if config.method not in METHOD_SCORES:
        raise ValueError(f"Unsupported Strong method: {config.method}")
    frozen = {
        "model": "resnet18_cifar", "seed": 0, "epochs": 200,
        "batch_size": 128, "lr": 0.1, "momentum": 0.9,
        "weight_decay": 5e-4, "nesterov": False, "rho": 0.05,
        "adaptive": False, "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": 200, "warmup_epochs": 0, "precision": "fp32",
    }
    drift = {name: (getattr(config, name), value) for name, value in frozen.items()
             if getattr(config, name) != value}
    if drift:
        raise ValueError(f"Frozen Strong protocol drift: {drift}")
    if config.sparse:
        sparse_frozen = {"n": 2, "m": 4, "sparse_scope": "phase1",
                         "mask_score": METHOD_SCORES[config.method],
                         "mask_update_interval": 20}
        drift = {name: (getattr(config, name), value) for name, value in sparse_frozen.items()
                 if getattr(config, name) != value}
        if drift:
            raise ValueError(f"Frozen 2:4 protocol drift: {drift}")
    if config.max_steps is not None and not 1 <= config.max_steps <= 50:
        raise ValueError("Smoke/preflight max_steps must be in [1, 50]")


def _capture_rng() -> Dict[str, Any]:
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _setup(config: StrongSAMConfig):
    model = build_model(config.model, num_classes=10)
    manager = None
    if config.sparse:
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
        rho = config.rho
        adaptive = config.adaptive

    optimizer = build_optimizer(model, "sam", _Args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer(optimizer), T_max=config.scheduler_t_max
    )
    return model, manager, optimizer, scheduler, make_param_mask_fn(model)


def _mask_clone(model: nn.Module):
    return [wrapper.mask.detach().clone() for wrapper in getattr(model, "_nm_wrappers", [])]


def _mask_diff(before, after) -> Tuple[int, int]:
    changed = sum(int((left != right).sum().item()) for left, right in zip(before, after))
    total = sum(left.numel() for left in before)
    return changed, total


def save_checkpoint(
    path: Path, model, optimizer, scheduler, manager, epoch, global_step,
    mask_update_interval: int = 20, extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.base_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch), "global_step": int(global_step),
        "rng_state": _capture_rng(),
        "mask_manager": manager.state_dict() if manager is not None else None,
        "next_mask_update_step": (
            ((global_step // mask_update_interval) + 1) * mask_update_interval
            if manager is not None else None
        ),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, manager):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.base_optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if manager is not None:
        if payload["mask_manager"] is None:
            raise RuntimeError("Sparse checkpoint is missing MaskManager state")
        manager.load_state_dict(payload["mask_manager"])
    _restore_rng(payload["rng_state"])
    return payload


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


def run(config: StrongSAMConfig, allow_full_training: bool = False) -> Dict[str, Any]:
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
    if config.subset_train is not None:
        indices = list(range(min(config.subset_train, len(train_set))))
        train_set = Subset(train_set, indices)
        clean_train_set = Subset(clean_train_set, indices)
    if config.subset_test is not None:
        indices = list(range(min(config.subset_test, len(test_set))))
        test_set = Subset(test_set, indices)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, generator=generator, drop_last=False,
    )
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers)
    clean_train_loader = DataLoader(
        clean_train_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    if not is_smoke and (len(train_set) != 50000 or len(test_set) != 10000):
        raise RuntimeError(
            f"Formal CIFAR-10 cardinality mismatch: train={len(train_set)}, test={len(test_set)}"
        )
    model, manager, optimizer, scheduler, mask_fn = _setup(config)
    if manager is not None:
        manager.write_report(output_dir / "SPARSE_LAYER_REPORT.txt")
    criterion = nn.CrossEntropyLoss()

    wrappers = list(getattr(model, "_nm_wrappers", []))
    bn_modules = [module for module in model.modules()
                  if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    wrapper_count = len(wrappers)
    dense_structure_ok = (not config.sparse and wrapper_count == 0)
    initial_sparsity = collect_sparsity_metrics(model, config.n, config.m)
    if config.sparse and (
        wrapper_count != 17 or initial_sparsity.total_blocks != 2_747_648
        or initial_sparsity.bad_block_count != 0
    ):
        raise RuntimeError(
            "Phase-1 sparse scope mismatch: "
            f"wrappers={wrapper_count}, blocks={initial_sparsity.total_blocks}, "
            f"bad_blocks={initial_sparsity.bad_block_count}"
        )
    global_step = 0
    epoch = 0
    traces = []
    single_batch: Dict[str, Any] = {}
    mask_timing: Dict[str, Any] = {
        "step19_captured": False, "step20_both_forwards_old": False,
        "update_after_step20": False, "step21_uses_new": False,
        "mask_changed_elements": 0, "mask_change_ratio": 0.0,
    }
    step19_mask = step20_new_mask = None
    nan_seen = inf_seen = False
    stop = False
    best_test_accuracy = float("-inf")
    mask_history = []
    epoch_rows = []
    elapsed_before_resume = 0.0
    checkpoint_last = output_dir / "checkpoint_last.pt"
    if not is_smoke and checkpoint_last.exists():
        payload = load_checkpoint(
            checkpoint_last, model, optimizer, scheduler, manager
        )
        epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        extra = payload.get("extra", {})
        best_test_accuracy = float(extra.get("best_test_accuracy", best_test_accuracy))
        mask_history = list(extra.get("mask_history", []))
        epoch_rows = list(extra.get("epoch_rows", []))
        elapsed_before_resume = float(extra.get("total_training_time", 0.0))
        if "train_generator_state" in extra:
            generator.set_state(extra["train_generator_state"])

    formal_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while epoch < config.epochs and not stop:
        epoch_started = time.perf_counter()
        epoch_center_sum = 0.0
        epoch_adversarial_sum = 0.0
        epoch_examples = 0
        epoch_eps_ratio_sum = 0.0
        epoch_batches = 0
        epoch_changes = []
        for inputs, targets in train_loader:
            step_number = global_step + 1
            if is_smoke and config.sparse and step_number == 19:
                step19_mask = _mask_clone(model)
                mask_timing["step19_captured"] = True
            observed_inputs = []
            observed_masks = []

            def observer(_pass_name, tensor):
                observed_inputs.append(tensor.detach().clone())
                observed_masks.append(_mask_clone(model))

            bn_before = [int(module.num_batches_tracked.item()) for module in bn_modules]
            update_boundary = bool(
                manager is not None
                and step_number % config.mask_update_interval == 0
            )
            result = sam_step(
                model, optimizer, criterion, inputs, targets, device,
                param_mask_fn=mask_fn if config.sparse else None,
                mask_score=config.mask_score if update_boundary else None,
                input_observer=observer if is_smoke else None,
                capture_masks=is_smoke,
            )
            bn_after = [int(module.num_batches_tracked.item()) for module in bn_modules]
            bn_ok = all(after == before + 1 for before, after in zip(bn_before, bn_after))
            same_input = result.first_input_data_ptr == result.second_input_data_ptr
            if is_smoke:
                same_input = bool(
                    len(observed_inputs) == 2
                    and torch.equal(observed_inputs[0], observed_inputs[1])
                    and same_input
                )
            masks_same = True
            if is_smoke and config.sparse:
                masks_same = all(torch.equal(left, right)
                                 for left, right in zip(observed_masks[0], observed_masks[1]))
            if is_smoke and config.sparse and step_number == 20:
                mask_timing["step20_both_forwards_old"] = bool(
                    step19_mask is not None
                    and all(torch.equal(left, right) for left, right in zip(step19_mask, observed_masks[0]))
                    and masks_same
                )

            global_step = step_number
            update_change = 0.0
            if update_boundary:
                old_mask = _mask_clone(model)
                if len(result.mask_scores) != wrapper_count:
                    raise RuntimeError("Mask-update boundary is missing score snapshots")
                update_change = manager.recompute_from_scores(result.mask_scores)
                new_mask = _mask_clone(model)
                changed, total = _mask_diff(old_mask, new_mask)
                sparsity_now = collect_sparsity_metrics(model, config.n, config.m)
                if sparsity_now.bad_block_count != 0:
                    raise RuntimeError(f"N:M legality failure at step {global_step}")
                layer_change = manager.layer_change_ratios()
                mask_event = {
                    "step": global_step,
                    "global_change_ratio": float(update_change),
                    "layer_change_ratio": layer_change,
                }
                mask_history.append(mask_event)
                epoch_changes.append(float(update_change))
                if is_smoke and global_step == 20:
                    step20_new_mask = new_mask
                    mask_timing.update({
                        "update_after_step20": True,
                        "mask_changed_elements": changed,
                        "mask_change_ratio": changed / max(total, 1),
                    })
            if is_smoke and config.sparse and global_step == 21 and step20_new_mask is not None:
                mask_timing["step21_uses_new"] = all(
                    torch.equal(left, right) for left, right in zip(step20_new_mask, observed_masks[0])
                )

            loss_values = (result.center_loss, result.adversarial_loss)
            nan_seen = nan_seen or any(math.isnan(value) for value in loss_values)
            inf_seen = inf_seen or any(math.isinf(value) for value in loss_values)
            trace = {
                "step": global_step, "center_loss": result.center_loss,
                "adversarial_loss": result.adversarial_loss,
                "g1_finite": result.g1_finite, "g2_finite": result.g2_finite,
                "epsilon_over_rho": result.info["eps_norm_over_rho"],
                "masked_epsilon_max_abs": result.info.get("masked_epsilon_max_abs", 0.0),
                "same_augmented_tensor": same_input, "bn_counter_correct": bn_ok,
                "mask_fixed_inside_sam_step": masks_same,
                "restore_max_abs_error": result.restore_info["restore_max_abs_error"],
                "leftover_epsilon_buffers": result.restore_info["leftover_epsilon_buffers"],
                "mask_update_change_ratio": update_change,
            }
            if is_smoke:
                traces.append(trace)
            if global_step == 1:
                single_batch = dict(trace)
            if not all((result.g1_finite, result.g2_finite, bn_ok, same_input, masks_same)):
                raise RuntimeError(f"Correctness failure at optimizer step {global_step}: {trace}")
            if abs(trace["epsilon_over_rho"] - 1.0) > 1e-5:
                raise RuntimeError(f"SAM radius failure at step {global_step}")
            if trace["leftover_epsilon_buffers"] != 0 or trace["restore_max_abs_error"] != 0:
                raise RuntimeError(f"SAM restore failure at step {global_step}")
            if nan_seen or inf_seen:
                raise RuntimeError(f"NaN/Inf at optimizer step {global_step}")
            batch_size = int(inputs.shape[0])
            epoch_center_sum += result.center_loss * batch_size
            epoch_adversarial_sum += result.adversarial_loss * batch_size
            epoch_examples += batch_size
            epoch_eps_ratio_sum += trace["epsilon_over_rho"]
            epoch_batches += 1
            if config.max_steps is not None and global_step >= config.max_steps:
                stop = True
                break
        if stop:
            break
        scheduler.step()  # exactly once after each completed epoch
        epoch += 1
        epoch_time = time.perf_counter() - epoch_started
        test_metrics = _evaluate(model, test_loader, criterion, device)
        best_test_accuracy = max(best_test_accuracy, test_metrics["accuracy"])
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(base_optimizer(optimizer).param_groups[0]["lr"]),
            "center_train_loss": epoch_center_sum / max(epoch_examples, 1),
            "adversarial_train_loss": epoch_adversarial_sum / max(epoch_examples, 1),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "epsilon_over_rho": epoch_eps_ratio_sum / max(epoch_batches, 1),
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
            config.mask_update_interval,
            extra={
                "best_test_accuracy": best_test_accuracy,
                "mask_history": mask_history,
                "epoch_rows": epoch_rows,
                "total_training_time": total_elapsed,
                "train_generator_state": generator.get_state(),
            },
        )

    sparsity = collect_sparsity_metrics(model, config.n, config.m)
    bad_blocks = int(sparsity.bad_block_count)
    legality = (bad_blocks == 0 and (check_block_sums(model, config.n, config.m) if config.sparse else True))
    checkpoint = output_dir / ("checkpoint_smoke.pt" if is_smoke else "checkpoint_last.pt")
    if is_smoke:
        save_checkpoint(
            checkpoint, model, optimizer, scheduler, manager, epoch, global_step,
            config.mask_update_interval,
        )
    saved_masks = _mask_clone(model)
    saved_params = [parameter.detach().clone() for parameter in model.parameters()]
    restored_model, restored_manager, restored_optimizer, restored_scheduler, _ = _setup(config)
    payload = load_checkpoint(
        checkpoint, restored_model, restored_optimizer, restored_scheduler, restored_manager
    )
    restored_masks = _mask_clone(restored_model)
    checkpoint_ok = (
        payload["global_step"] == global_step and payload["epoch"] == epoch
        and all(torch.equal(left, right) for left, right in zip(saved_masks, restored_masks))
        and all(torch.equal(left, right) for left, right in zip(saved_params, restored_model.parameters()))
        and restored_scheduler.state_dict() == scheduler.state_dict()
        and (manager is None or restored_manager.state_dict() == manager.state_dict())
        and payload["next_mask_update_step"] == (
            ((global_step // config.mask_update_interval) + 1) * config.mask_update_interval
            if manager is not None else None
        )
    )

    if not is_smoke:
        final_test = _evaluate(model, test_loader, criterion, device)
        clean_train = _evaluate(model, clean_train_loader, criterion, device)
        total_training_time = elapsed_before_resume + (time.perf_counter() - formal_start)
        change_values = [item["global_change_ratio"] for item in mask_history]
        layer_names = sorted({
            name for item in mask_history for name in item["layer_change_ratio"]
        })
        layer_mean = {
            name: float(np.mean([
                item["layer_change_ratio"].get(name, 0.0) for item in mask_history
            ]))
            for name in layer_names
        }
        formal_result = {
            "method": config.method,
            "mask_score": METHOD_SCORES[config.method],
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
            "center_train_loss": epoch_rows[-1]["center_train_loss"],
            "adversarial_train_loss": epoch_rows[-1]["adversarial_train_loss"],
            "epsilon_over_rho": epoch_rows[-1]["epsilon_over_rho"],
            "nan": nan_seen,
            "inf": inf_seen,
            "actual_sparsity": sparsity.actual_sparsity,
            "sparse_wrapper_count": wrapper_count,
            "total_complete_blocks": sparsity.total_blocks,
            "bad_block_count": bad_blocks,
            "mask_change_mean": float(np.mean(change_values)) if change_values else 0.0,
            "mask_change_final": change_values[-1] if change_values else 0.0,
            "layer_mask_change_mean": layer_mean,
            "checkpoint_resume": checkpoint_ok,
            "full_training_completed": epoch == config.epochs,
        }
        (output_dir / "final_metrics.json").write_text(
            json.dumps(formal_result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return formal_result

    result = {
        "method": config.method, "device": str(device), "global_step": global_step,
        "epochs_touched": epoch, "single_batch": single_batch, "steps": traces,
        "epsilon_over_rho": traces[-1]["epsilon_over_rho"],
        "same_augmented_tensor": all(item["same_augmented_tensor"] for item in traces),
        "bn_counter_correct": all(item["bn_counter_correct"] for item in traces),
        "restore_correct": all(item["restore_max_abs_error"] == 0 and item["leftover_epsilon_buffers"] == 0 for item in traces),
        "g1_g2_finite": all(item["g1_finite"] and item["g2_finite"] for item in traces),
        "dense_wrapper_count": wrapper_count if not config.sparse else None,
        "dense_structure_ok": dense_structure_ok if not config.sparse else None,
        "masked_epsilon_zero": all(item["masked_epsilon_max_abs"] == 0 for item in traces),
        "bad_block_count": bad_blocks, "nm_legality": legality,
        "mask_timing": mask_timing, "checkpoint_resume": checkpoint_ok,
        "nan": nan_seen, "inf": inf_seen, "full_training_started": False,
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


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
