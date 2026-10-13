"""Frozen Phase-2A Dense-SGD baseline for CIFAR-10.

This entrypoint is deliberately separate from the research-method runner.  It
contains no SAM, sparsity, mask, STE, or method-selection path.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.cifar10 import CIFAR10_MEAN, CIFAR10_STD, build_cifar10
from .data.strong_baseline_transforms import build_strong_baseline_transforms
from .models import build_model
from .sparsity.parametrization import NMSparsifiedWeight


@dataclass(frozen=True)
class FrozenBaselineConfig:
    dataset: str = "CIFAR-10"
    train_size: int = 50_000
    test_size: int = 10_000
    num_classes: int = 10
    model: str = "resnet18_cifar"
    epochs: int = 200
    batch_size: int = 128
    optimizer: str = "SGD"
    learning_rate: float = 0.1
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 5e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_t_max: int = 200
    scheduler_step_unit: str = "epoch"
    scheduler_step_timing: str = "after each completed epoch"
    warmup_epochs: int = 0
    cutout_holes: int = 1
    cutout_length: int = 16
    label_smoothing: float = 0.0
    mixup: bool = False
    cutmix: bool = False
    autoaugment: bool = False
    randaugment: bool = False
    ema: bool = False
    amp: bool = False
    precision: str = "FP32"
    seed: int = 0
    num_workers: int = 2


CSV_FIELDS = [
    "epoch", "learning_rate", "learning_rate_after_step", "online_train_loss",
    "online_train_accuracy", "test_loss", "test_accuracy", "time_epoch_sec",
    "images_per_second", "test_images_per_second", "nan", "inf",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def config_dict(config: FrozenBaselineConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["train_transform"] = [
        "RandomCrop(32,padding=4)", "RandomHorizontalFlip", "ToTensor",
        "Cutout(n_holes=1,length=16)", "Normalize(CIFAR10_MEAN,CIFAR10_STD)",
    ]
    data["test_transform"] = ["ToTensor", "Normalize(CIFAR10_MEAN,CIFAR10_STD)"]
    data["clean_train_eval_transform"] = list(data["test_transform"])
    return data


def config_hash(config: FrozenBaselineConfig) -> str:
    encoded = json.dumps(config_dict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_info(device: torch.device) -> Dict[str, Any]:
    import torchvision

    info: Dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine(),
        "logical_processors": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision": "FP32",
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def build_datasets(data_root: str):
    train_transform, eval_transform = build_strong_baseline_transforms()
    train_aug = build_cifar10(data_root, train=True, transform=train_transform)
    train_eval = build_cifar10(data_root, train=True, transform=eval_transform)
    test = build_cifar10(data_root, train=False, transform=eval_transform)
    return train_aug, train_eval, test, train_transform, eval_transform


def build_loaders(data_root: str, config: FrozenBaselineConfig, generator: torch.Generator):
    train_aug, train_eval, test, train_transform, eval_transform = build_datasets(data_root)
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
    )
    train_aug_loader = DataLoader(
        train_aug, shuffle=True, drop_last=False, generator=generator, **common
    )
    train_eval_loader = DataLoader(train_eval, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test, shuffle=False, drop_last=False, **common)
    return {
        "train_aug_dataset": train_aug,
        "train_eval_dataset": train_eval,
        "test_dataset": test,
        "train_aug_loader": train_aug_loader,
        "train_eval_loader": train_eval_loader,
        "test_loader": test_loader,
        "train_transform": train_transform,
        "eval_transform": eval_transform,
    }


def make_model(device: torch.device) -> nn.Module:
    return build_model("resnet18_cifar", num_classes=10).to(device)


def make_optimizer(model: nn.Module, config: FrozenBaselineConfig):
    return torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum,
        weight_decay=config.weight_decay, nesterov=config.nesterov,
    )


def make_scheduler(optimizer, config: FrozenBaselineConfig):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_t_max, eta_min=0.0
    )


def stage_shapes(model: nn.Module, device: torch.device) -> Dict[str, List[int]]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.zeros(2, 3, 32, 32, device=device)
        shapes: Dict[str, List[int]] = {"input": list(x.shape)}
        x = F.relu(model.bn1(model.conv1(x)), inplace=False)
        shapes["stem"] = list(x.shape)
        x = model.layer1(x); shapes["layer1"] = list(x.shape)
        x = model.layer2(x); shapes["layer2"] = list(x.shape)
        x = model.layer3(x); shapes["layer3"] = list(x.shape)
        x = model.layer4(x); shapes["layer4"] = list(x.shape)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1); shapes["gap"] = list(x.shape)
        x = model.linear(x); shapes["logits"] = list(x.shape)
    if was_training:
        model.train()
    return shapes


EXPECTED_SHAPES = {
    "input": [2, 3, 32, 32], "stem": [2, 64, 32, 32],
    "layer1": [2, 64, 32, 32], "layer2": [2, 128, 16, 16],
    "layer3": [2, 256, 8, 8], "layer4": [2, 512, 4, 4],
    "gap": [2, 512], "logits": [2, 10],
}


def parameter_statistics(model: nn.Module) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    conv = sum(parameter.numel() for module in model.modules() if isinstance(module, nn.Conv2d) for parameter in module.parameters(recurse=False))
    linear = sum(parameter.numel() for module in model.modules() if isinstance(module, nn.Linear) for parameter in module.parameters(recurse=False))
    batchnorm = sum(parameter.numel() for module in model.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for parameter in module.parameters(recurse=False))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "conv2d_parameters": conv,
        "linear_parameters": linear,
        "batchnorm_learnable_parameters": batchnorm,
        "fp32_parameter_memory_mib": total * 4 / (1024**2),
    }


def dense_audit(model: nn.Module) -> Dict[str, int]:
    wrappers = [module for module in model.modules() if isinstance(module, NMSparsifiedWeight)]
    mask_buffers = [name for name, _ in model.named_buffers() if "mask" in name.lower()]
    sparse_parameters = [
        name for name, _ in model.named_parameters()
        if "weight_master" in name or "sparse" in name.lower()
    ]
    return {
        "number_of_sparse_wrappers": len(wrappers),
        "number_of_nm_masks": len(mask_buffers),
        "number_of_sparse_parameters": len(sparse_parameters),
    }


def optimizer_audit(model: nn.Module, optimizer) -> Dict[str, Any]:
    model_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    counts = {parameter_id: optimizer_ids.count(parameter_id) for parameter_id in model_ids}
    return {
        "class": type(optimizer).__name__,
        "param_groups": [
            {key: value for key, value in group.items() if key != "params"}
            | {"parameter_tensors": len(group["params"])}
            for group in optimizer.param_groups
        ],
        "all_trainable_parameters_present_once": all(count == 1 for count in counts.values()),
        "optimizer_parameter_tensor_count": len(optimizer_ids),
        "trainable_parameter_tensor_count": len(model_ids),
    }


def git_commit(project_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_dir,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable (workspace is not a Git repository)"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run_preflight(data_root: str, output_dir: Path, config: FrozenBaselineConfig, device: torch.device):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    data = build_loaders(data_root, config, generator)
    model = make_model(device)
    optimizer = make_optimizer(model, config)
    shapes = stage_shapes(model, device)
    stats = parameter_statistics(model)
    dense = dense_audit(model)
    optimizer_info = optimizer_audit(model, optimizer)
    environment = environment_info(device)

    failures = []
    if len(data["train_aug_dataset"]) != config.train_size:
        failures.append("train dataset size mismatch")
    if len(data["train_eval_dataset"]) != config.train_size:
        failures.append("clean train-eval dataset size mismatch")
    if len(data["test_dataset"]) != config.test_size:
        failures.append("test dataset size mismatch")
    if shapes != EXPECTED_SHAPES:
        failures.append("architecture stage shape mismatch")
    if any(dense.values()):
        failures.append("sparse state detected in dense baseline")
    if optimizer_info["class"] != "SGD" or not optimizer_info["all_trainable_parameters_present_once"]:
        failures.append("optimizer audit failed")

    architecture_lines = ["CIFAR-ResNet18 architecture sanity"]
    for name, shape in shapes.items():
        architecture_lines.append(f"{name}: {shape}")
    architecture_lines.extend(["", json.dumps(stats, indent=2), "", json.dumps(dense, indent=2)])
    (output_dir / "architecture.txt").write_text("\n".join(architecture_lines) + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "config.json", {
        "config": config_dict(config), "config_hash": config_hash(config)
    })

    audit = f"""# Baseline Pre-flight Audit

Status: **{'PASS' if not failures else 'FAIL'}**

## Dataset audit

- train_dataset_size = {len(data['train_aug_dataset'])}
- train_eval_dataset_size = {len(data['train_eval_dataset'])}
- test_dataset_size = {len(data['test_dataset'])}
- num_classes = {config.num_classes}
- no subset is used

## Transform audit

- train: RandomCrop(32,padding=4) -> RandomHorizontalFlip -> ToTensor -> Cutout -> Normalize
- Cutout: one 16x16 hole; implementation `data/strong_baseline_transforms.py`; applied in the [0,1] tensor domain before normalization
- test: ToTensor -> Normalize only
- clean train-eval: ToTensor -> Normalize only
- mean = {CIFAR10_MEAN}; std = {CIFAR10_STD}
- MixUp/CutMix/AutoAugment/RandAugment/EMA/label smoothing: disabled

## Architecture audit

- model: CIFAR-ResNet18
- stage shapes: `{json.dumps(shapes)}`
- expected shapes matched: {shapes == EXPECTED_SHAPES}
- parameters: `{json.dumps(stats)}`

## Dense-only audit

- number_of_sparse_wrappers = {dense['number_of_sparse_wrappers']}
- number_of_nm_masks = {dense['number_of_nm_masks']}
- number_of_sparse_parameters = {dense['number_of_sparse_parameters']}
- ordinary Conv2d/Linear dense forward only

## Optimizer audit

- optimizer class: {optimizer_info['class']}
- param groups: `{json.dumps(optimizer_info['param_groups'])}`
- every trainable tensor appears exactly once: {optimizer_info['all_trainable_parameters_present_once']}
- no SAM wrapper

## Scheduler audit

- CosineAnnealingLR, T_max=200, eta_min=0
- epoch-level; `scheduler.step()` runs after each completed epoch
- epoch 1 trains at lr=0.1; no pre-epoch scheduler step
- warmup=0

## Reproducibility and precision

- seed=0 for Python, NumPy, and PyTorch; CUDA all-device seed if applicable
- FP32 only; no AMP/BF16/FP16
- environment: `{json.dumps(environment)}`
- deterministic algorithms={environment['deterministic_algorithms']}; cuDNN benchmark={environment['cudnn_benchmark']}

## Failures

{chr(10).join('- ' + failure for failure in failures) if failures else '- none'}
"""
    project_audit = Path(__file__).resolve().parent / "BASELINE_PREFLIGHT_AUDIT.md"
    project_audit.write_text(audit, encoding="utf-8")
    (output_dir / "BASELINE_PREFLIGHT_AUDIT.md").write_text(audit, encoding="utf-8")
    if failures:
        raise RuntimeError("Pre-flight failed: " + "; ".join(failures))
    return data, stats, environment, shapes, optimizer_info


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu().to(dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [rng.detach().cpu().to(dtype=torch.uint8) for rng in state["torch_cuda"]]
        )


def save_checkpoint(
    path: Path, model: nn.Module, optimizer, scheduler, epoch: int,
    global_step: int, config: FrozenBaselineConfig, loader_generator: torch.Generator,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "seed": config.seed,
        "config": config_dict(config),
        "config_hash": config_hash(config),
        "rng_state": capture_rng_state(),
        "data_loader_generator_state": loader_generator.get_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, generator, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_state(payload["rng_state"])
    generator.set_state(
        payload["data_loader_generator_state"].detach().cpu().to(dtype=torch.uint8)
    )
    return int(payload["epoch"]), int(payload["global_step"]), payload


def one_batch_sanity(loader, config: FrozenBaselineConfig, device: torch.device) -> Dict[str, Any]:
    set_seed(config.seed)
    model = make_model(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    parameter_before = next(model.parameters()).detach().clone()
    bn = next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    bn_before = bn.running_mean.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    optimizer.step()
    return {
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": bool(gradients_finite),
        "parameter_changed": not torch.equal(parameter_before, next(model.parameters()).detach()),
        "bn_running_stats_changed": not torch.equal(bn_before, bn.running_mean.detach()),
        "nan": bool(torch.isnan(loss).item()),
        "inf": bool(torch.isinf(loss).item()),
    }


def mini_sanity(loader, config: FrozenBaselineConfig, device: torch.device, iterations: int = 20) -> Dict[str, Any]:
    set_seed(config.seed)
    model = make_model(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    accuracies: List[float] = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= iterations:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        accuracies.append(float((logits.argmax(1) == targets).float().mean().item()))
    first = statistics.mean(losses[:5])
    last = statistics.mean(losses[-5:])
    return {
        "iterations": len(losses),
        "first5_loss": first,
        "last5_loss": last,
        "loss_decreased": last < first,
        "first5_accuracy": statistics.mean(accuracies[:5]),
        "last5_accuracy": statistics.mean(accuracies[-5:]),
        "max_accuracy": max(accuracies),
        "all_finite": all(math.isfinite(value) for value in losses),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


class TinyResumeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, inputs):
        outputs = F.relu(self.bn(self.conv(inputs)))
        return self.fc(F.adaptive_avg_pool2d(outputs, 1).flatten(1))


def _run_tiny_steps(model, optimizer, scheduler, batches, start, end):
    criterion = nn.CrossEntropyLoss()
    for index in range(start, end):
        inputs, targets = batches[index]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def resume_equivalence_sanity() -> Dict[str, Any]:
    set_seed(123)
    initial = TinyResumeCNN()
    initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    batches = [(torch.randn(4, 3, 8, 8), torch.randint(0, 3, (4,))) for _ in range(4)]

    continuous = TinyResumeCNN(); continuous.load_state_dict(initial_state)
    opt_a = torch.optim.SGD(continuous.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=4)
    _run_tiny_steps(continuous, opt_a, sch_a, batches, 0, 4)

    split = TinyResumeCNN(); split.load_state_dict(initial_state)
    opt_b = torch.optim.SGD(split.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=4)
    _run_tiny_steps(split, opt_b, sch_b, batches, 0, 2)
    buffer = io.BytesIO()
    torch.save({
        "model": split.state_dict(), "optimizer": opt_b.state_dict(),
        "scheduler": sch_b.state_dict(), "global_step": 2,
        "rng": capture_rng_state(),
    }, buffer)
    buffer.seek(0)
    restored = TinyResumeCNN()
    opt_c = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=4)
    saved = torch.load(buffer, weights_only=False)
    restored.load_state_dict(saved["model"])
    opt_c.load_state_dict(saved["optimizer"])
    sch_c.load_state_dict(saved["scheduler"])
    restore_rng_state(saved["rng"])
    _run_tiny_steps(restored, opt_c, sch_c, batches, 2, 4)

    max_diff = max(
        float((left - right).abs().max().item())
        for left, right in zip(continuous.parameters(), restored.parameters())
    )
    optimizer_equal = _nested_equal(opt_a.state_dict(), opt_c.state_dict())
    scheduler_equal = _nested_equal(sch_a.state_dict(), sch_c.state_dict())
    return {
        "parameter_max_abs_diff": max_diff,
        "optimizer_state_equal": optimizer_equal,
        "scheduler_state_equal": scheduler_equal,
        "global_step": 4,
        "passed": max_diff == 0.0 and optimizer_equal and scheduler_equal,
    }


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def run_sanity(data, output_dir: Path, config: FrozenBaselineConfig, device: torch.device):
    batch = one_batch_sanity(data["train_aug_loader"], config, device)
    mini = mini_sanity(data["train_aug_loader"], config, device, iterations=20)
    resume = resume_equivalence_sanity()
    result = {"one_batch": batch, "mini_20_iterations": mini, "resume": resume}
    write_json(output_dir / "sanity_results.json", result)
    failures = []
    if not all([
        batch["logits_shape"][1] == 10, batch["loss_finite"], batch["gradients_finite"],
        batch["parameter_changed"], batch["bn_running_stats_changed"],
        not batch["nan"], not batch["inf"],
    ]):
        failures.append("one-batch sanity")
    if not mini["all_finite"] or not mini["loss_decreased"]:
        failures.append("20-iteration mini sanity")
    if not resume["passed"]:
        failures.append("resume equivalence sanity")
    if failures:
        raise RuntimeError("Sanity failed: " + ", ".join(failures))
    return result


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    started = time.perf_counter()
    total = 0
    loss_sum = 0.0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(criterion(logits, targets).item()) * inputs.shape[0]
        correct += int((logits.argmax(1) == targets).sum().item())
        total += int(inputs.shape[0])
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "seconds": elapsed,
        "images_per_second": total / max(elapsed, 1e-12),
    }


def train_epoch(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    started = time.perf_counter()
    loss_sum = 0.0
    correct = 0
    total = 0
    nan = False
    inf = False
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        value = float(loss.item())
        nan = nan or math.isnan(value)
        inf = inf or math.isinf(value)
        loss_sum += value * inputs.shape[0]
        correct += int((logits.argmax(1) == targets).sum().item())
        total += int(inputs.shape[0])
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "seconds": elapsed,
        "images_per_second": total / max(elapsed, 1e-12),
        "nan": nan,
        "inf": inf,
        "iterations": len(loader),
    }


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["online_train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]
    train_acc = [float(row["online_train_accuracy"]) for row in rows]
    test_acc = [float(row["test_accuracy"]) for row in rows]
    learning_rates = [float(row["learning_rate"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, train_loss); axes[0, 0].set(title="Online Train Loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].plot(epochs, test_loss); axes[0, 1].set(title="Test Loss", xlabel="Epoch", ylabel="Loss")
    axes[1, 0].plot(epochs, train_acc, label="online train"); axes[1, 0].plot(epochs, test_acc, label="test"); axes[1, 0].legend(); axes[1, 0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1, 1].plot(epochs, learning_rates); axes[1, 1].set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    figures = [
        ("train_loss.png", train_loss, "Train Loss", "Loss"),
        ("test_loss.png", test_loss, "Test Loss", "Loss"),
        ("accuracy.png", (train_acc, test_acc), "Online Train and Test Accuracy", "Accuracy"),
        ("learning_rate.png", learning_rates, "Learning Rate", "LR"),
    ]
    for filename, values, title, ylabel in figures:
        fig, axis = plt.subplots(figsize=(6, 4))
        if filename == "accuracy.png":
            axis.plot(epochs, values[0], label="online train")
            axis.plot(epochs, values[1], label="test")
            axis.legend()
        else:
            axis.plot(epochs, values)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)


def formal_train(
    data_root: str, output_dir: Path, config: FrozenBaselineConfig,
    device: torch.device, resume: bool = False,
) -> Dict[str, Any]:
    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    data = build_loaders(data_root, config, generator)
    model = make_model(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = output_dir / "checkpoint_last.pt"
    metrics_path = output_dir / "epoch_metrics.csv"
    rows = _read_rows(metrics_path) if resume else []
    start_epoch = 0
    global_step = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        start_epoch, global_step, payload = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, generator, device
        )
        if payload["config_hash"] != config_hash(config):
            raise RuntimeError("Resume checkpoint config hash does not match frozen protocol")
        if len(rows) != start_epoch:
            raise RuntimeError("Metric-row count does not match resume epoch")
    elif checkpoint_path.exists() or metrics_path.exists():
        raise RuntimeError("Formal output already exists; use --resume or a new output directory")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_index in range(start_epoch, config.epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train = train_epoch(model, data["train_aug_loader"], criterion, optimizer, device)
        test = evaluate(model, data["test_loader"], criterion, device)
        scheduler.step()
        global_step += int(train["iterations"])
        row = {
            "epoch": epoch_index + 1,
            "learning_rate": learning_rate,
            "learning_rate_after_step": float(optimizer.param_groups[0]["lr"]),
            "online_train_loss": train["loss"],
            "online_train_accuracy": train["accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "time_epoch_sec": train["seconds"],
            "images_per_second": train["images_per_second"],
            "test_images_per_second": test["images_per_second"],
            "nan": train["nan"],
            "inf": train["inf"],
        }
        rows.append(row)
        _write_rows(metrics_path, rows, CSV_FIELDS)
        _write_rows(output_dir / "train_log.csv", rows, CSV_FIELDS)
        _write_rows(
            output_dir / "lr_curve.csv",
            [{"epoch": r["epoch"], "learning_rate": r["learning_rate"], "learning_rate_after_step": r["learning_rate_after_step"]} for r in rows],
            ["epoch", "learning_rate", "learning_rate_after_step"],
        )
        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, epoch_index + 1,
            global_step, config, generator,
        )
        print(
            f"Epoch {epoch_index + 1:03d}/{config.epochs}: "
            f"lr={learning_rate:.8f} train_loss={train['loss']:.4f} "
            f"train_acc={train['accuracy']:.4f} test_loss={test['loss']:.4f} "
            f"test_acc={test['accuracy']:.4f} time={train['seconds']:.1f}s "
            f"img/s={train['images_per_second']:.1f}",
            flush=True,
        )
        if train["nan"] or train["inf"]:
            raise RuntimeError("NaN/Inf detected; stopping frozen baseline without protocol changes")

    # Required final evaluation loads the epoch-200 checkpoint into a fresh model.
    final_model = make_model(device)
    final_optimizer = make_optimizer(final_model, config)
    final_scheduler = make_scheduler(final_optimizer, config)
    final_generator = torch.Generator().manual_seed(config.seed)
    completed_epoch, final_global_step, payload = load_checkpoint(
        checkpoint_path, final_model, final_optimizer, final_scheduler,
        final_generator, device,
    )
    if completed_epoch != config.epochs:
        raise RuntimeError("Final checkpoint is not epoch 200")
    clean_train = evaluate(final_model, data["train_eval_loader"], criterion, device)
    final_test = evaluate(final_model, data["test_loader"], criterion, device)
    times = [float(row["time_epoch_sec"]) for row in rows]
    training_images_per_second = config.train_size * config.epochs / sum(times)
    final_lr = float(final_optimizer.param_groups[0]["lr"])
    best_diagnostic = max(float(row["test_accuracy"]) for row in rows)
    learning_rates = [float(row["learning_rate"]) for row in rows]
    lr_after = [float(row["learning_rate_after_step"]) for row in rows]
    scheduler_anomaly = not (
        abs(learning_rates[0] - 0.1) < 1e-12
        and all(right <= left + 1e-15 for left, right in zip(learning_rates, learning_rates[1:]))
        and abs(lr_after[-1]) < 1e-12
    )
    final_accuracy = final_test["accuracy"]
    if final_accuracy >= 0.95:
        gate = "PASS_STRONG_BASELINE"
        frozen_label = "FROZEN_STRONG_CNN_BASELINE_V1"
    elif final_accuracy >= 0.94:
        gate = "PASS_PIPELINE_BUT_REVIEW"
        frozen_label = None
    else:
        gate = "FAIL_BASELINE_GATE"
        frozen_label = None
    summary = {
        "completed_epoch": completed_epoch,
        "global_step": final_global_step,
        "clean_train_loss": clean_train["loss"],
        "clean_train_accuracy": clean_train["accuracy"],
        "final_test_loss": final_test["loss"],
        "final_test_accuracy": final_accuracy,
        "generalization_gap_accuracy": clean_train["accuracy"] - final_accuracy,
        "best_observed_test_accuracy_diagnostic_only": best_diagnostic,
        "total_training_time_sec": sum(times),
        "mean_time_per_epoch_sec": statistics.mean(times),
        "median_time_per_epoch_sec": statistics.median(times),
        "training_images_per_second": training_images_per_second,
        "test_inference_images_per_second": final_test["images_per_second"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "final_learning_rate": final_lr,
        "nan": any(str(row["nan"]).lower() == "true" for row in rows),
        "inf": any(str(row["inf"]).lower() == "true" for row in rows),
        "scheduler_anomaly": scheduler_anomaly,
        "baseline_gate": gate,
        "frozen_label": frozen_label,
    }
    write_json(output_dir / "final_metrics.json", summary)
    plot_curves(rows, output_dir)
    write_final_report(output_dir, config, device, data, summary)
    return summary


def write_final_report(output_dir, config, device, data, summary):
    model = make_model(device)
    stats = parameter_statistics(model)
    environment = environment_info(device)
    report = f"""# Strong CNN Baseline Report

## 1. Frozen Protocol

`{json.dumps(config_dict(config), ensure_ascii=False)}`

No SAM, N:M, sparse parametrization, mask, STE, AMP, label smoothing, MixUp, CutMix, AutoAugment, RandAugment, or EMA was used.

## 2. Environment

`{json.dumps(environment, ensure_ascii=False)}`

## 3. Dataset Audit

- CIFAR-10 train: {len(data['train_aug_dataset'])}
- CIFAR-10 test: {len(data['test_dataset'])}
- train augmentation: crop, flip, ToTensor, one 16x16 Cutout, Normalize
- deterministic test and clean train-eval: ToTensor, Normalize

## 4. Architecture Audit

- CIFAR-ResNet18 stage shapes: `{json.dumps(stage_shapes(model, device))}`
- parameter counts: `{json.dumps(stats)}`
- sparse wrappers=0, N:M masks=0, sparse parameters=0
- optimizer is ordinary SGD; no SAM

## 5. Optimization Protocol

- SGD: lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=False
- CrossEntropyLoss on raw logits; label smoothing=0
- CosineAnnealingLR: T_max=200, epoch-level step after each completed epoch
- warmup=0; FP32

## 6. Numerical Sanity

- NaN: {summary['nan']}
- Inf: {summary['inf']}
- scheduler anomaly: {summary['scheduler_anomaly']}
- one-batch, 20-iteration, BN, gradient, parameter-update, and resume-equivalence results are in `sanity_results.json`

## 7. 200-Epoch Training Result

- clean_train_loss: {summary['clean_train_loss']:.8f}
- clean_train_accuracy: {summary['clean_train_accuracy']:.6%}
- final_test_loss: {summary['final_test_loss']:.8f}
- final_test_accuracy: {summary['final_test_accuracy']:.6%}
- generalization_gap_accuracy: {summary['generalization_gap_accuracy']:.6%}
- total_training_time_sec: {summary['total_training_time_sec']:.3f}
- mean_time_per_epoch_sec: {summary['mean_time_per_epoch_sec']:.3f}
- median_time_per_epoch_sec: {summary['median_time_per_epoch_sec']:.3f}
- training_images_per_second: {summary['training_images_per_second']:.3f}
- test_inference_images_per_second: {summary['test_inference_images_per_second']:.3f}
- final_learning_rate: {summary['final_learning_rate']:.12f}
- best_observed_test_accuracy: {summary['best_observed_test_accuracy_diagnostic_only']:.6%} (diagnostic only; not model selection)

## 8. Convergence Analysis

The loss, accuracy, and learning-rate curves are stored in `training_curves.png` and the individual diagnostic figures. Interpretation is limited to convergence health; no method comparison is made in Phase 2A.

## 9. Baseline Gate

**{summary['baseline_gate']}**

{('Frozen label: **' + summary['frozen_label'] + '**') if summary['frozen_label'] else 'The baseline is not automatically frozen and requires review.'}

## 10. Frozen Baseline Fingerprint

- git commit: {git_commit(Path(__file__).resolve().parent)}
- config SHA-256: `{config_hash(config)}`
- checkpoint: `{(output_dir / 'checkpoint_last.pt').resolve()}`
- seed: {config.seed}
"""
    (output_dir / "BASELINE_RESULT_REPORT.md").write_text(report, encoding="utf-8")
    project_report = Path(__file__).resolve().parent / "STRONG_CNN_BASELINE_REPORT.md"
    project_report.write_text(report, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "sanity", "train", "all", "resume-test"), default="all")
    parser.add_argument("--data-root", default="./nm_sam_cifar10/data/cifar10")
    parser.add_argument("--output-dir", default="./nm_sam_cifar10/results/strong_cnn_baseline/dense_sgd_seed0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = FrozenBaselineConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    if args.mode == "resume-test":
        result = resume_equivalence_sanity()
        write_json(output_dir / "resume_sanity.json", result)
        if not result["passed"]:
            raise RuntimeError("Resume sanity failed")
        print(json.dumps(result, indent=2))
        return
    data = None
    if args.mode in ("preflight", "sanity", "all"):
        data, _, _, _, _ = run_preflight(args.data_root, output_dir, config, device)
        print("PRE-FLIGHT PASS", flush=True)
    if args.mode in ("sanity", "all"):
        assert data is not None
        result = run_sanity(data, output_dir, config, device)
        print("SANITY PASS: " + json.dumps(result), flush=True)
    if args.mode in ("train", "all"):
        # Reset all RNGs and instantiate a fresh model inside formal_train.
        summary = formal_train(args.data_root, output_dir, config, device, resume=args.resume)
        print("FINAL: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
