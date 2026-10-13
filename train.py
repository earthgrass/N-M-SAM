"""Unified experiment runner assembled from model/optimizer/sparsity/step components."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nm_sam_cifar10.data.cifar10 import build_cifar10, build_transforms
    from nm_sam_cifar10.methods import METHOD_NAMES, get_method_spec
    from nm_sam_cifar10.metrics.efficiency import compute_efficiency
    from nm_sam_cifar10.metrics.logger import CSVLogger, EpochRecord, FileLogger, JSONConfigLogger
    from nm_sam_cifar10.metrics.sharpness import compute_sharpness
    from nm_sam_cifar10.metrics.sanity import check_block_sums, check_effective_weight_zeros, check_mask_stability_across_sam_step
    from nm_sam_cifar10.metrics.sparsity_metrics import collect_sparsity_metrics
    from nm_sam_cifar10.metrics.task_metrics import evaluate_task
    from nm_sam_cifar10.models import build_model
    from nm_sam_cifar10.models.audit import write_architecture_audit
    from nm_sam_cifar10.optim.optimizer_factory import base_optimizer, build_optimizer
    from nm_sam_cifar10.sparsity import SparsityConfig
    from nm_sam_cifar10.sparsity.mask_manager import MaskManager, SPARSE_SCOPES, plan_sparse_layers
    from nm_sam_cifar10.sparsity.parametrization import make_param_mask_fn
    from nm_sam_cifar10.trainers import sam_step, sgd_step
    from nm_sam_cifar10.utils.seed import get_device, set_seed
else:
    from .data.cifar10 import build_cifar10, build_transforms
    from .methods import METHOD_NAMES, get_method_spec
    from .metrics.efficiency import compute_efficiency
    from .metrics.logger import CSVLogger, EpochRecord, FileLogger, JSONConfigLogger
    from .metrics.sharpness import compute_sharpness
    from .metrics.sanity import check_block_sums, check_effective_weight_zeros, check_mask_stability_across_sam_step
    from .metrics.sparsity_metrics import collect_sparsity_metrics
    from .metrics.task_metrics import evaluate_task
    from .models import build_model
    from .models.audit import write_architecture_audit
    from .optim.optimizer_factory import base_optimizer, build_optimizer
    from .sparsity import SparsityConfig
    from .sparsity.mask_manager import MaskManager, SPARSE_SCOPES, plan_sparse_layers
    from .sparsity.parametrization import make_param_mask_fn
    from .trainers import sam_step, sgd_step
    from .utils.seed import get_device, set_seed


METHODS = METHOD_NAMES


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N:M sparsity + SAM on CIFAR-10")
    parser.add_argument("--method", choices=METHODS, default="dense_sgd")
    parser.add_argument("--model", default="resnet18_cifar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--sparse-scope", choices=SPARSE_SCOPES, default="phase1")
    parser.add_argument("--mask-update-interval", type=int, default=100)
    parser.add_argument("--sharpness-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default="./nm_sam_cifar10/data/cifar10")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subset-train", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--eval-subset", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--milestones", type=int, nargs="*", default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--fake-train-size", type=int, default=2000)
    parser.add_argument("--fake-test-size", type=int, default=500)
    return parser.parse_args(argv)


def method_profile(method: str) -> Dict[str, Any]:
    spec = get_method_spec(method)
    return {
        "is_sparse": spec.is_sparse,
        "optimizer": spec.optimizer,
        "mask_score": spec.mask_score,
        "training_step": spec.training_step,
    }


def evaluate(model, loader, criterion, device):
    metrics = evaluate_task(model, loader, criterion, device)
    return metrics.loss, metrics.accuracy


def _dense_layer_report(model: nn.Module, path: Path, n: int, m: int, scope: str) -> None:
    decisions = plan_sparse_layers(model, scope, n, m)
    lines = [
        "N:M SPARSE LAYER REPORT",
        "sparsity: disabled (dense method)",
        f"configured reference N:M: {n}:{m}",
        "actual overall weight sparsity: 0.000000",
        "total N:M blocks: 0",
        "",
        "layer\tstatus\ttype\tshape\treason",
    ]
    for decision in decisions:
        lines.append(f"{decision.name}\tdense\t{decision.layer_type}\t{decision.weight_shape}\tdense method")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sharpness_probe(model, dataset, criterion, device, args, mask_fn=None):
    count = min(args.eval_subset, len(dataset))
    if count <= 0:
        return None
    loader = DataLoader(
        Subset(dataset, list(range(count))), batch_size=min(count, 256),
        shuffle=False, num_workers=0,
    )
    try:
        inputs, targets = next(iter(loader))
    except StopIteration:
        return None
    return compute_sharpness(
        model, inputs, targets, criterion, args.rho, device,
        mask_fn=mask_fn, adaptive=args.adaptive,
    )


SUMMARY_FIELDS = (
    "method", "train_loss", "train_accuracy", "test_loss", "test_accuracy",
    "best_test_accuracy", "final_test_accuracy", "generalization_gap",
    "center_loss", "worst_case_loss", "sharpness", "epsilon_norm",
    "epsilon_ratio", "configured_sparsity", "actual_sparsity",
    "bad_block_count", "total_nm_blocks", "mask_change_ratio",
    "mask_stability", "time_per_epoch", "total_time", "iterations_per_second",
    "images_per_second", "sam_overhead_ratio", "software_wallclock_ratio", "nan", "inf",
)


def _write_summary_core(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: summary.get(field, "NA") for field in SUMMARY_FIELDS})


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    spec = get_method_spec(args.method)
    if args.output_dir is None:
        args.output_dir = f"./nm_sam_cifar10/results/{args.method}_seed{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(prefer_cuda=True) if args.device == "auto" else torch.device(args.device)
    project_dir = Path(__file__).resolve().parent
    model = build_model(args.model, num_classes=10)
    if args.model.lower() in ("resnet18_cifar", "resnet18", "resnet18-cifar"):
        write_architecture_audit(
            model, project_dir / "CNN_ARCHITECTURE_AUDIT.md",
            sparse_scope=args.sparse_scope, n=args.n, m=args.m,
        )

    mask_manager: Optional[MaskManager] = None
    if spec.is_sparse:
        sparsity_cfg = SparsityConfig(n=args.n, m=args.m, enabled=True, score=spec.mask_score or "magnitude")
        mask_manager = MaskManager(model, sparsity_cfg, scope=args.sparse_scope)
        mask_manager.write_report(output_dir / "SPARSE_LAYER_REPORT.txt")
    else:
        _dense_layer_report(model, output_dir / "SPARSE_LAYER_REPORT.txt", args.n, args.m, args.sparse_scope)
    model = model.to(device)
    param_mask_fn = make_param_mask_fn(model)

    config = vars(args).copy()
    config["method_components"] = {
        "optimizer": spec.optimizer, "sparsity": spec.sparsity,
        "mask_score": spec.mask_score, "training_step": spec.training_step,
    }
    config["augmentation"] = {
        "train": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip", "ToTensor", "Cutout(n_holes=1,length=16)", "Normalize"],
        "test": ["ToTensor", "Normalize"],
    }
    config_logger = JSONConfigLogger(str(output_dir / "config.json"), config)
    config_logger.save_config()
    file_logger = FileLogger(str(output_dir / "train.log"))
    csv_logger = CSVLogger(str(output_dir / "metrics.csv"))
    file_logger.write(f"Method components: {json.dumps(config['method_components'])}")
    file_logger.write(f"Device: {device}")

    train_transform, test_transform = build_transforms()
    train_set = build_cifar10(
        args.data_root, True, train_transform, args.use_fake_data,
        args.fake_train_size, args.fake_test_size,
    )
    test_set = build_cifar10(
        args.data_root, False, test_transform, args.use_fake_data,
        args.fake_train_size, args.fake_test_size,
    )
    if args.subset_train:
        train_set = Subset(train_set, list(range(min(args.subset_train, len(train_set)))))
    if args.subset_test:
        test_set = Subset(test_set, list(range(min(args.subset_test, len(test_set)))))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    file_logger.write(f"Train set: {len(train_set)}; test set: {len(test_set)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, spec.optimizer, args)
    milestones = args.milestones or sorted(set([int(0.5 * args.epochs), int(0.75 * args.epochs)]))
    milestones = [milestone for milestone in milestones if milestone > 0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(base_optimizer(optimizer), milestones=milestones, gamma=args.gamma)

    global_step = 0
    total_time = 0.0
    best_test_accuracy = 0.0
    last_mask_change = 0.0
    nan_seen = False
    inf_seen = False
    final: Dict[str, Any] = {}

    for epoch in range(args.epochs):
        started = time.perf_counter()
        loss_sum = 0.0
        correct_weighted = 0.0
        samples_seen = 0
        iterations = 0
        last_step_info: Dict[str, float] = {}

        for inputs, targets in train_loader:
            if spec.training_step == "sgd":
                result = sgd_step(model, optimizer, criterion, inputs, targets, device)
            else:
                result = sam_step(
                    model, optimizer, criterion, inputs, targets, device,
                    param_mask_fn=param_mask_fn if spec.is_sparse else None,
                    mask_score=spec.mask_score,
                )
                if result.first_input_data_ptr != result.second_input_data_ptr:
                    raise RuntimeError("SAM passes did not reuse the same augmented input tensor")
                if not check_mask_stability_across_sam_step(result.pre_masks, result.post_masks):
                    raise RuntimeError("Mask changed between the two SAM forwards")
                last_step_info = result.info

            if spec.is_sparse and args.mask_update_interval > 0 and (global_step + 1) % args.mask_update_interval == 0:
                assert mask_manager is not None
                last_mask_change = mask_manager.recompute(spec.mask_score or "magnitude")

            loss_sum += result.loss * result.batch_size
            correct_weighted += result.accuracy * result.batch_size
            samples_seen += result.batch_size
            iterations += 1
            global_step += 1
            nan_seen = nan_seen or math.isnan(result.loss)
            inf_seen = inf_seen or math.isinf(result.loss)

        scheduler.step()
        epoch_time = time.perf_counter() - started
        total_time += epoch_time
        train_loss = loss_sum / max(samples_seen, 1)
        train_accuracy = correct_weighted / max(samples_seen, 1)
        test_metrics = evaluate_task(model, test_loader, criterion, device)
        best_test_accuracy = max(best_test_accuracy, test_metrics.accuracy)
        generalization_gap = train_accuracy - test_metrics.accuracy

        center_loss = worst_case_loss = sharpness = None
        probe_info: Dict[str, float] = {}
        if args.sharpness_interval > 0 and (epoch % args.sharpness_interval == 0 or epoch == args.epochs - 1):
            probe = _sharpness_probe(
                model, train_set, criterion, device, args,
                param_mask_fn if spec.is_sparse else None,
            )
            if probe is not None:
                center_loss, worst_case_loss, sharpness, probe_info = probe

        sparsity = collect_sparsity_metrics(model, args.n, args.m)
        if sparsity.bad_block_count:
            raise RuntimeError(f"Detected {sparsity.bad_block_count} invalid N:M blocks")
        if spec.is_sparse:
            assert check_block_sums(model, args.n, args.m)
            assert check_effective_weight_zeros(model)
        efficiency = compute_efficiency(epoch_time, total_time, iterations, samples_seen)
        layer_changes = mask_manager.layer_change_ratios() if mask_manager else {}
        epsilon_norm = probe_info.get("eps_norm", last_step_info.get("eps_norm"))
        epsilon_ratio = probe_info.get("eps_norm_over_rho", last_step_info.get("eps_norm_over_rho"))
        record = EpochRecord(
            epoch=epoch, train_loss=train_loss, train_acc=train_accuracy,
            test_loss=test_metrics.loss, test_acc=test_metrics.accuracy,
            epoch_time=epoch_time, cumulative_time=total_time,
            generalization_gap=generalization_gap, center_loss=center_loss,
            sharpness=sharpness, worst_case_loss=worst_case_loss,
            mask_change_ratio=last_mask_change if spec.is_sparse else None,
            active_weight_ratio=(1.0 - sparsity.actual_sparsity) if spec.is_sparse else None,
            layer_sparsity=json.dumps(sparsity.layer_sparsity) if spec.is_sparse else None,
            configured_sparsity=sparsity.configured_sparsity,
            actual_sparsity=sparsity.actual_sparsity,
            total_nm_blocks=sparsity.total_blocks,
            bad_block_count=sparsity.bad_block_count,
            layer_mask_change_ratio=json.dumps(layer_changes) if spec.is_sparse else None,
            mask_stability=(1.0 - last_mask_change) if spec.is_sparse else None,
            iterations_per_second=efficiency.iterations_per_second,
            images_per_second=efficiency.images_per_second,
            grad1_norm=last_step_info.get("g_norm"), grad2_norm=probe_info.get("g_norm"),
            epsilon_norm=epsilon_norm, epsilon_norm_over_rho=epsilon_ratio,
            sam_skipped=bool(last_step_info.get("skipped", False)) if spec.is_sam else None,
            max_abs_logit=probe_info.get("max_abs_logit"),
            nan_detected=nan_seen, inf_detected=inf_seen,
        )
        csv_logger.append(record)
        csv_logger.flush()

        checkpoint = {"model": model.state_dict(), "epoch": epoch, "test_acc": test_metrics.accuracy}
        torch.save(checkpoint, output_dir / "last.pt")
        if test_metrics.accuracy >= best_test_accuracy - 1e-12:
            torch.save(checkpoint, output_dir / "best.pt")
        file_logger.write(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.accuracy:.4f} "
            f"center={center_loss} sharpness={sharpness} eps/rho={epsilon_ratio} time={epoch_time:.2f}s"
        )
        final = {
            "method": spec.name, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "test_loss": test_metrics.loss, "test_accuracy": test_metrics.accuracy,
            "best_test_accuracy": best_test_accuracy, "final_test_accuracy": test_metrics.accuracy,
            "generalization_gap": generalization_gap, "center_loss": center_loss,
            "worst_case_loss": worst_case_loss, "sharpness": sharpness,
            "epsilon_norm": epsilon_norm, "epsilon_ratio": epsilon_ratio,
            "configured_sparsity": sparsity.configured_sparsity,
            "actual_sparsity": sparsity.actual_sparsity,
            "bad_block_count": sparsity.bad_block_count, "total_nm_blocks": sparsity.total_blocks,
            "mask_change_ratio": last_mask_change if spec.is_sparse else "NA",
            "mask_stability": (1.0 - last_mask_change) if spec.is_sparse else "NA",
            "time_per_epoch": total_time / (epoch + 1), "total_time": total_time,
            "iterations_per_second": efficiency.iterations_per_second,
            "images_per_second": efficiency.images_per_second,
            "sam_overhead_ratio": "NA", "software_wallclock_ratio": "NA",
            "nan": nan_seen, "inf": inf_seen,
        }

    _write_summary_core(output_dir / "summary_core.csv", final)
    config_logger.append_summary(final)
    file_logger.write(f"DONE: {json.dumps(final, default=str)}")
    return final


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
