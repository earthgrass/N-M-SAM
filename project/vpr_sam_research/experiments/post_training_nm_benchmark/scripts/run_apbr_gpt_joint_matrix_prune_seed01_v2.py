"""Isolated Seed 0/1 APBR-GPT pruning endpoint; train data only."""
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(PROJECT / "src")]

import run_cifar100_50k_formal_training as base
from baseline_adapters.apbr_gpt_joint import dynamic_joint_obs_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from benchmark.nm import eligible_layers, validate_masks
from vpr_sam.optim.geometry import Geometry

PROTOCOL = "cifar100_50k10k_final_v1"


class CalibrationSubset(Dataset):
    def __init__(self, data, labels, ids, transform):
        self.data, self.labels, self.ids, self.transform = data, labels, list(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_save(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    return base.digest_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--selector", choices=("original_gpt", "group_obs", "dynamic_apbr_gpt",
                                                "dynamic_sam_gpt"), required=True)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing nonempty pruning output")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL or config["formal_calibration"]["samples"] != 1024:
        raise ValueError("Frozen calibration protocol mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if parent.get("epoch") != 200 or parent.get("test_accessed") is not False:
        raise ValueError("Parent is not a frozen no-Test epoch-200 checkpoint")
    initial = torch.load(args.initial_asset, map_location="cpu", weights_only=False)
    if initial.get("seed") != args.seed or initial.get("model_tensor_sha256") != parent.get("identity", {}).get("initial_model_sha256"):
        raise ValueError("Initial cohort mismatch")
    if parent.get("identity", {}).get("initial_checkpoint_sha256") != base.digest_file(args.initial_asset):
        raise ValueError("Parent/initial checkpoint mismatch")

    train_file = args.dataset_root / "extracted/cifar-100-python/train"
    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    ids = np.random.default_rng(config["formal_calibration"]["selection_seed"]).permutation(50_000)[:1024]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"])])
    batches = DataLoader(CalibrationSubset(data, labels, ids.tolist(), transform), batch_size=128,
                         shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device); model.load_state_dict(parent["model"]); model.eval()
    layers = eligible_layers(model)
    adapters = {name: SparseGPTAdapter(layer, compute_dtype=torch.float32) for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _module, values, item=adapters[name]: item.add_batch(values[0]))
             for name, layer in layers.items()]
    calibration_started = time.perf_counter()
    with torch.no_grad():
        for inputs, _targets, _sample_ids in batches:
            model(inputs.to(device, non_blocking=True))
    for hook in hooks: hook.remove()
    torch.cuda.synchronize(); calibration_seconds = time.perf_counter() - calibration_started

    probes = None
    if args.selector in {"dynamic_apbr_gpt", "dynamic_sam_gpt"}:
        geometry_method = "vpr" if args.selector == "dynamic_apbr_gpt" else "sam"
        geometry = Geometry(model.named_parameters(), initial["apbr_geometry"]["config"], geometry_method, args.seed)
        geometry.load_state_dict(parent["stepper"]["geometry"])
        names = [name for name, _shape in geometry.mapping]
        probes = {name: [] for name in names}
        probe_batches = DataLoader(CalibrationSubset(data, labels, ids.tolist(), transform), batch_size=128,
                                   shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
        model.train()
        for inputs, targets, _sample_ids in probe_batches:
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs.to(device, non_blocking=True)),
                                                      targets.to(device, non_blocking=True))
            loss.backward(); epsilon, _stats = geometry.perturbations(0.05)
            for name, value in zip(names, epsilon): probes[name].append(value.detach().clone())
        model.zero_grad(set_to_none=True); model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    pruning_started = time.perf_counter(); masks = {}; rows = []
    with torch.no_grad():
        for name, layer in layers.items():
            hessian = adapters[name].hessians()[0]
            weight = weight_to_matrix(layer.weight.detach())
            if args.selector == "original_gpt":
                result = sparsegpt_matrix(weight, hessian, keep_n=2, m=4, blocksize=128,
                                          percdamp=.01, compute_dtype=torch.float32)
                final_weight, keep = result.weight, result.keep_mask
                row = {"layer": name, "damp": result.damp,
                       "decomposition_seconds": result.decomposition_seconds,
                       "selection_compensation_seconds": result.compensation_seconds}
            else:
                if args.selector == "group_obs":
                    layer_probes = torch.zeros(1, weight.shape[0], weight.shape[1],
                                               device=weight.device, dtype=weight.dtype)
                else:
                    layer_probes = torch.stack(probes[name + ".weight"]).reshape(8, weight.shape[0], -1)
                result = dynamic_joint_obs_matrix(weight, hessian, layer_probes,
                    shuffled_seed=args.shuffle_seed, keep_n=2, m=4, blocksize=128,
                    percdamp=.01, compute_dtype=torch.float32)
                final_weight, keep = result.weight, result.keep_mask
                row = {"layer": name, "damp": result.damp,
                       "decomposition_seconds": result.decomposition_seconds,
                       "selection_compensation_seconds": result.selection_compensation_seconds,
                       "selected_gpt_cost": float(result.selected_gpt_cost.sum()),
                       "selected_apbr_cost": float(result.selected_apbr_cost.sum())}
            layer.weight.copy_(final_weight.reshape_as(layer.weight))
            masks[name + ".weight"] = keep.reshape_as(layer.weight).cpu(); rows.append(row)
            del adapters[name]
    validate_masks(model, masks, n=2, m=4, complete=True)
    finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    if not finite:
        raise FloatingPointError("Non-finite pruned model")
    mask_sha = base.tensor_sha(masks)
    payload = {"schema": 1, "status": "healthy", "protocol_id": PROTOCOL,
        "stage": "raw_hard", "method": args.endpoint, "seed": args.seed, "epoch": 0,
        "selector": args.selector, "shuffle_seed": args.shuffle_seed,
        "model": model_state(model), "masks": masks, "mask_sha256": mask_sha,
        "parent": {"path": str(args.parent), "sha256": base.digest_file(args.parent)},
        "initial_asset_sha256": base.digest_file(args.initial_asset),
        "calibration_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "calibration_samples": 1024, "calibration_seconds": calibration_seconds,
        "pruning_seconds": time.perf_counter() - pruning_started, "layer_audit": rows,
        "strict_2_4": True, "finite": finite, "recovery_epochs": 0,
        "test_accessed": False, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    raw_path = output / "raw_hard.pt"; raw_sha = atomic_save(raw_path, payload)
    base.atomic_json(output / "status.json", {"status": "completed", "protocol_id": PROTOCOL,
        "endpoint": args.endpoint, "raw_hard_sha256": raw_sha, "mask_sha256": mask_sha,
        "strict_2_4": True, "finite": finite, "test_accessed": False})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1]); target.mkdir(parents=True, exist_ok=True)
            base.atomic_json(target / "failure.json", {"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False, "test_accessed": False})
        raise
