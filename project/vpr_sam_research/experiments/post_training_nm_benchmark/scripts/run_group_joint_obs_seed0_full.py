"""Seed-0 full-network Group-OBS/APBR selection experiment.

The runner is deliberately one-shot and serial. It first replays Original
SparseGPT on the current GPU. Historical mask and compensated-weight drift are
cross-environment diagnostics. The hard implementation gate is bitwise
equality between current-GPU native OBS and current-GPU fixed-mask OBS using
the same mask. All checkpoints are frozen before the one-time Final Test
evaluations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_apbr_weights_original_sparsegpt_one as common
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import matrix_to_weight, weight_to_matrix
from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import (
    APBR_SHUFFLE_SEEDS,
    DELETION_PAIRS,
    prepare_obs_system,
    select_group_mask,
    static_group_obs_costs,
)
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from benchmark.nm import eligible_layers, validate_masks


PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "apbr_gpt_group_joint_obs_seed0_full_20261031"
PARENT_SHA256 = "4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82"
ARCHIVE_SIZE = 169_001_437
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
CALIBRATION_COUNT = 1024
CALIBRATION_SEED = 20261027
PERCDAMP = 0.01
BLOCKSIZE = 128
METHODS = (
    "original_sparsegpt",
    "group_obs",
    "apbr_tau_0p005",
    "apbr_tau_0p01",
    "apbr_tau_0p02",
    "shuffled_20261031_tau_0p01",
    "shuffled_20261032_tau_0p01",
    "shuffled_20261033_tau_0p01",
)


class ArrayDataset(Dataset):
    def __init__(self, data, labels, ids, transform):
        self.data, self.labels, self.ids, self.transform = data, labels, list(ids), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def atomic_torch(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return train.digest_file(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def tensor_quantiles(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float().flatten()
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.95, 0.99], device=x.device))
    return {
        "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
        "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
        "p99": float(q[3]), "max": float(x.max()),
    }


def pair_index_from_keep(keep: torch.Tensor) -> torch.Tensor:
    end = keep.shape[1] // 4 * 4
    deleted = (~keep[:, :end].reshape(keep.shape[0], -1, 4))
    pairs = torch.tensor(DELETION_PAIRS, device=keep.device)
    templates = torch.zeros(6, 4, dtype=torch.bool, device=keep.device)
    templates.scatter_(1, pairs, True)
    matches = (deleted[..., None, :] == templates).all(-1)
    if not bool((matches.sum(-1) == 1).all()):
        raise RuntimeError("A keep-mask does not map to one deletion pair")
    return matches.to(torch.int64).argmax(-1)


def local_mse(weight: torch.Tensor, candidate: torch.Tensor, hessian: torch.Tensor) -> float:
    delta = (candidate - weight).float()
    return float(0.5 * torch.einsum("ri,ij,rj->", delta, hessian.float(), delta) / weight.shape[0])


def load_pickle(path: Path, expected: int):
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (expected, 3072) or labels.shape != (expected,):
        raise ValueError(f"CIFAR pickle identity mismatch: {path}")
    target = 500 if expected == 50_000 else 100
    if not bool(np.all(np.bincount(labels, minlength=100) == target)):
        raise ValueError(f"CIFAR label distribution mismatch: {path}")
    return data, labels


def gpu_environment() -> dict:
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if processes:
        raise RuntimeError(f"GPU already has compute processes: {processes}")
    fields = [item.strip() for item in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")]
    prop = torch.cuda.get_device_properties(0)
    return {
        "name": fields[0], "uuid": fields[1], "memory_total_mib": int(fields[2]),
        "memory_free_mib": int(fields[3]), "driver": fields[4],
        "compute_capability": [prop.major, prop.minor], "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def evaluate_once(state, test_loader, device) -> tuple[float, float, int]:
    model = build_cifar100_model().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    correct = count = 0
    loss = 0.0
    with torch.inference_mode():
        for inputs, labels, _ids in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()
    del model
    torch.cuda.empty_cache()
    return correct / count, loss / count, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--historical-native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required")
    disk = shutil.disk_usage(output)
    if disk.free < 15 * 1024**3:
        raise RuntimeError("At least 15 GiB free disk is required")
    gpu = gpu_environment()
    train.seed_all(0)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL:
        raise ValueError("Protocol mismatch")
    parent_sha = train.digest_file(args.parent)
    if parent_sha != PARENT_SHA256:
        raise ValueError("Dense parent SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    train.validate_checkpoint(parent)
    identity = parent.get("identity", {})
    if (parent.get("epoch") != 200 or parent.get("steps") != 78_200
            or identity.get("method") != "apbr_sam" or identity.get("seed") != 0
            or parent.get("test_accessed") is not False):
        raise ValueError("Dense parent identity mismatch")
    historical_sha = train.digest_file(args.historical_native)
    historical = torch.load(args.historical_native, map_location="cpu", weights_only=False)
    if (historical.get("method") != "original_sparsegpt"
            or historical.get("parent", {}).get("sha256") != parent_sha
            or historical.get("test_accessed") is not False):
        raise ValueError("Historical native SparseGPT identity mismatch")

    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != ARCHIVE_SIZE
            or train.digest_file(archive, "md5") != ARCHIVE_MD5
            or train.digest_file(archive) != ARCHIVE_SHA256):
        raise ValueError("Official CIFAR-100 archive mismatch")
    extracted = args.dataset_root / "extracted" / "cifar-100-python"
    train_data, train_labels = load_pickle(extracted / "train", 50_000)
    calibration_ids = np.random.default_rng(CALIBRATION_SEED).permutation(50_000)[:CALIBRATION_COUNT].tolist()
    calibration_sha = hashlib.sha256(np.asarray(calibration_ids, dtype="<i8").tobytes()).hexdigest()
    if calibration_sha != historical.get("calibration_ids_sha256"):
        raise ValueError("Calibration identity mismatch")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    calibration_loader = DataLoader(
        ArrayDataset(train_data, train_labels, calibration_ids, transform), batch_size=128,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )

    device = torch.device("cuda:0")
    dense_model = build_cifar100_model().to(device)
    dense_model.load_state_dict(parent["model"], strict=True)
    dense_model.eval()
    layers = eligible_layers(dense_model)
    if len(layers) != 19:
        raise ValueError(f"Expected 19 eligible layers, found {len(layers)}")
    eligible_positions = sum(layer.weight.numel() for layer in layers.values())
    deletions = sum((layer.weight[0].numel() // 4 * 2) * layer.weight.shape[0] for layer in layers.values())
    if eligible_positions != 11_157_504 or deletions != 5_578_752:
        raise ValueError("Eligible coordinate count mismatch")

    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    expected = [(name, tuple(value.shape)) for name, value in dense_model.named_parameters()]
    if geometry.get("method") != "vpr" or mapping != expected:
        raise ValueError("APBR geometry mapping mismatch")
    logs = {name: value for (name, _shape), value in zip(mapping, geometry["logs"])}
    for name, layer in layers.items():
        value = logs[name + ".weight"]
        if value.dtype != torch.float64 or value.shape != layer.weight.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"APBR geometry tensor mismatch: {name}")
    geometry_sha = train.tensor_sha({name: value for name, value in logs.items()})

    source_hashes = {
        "runner": train.digest_file(Path(__file__)),
        "group_adapter": train.digest_file(ROOT / "baseline_adapters/group_joint_obs/adapter.py"),
        "sparsegpt_adapter": train.digest_file(ROOT / "baseline_adapters/sparsegpt/adapter.py"),
        "fixedmask_adapter": train.digest_file(ROOT / "baseline_adapters/fixedmask_obs/adapter.py"),
        "conv2d_adapter": train.digest_file(ROOT / "baseline_adapters/common/conv2d.py"),
        "nm_contract": train.digest_file(ROOT / "benchmark/nm.py"),
        "config": train.digest_file(args.config),
    }
    preflight = {
        "status": "GPU_PREFLIGHT_PASS", "experiment_id": EXPERIMENT,
        "gpu": gpu, "disk_free_bytes": disk.free, "parent_sha256": parent_sha,
        "historical_native_sha256": historical_sha, "geometry_sha256": geometry_sha,
        "archive_sha256": ARCHIVE_SHA256, "calibration_ids_sha256": calibration_sha,
        "eligible_layers": list(layers), "eligible_positions": eligible_positions,
        "deletions": deletions, "source_hashes": source_hashes, "test_accessed": False,
    }
    train.atomic_json(output / "preflight.json", preflight)

    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, values, item=adapter: item.add_batch(values[0])
        ))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    calibration_started = time.perf_counter()
    seen_ids = []
    with torch.inference_mode():
        for inputs, _labels, ids in calibration_loader:
            dense_model(inputs.to(device, non_blocking=True))
            seen_ids.extend(map(int, ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - calibration_started
    for hook in hooks:
        hook.remove()
    if seen_ids != calibration_ids:
        raise ValueError("Calibration order mismatch")
    hessians = {name: adapter.hessians()[0].detach().clone() for name, adapter in adapters.items()}
    del adapters

    # Mandatory all-layer replay gate before any alternative selector runs.
    replay_rows, native_cache = [], {}
    for name, layer in layers.items():
        key = name + ".weight"
        weight = weight_to_matrix(layer.weight.detach())
        hessian = hessians[name]
        torch.cuda.synchronize()
        native_started = time.perf_counter()
        native = sparsegpt_matrix(
            weight, hessian, keep_n=2, m=4, blocksize=BLOCKSIZE,
            percdamp=PERCDAMP, compute_dtype=torch.float32,
        )
        torch.cuda.synchronize()
        native_wall_seconds = time.perf_counter() - native_started
        expected_mask = weight_to_matrix(historical["masks"][key].to(device))
        expected_weight = weight_to_matrix(historical["model"][key].to(device))
        historical_mask_bitwise = torch.equal(native.keep_mask, expected_mask)
        historical_mask_flips = int(torch.count_nonzero(native.keep_mask != expected_mask))
        historical_mask_flip_rate = historical_mask_flips / native.keep_mask.numel()
        historical_difference = native.weight - expected_weight
        historical_weight_bitwise = torch.equal(native.weight, expected_weight)
        historical_max_abs = float(historical_difference.abs().max())
        historical_relative_l2 = float(
            historical_difference.norm() / expected_weight.norm()
        )
        fixed = fixedmask_obs_matrix(
            weight, hessian, native.keep_mask, keep_n=2, m=4, blocksize=BLOCKSIZE,
            percdamp=PERCDAMP, compute_dtype=torch.float32,
        )
        if not torch.equal(fixed.weight, native.weight):
            difference = fixed.weight - native.weight
            raise RuntimeError(
                f"Fixed-mask provider replay mismatch: {name}; "
                f"max_abs={float(difference.abs().max())} rel={float(difference.norm()/native.weight.norm())}"
            )
        replay_rows.append({
            "layer": name,
            "mask_equal_historical": historical_mask_bitwise,
            "historical_mask_flip_count": historical_mask_flips,
            "historical_mask_flip_rate": historical_mask_flip_rate,
            "native_weight_bitwise_historical": historical_weight_bitwise,
            "historical_max_abs_error": historical_max_abs,
            "historical_relative_l2_error": historical_relative_l2,
            "fixed_provider_bitwise_native": True,
            "same_gpu_max_abs_error": 0.0,
            "same_gpu_relative_l2_error": 0.0,
        })
        native_cache[name] = (
            native.weight.detach().clone(), native.keep_mask.detach().clone(),
            native.decomposition_seconds, native.compensation_seconds, native_wall_seconds,
        )
    atomic_csv(output / "original_replay_per_layer.csv", replay_rows)
    train.atomic_json(output / "original_replay_gate.json", {
        "status": "SAME_GPU_NATIVE_VS_FIXEDMASK_REPLAY_PASS", "layers": 19,
        "historical_mask_bitwise_all": all(
            row["mask_equal_historical"] for row in replay_rows
        ),
        "historical_mask_mismatch_is_diagnostic_only": True,
        "historical_weight_bitwise_all": all(
            row["native_weight_bitwise_historical"] for row in replay_rows
        ),
        "same_gpu_native_vs_fixedmask_weight_bitwise_all": True,
        "historical_weight_mismatch_is_diagnostic_only": True,
        "parent_sha256": parent_sha, "historical_native_sha256": historical_sha,
        "test_accessed": False,
    })

    base_state = state_cpu(dense_model)
    states = {method: {key: value.clone() for key, value in base_state.items()} for method in METHODS}
    masks = {method: {} for method in METHODS}
    layer_rows, candidate_summary = [], []
    total_started = time.perf_counter()
    method_seconds = {method: 0.0 for method in METHODS}
    peak_allocated = peak_reserved = 0

    for layer_index, (name, layer) in enumerate(layers.items(), start=1):
        key = name + ".weight"
        weight = weight_to_matrix(layer.weight.detach())
        hessian = hessians[name]
        log_radius = weight_to_matrix(logs[key].to(device))
        torch.cuda.reset_peak_memory_stats(device)
        layer_started = time.perf_counter()
        system = prepare_obs_system(hessian, percdamp=PERCDAMP, compute_dtype=torch.float32)
        group_started = time.perf_counter()
        group = select_group_mask(weight, system)
        group_mask_seconds = time.perf_counter() - group_started
        true_apbr = {}
        selections = {"group_obs": group}
        for tau, method in ((0.005, "apbr_tau_0p005"), (0.01, "apbr_tau_0p01"), (0.02, "apbr_tau_0p02")):
            started = time.perf_counter()
            selections[method] = select_group_mask(weight, system, log_radius=log_radius, tau=tau)
            true_apbr[method] = time.perf_counter() - started
        for seed in APBR_SHUFFLE_SEEDS:
            method = f"shuffled_{seed}_tau_0p01"
            started = time.perf_counter()
            selections[method] = select_group_mask(
                weight, system, log_radius=log_radius, tau=0.01, shuffled_seed=seed
            )
            true_apbr[method] = time.perf_counter() - started

        native_weight, native_mask, native_decomposition, native_compensation, native_wall = native_cache[name]
        all_outputs = {
            "original_sparsegpt": (
                native_weight, native_mask, native_decomposition, native_compensation, native_wall
            )
        }
        for method, selection in selections.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            compensated = fixedmask_obs_matrix(
                weight, hessian, selection.keep_mask, keep_n=2, m=4,
                blocksize=BLOCKSIZE, percdamp=PERCDAMP, compute_dtype=torch.float32,
            )
            torch.cuda.synchronize()
            compensation_seconds = time.perf_counter() - started
            all_outputs[method] = (
                compensated.weight, compensated.keep_mask,
                compensated.decomposition_seconds, compensated.compensation_seconds,
                compensation_seconds,
            )

        gpt_costs = group.gpt_costs
        true_apbr_costs = selections["apbr_tau_0p01"].apbr_costs
        native_pair = pair_index_from_keep(native_mask)
        method_pair = {"original_sparsegpt": native_pair, **{
            method: selection.selected_pair_index for method, selection in selections.items()
        }}
        candidate_payload = {
            "schema": 1, "layer": name, "pair_order": DELETION_PAIRS,
            "gpt_costs": gpt_costs.detach().cpu(),
            "apbr_costs": true_apbr_costs.detach().cpu(),
            "shuffled_apbr_costs": {
                str(seed): selections[f"shuffled_{seed}_tau_0p01"].apbr_costs.detach().cpu()
                for seed in APBR_SHUFFLE_SEEDS
            },
            "selected_pair_index": {method: value.detach().cpu() for method, value in method_pair.items()},
            "candidate_count": {
                method: selection.candidate_count.detach().cpu()
                for method, selection in selections.items() if method != "group_obs"
            },
        }
        candidate_path = output / "candidate_costs" / f"{layer_index:02d}_{name.replace('.', '_')}.pt"
        candidate_sha = atomic_torch(candidate_path, candidate_payload)

        group_apbr_selected = true_apbr_costs.gather(-1, group.selected_pair_index[..., None]).squeeze(-1)
        original_gpt_selected = gpt_costs.gather(-1, native_pair[..., None]).squeeze(-1)
        original_apbr_selected = true_apbr_costs.gather(-1, native_pair[..., None]).squeeze(-1)
        for method, (final_weight, keep, decomposition_seconds, compensation_seconds, method_wall_seconds) in all_outputs.items():
            if not bool(torch.isfinite(final_weight).all()) or int(torch.count_nonzero(final_weight[~keep])):
                raise FloatingPointError(f"Invalid compensated output: {method}/{name}")
            matrix_mask = keep
            shape_mask = matrix_mask.reshape_as(layer.weight).detach().cpu()
            masks[method][key] = shape_mask
            states[method][key] = matrix_to_weight(final_weight.detach().cpu(), layer.weight.shape)
            selected_pair = method_pair[method]
            selected_gpt = gpt_costs.gather(-1, selected_pair[..., None]).squeeze(-1)
            selected_apbr = true_apbr_costs.gather(-1, selected_pair[..., None]).squeeze(-1)
            raw_weight = weight * matrix_mask
            mask_flip = int((matrix_mask != native_mask).sum())
            update = final_weight - weight
            row = {
                "layer_index": layer_index, "layer": name, "method": method,
                "weight_shape": list(layer.weight.shape), "K": weight.shape[1],
                "mask_flip_count_vs_original": mask_flip,
                "mask_flip_rate_vs_original": mask_flip / matrix_mask.numel(),
                "gpt_cost_mean": float(selected_gpt.float().mean()),
                "gpt_cost_delta_mean_vs_group_obs": float((selected_gpt - group.minimum_gpt_cost).float().mean()),
                "apbr_cost_mean": float(selected_apbr.float().mean()),
                "apbr_cost_delta_mean_vs_group_obs": float((selected_apbr - group_apbr_selected).float().mean()),
                "mask_only_layer_mse": local_mse(weight, raw_weight, hessian),
                "obs_layer_mse": local_mse(weight, final_weight, hessian),
                "survivor_update_rms": float(update[matrix_mask].float().square().mean().sqrt()),
                "survivor_update_max_abs": float(update[matrix_mask].abs().max()),
                "decomposition_seconds": decomposition_seconds,
                "compensation_seconds": compensation_seconds,
                "method_layer_wall_seconds": method_wall_seconds,
                "finite": True, "deleted_zero": True, "strict_2of4": True,
            }
            if method in selections:
                selection = selections[method]
                row.update({
                    "mask_generation_seconds": group_mask_seconds if method == "group_obs" else true_apbr[method],
                    "candidate_count_mean": float(selection.candidate_count.float().mean()),
                    "candidate_count_max": int(selection.candidate_count.max()),
                    **{f"relative_gpt_increase_{k}": v for k, v in tensor_quantiles(selection.relative_gpt_increase).items()},
                })
            else:
                row.update({
                    "mask_generation_seconds": 0.0,
                    "candidate_count_mean": None, "candidate_count_max": None,
                    **{f"relative_gpt_increase_{k}": v for k, v in tensor_quantiles(
                        torch.where(group.minimum_gpt_cost > 0,
                                    (original_gpt_selected - group.minimum_gpt_cost) / group.minimum_gpt_cost,
                                    torch.zeros_like(group.minimum_gpt_cost))
                    ).items()},
                })
            layer_rows.append(row)
            method_seconds[method] += (
                row["mask_generation_seconds"] + decomposition_seconds + compensation_seconds
            )

        for candidate, pair in enumerate(DELETION_PAIRS):
            candidate_summary.append({
                "layer": name, "candidate_index": candidate,
                "deleted_local_i": pair[0], "deleted_local_j": pair[1],
                "gpt_cost_mean": float(gpt_costs[..., candidate].float().mean()),
                "gpt_cost_std": float(gpt_costs[..., candidate].float().std(unbiased=False)),
                "apbr_cost_mean": float(true_apbr_costs[..., candidate].float().mean()),
                "candidate_tensor_file": str(candidate_path.relative_to(output)),
                "candidate_tensor_sha256": candidate_sha,
            })

        for method in METHODS:
            partial_model = build_cifar100_model()
            partial_model.load_state_dict(states[method], strict=True)
            validate_masks(partial_model, masks[method], n=2, m=4, complete=(layer_index == 19)) if layer_index == 19 else None
        progress = {
            "schema": 1, "status": "healthy_partial", "experiment_id": EXPERIMENT,
            "completed_layers": list(layers)[:layer_index], "states": states, "masks": masks,
            "test_accessed": False, "automatic_retry": False,
        }
        atomic_torch(output / "progress.pt", progress)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        torch.cuda.synchronize()
        print(json.dumps({
            "event": "layer_complete", "layer_index": layer_index, "layer": name,
            "seconds": time.perf_counter() - layer_started,
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device), "test_accessed": False,
        }), flush=True)
        del system, selections, all_outputs, candidate_payload
        torch.cuda.empty_cache()

    for method in METHODS:
        check_model = build_cifar100_model()
        check_model.load_state_dict(states[method], strict=True)
        validate_masks(check_model, masks[method], n=2, m=4, complete=True)
    atomic_csv(output / "metrics_per_layer.csv", layer_rows)
    atomic_csv(output / "candidate_costs.csv", candidate_summary)

    checkpoint_records = {}
    for method in METHODS:
        method_dir = output / "methods" / method
        mask_sha = train.tensor_sha(masks[method])
        mask_file_sha = atomic_torch(method_dir / "mask.pt", masks[method])
        payload = {
            "schema": 1, "status": "healthy", "protocol_id": PROTOCOL,
            "experiment_id": EXPERIMENT, "stage": "post_obs_raw_hard", "method": method,
            "seed": 0, "epoch": 0, "recovery_epochs": 0, "model": states[method],
            "model_sha256": train.tensor_sha(states[method]), "masks": masks[method],
            "mask_sha256": mask_sha, "parent_sha256": parent_sha,
            "geometry_sha256": geometry_sha, "calibration_ids_sha256": calibration_sha,
            "test_accessed": False, "gradient_training": False, "bn_calibration": False,
            "source_hashes": source_hashes,
        }
        checkpoint_sha = atomic_torch(method_dir / "checkpoint.pt", payload)
        checkpoint_records[method] = {
            "checkpoint_sha256": checkpoint_sha, "mask_file_sha256": mask_file_sha,
            "mask_tensor_sha256": mask_sha, "model_tensor_sha256": payload["model_sha256"],
        }
    train.atomic_json(output / "frozen_pretest_manifest.json", {
        "status": "ALL_METHODS_FROZEN_PRETEST", "methods": checkpoint_records,
        "test_accessed": False, "selection_frozen": True,
    })

    # Test content is opened only after all eight checkpoints are frozen.
    test_data, test_labels = load_pickle(extracted / "test", 10_000)
    test_loader = DataLoader(
        ArrayDataset(test_data, test_labels, range(10_000), transform), batch_size=128,
        shuffle=False, drop_last=False, num_workers=2, pin_memory=True,
    )
    results = []
    for method in METHODS:
        started = time.perf_counter()
        accuracy, ce, count = evaluate_once(states[method], test_loader, device)
        seconds = time.perf_counter() - started
        checkpoint_path = output / "methods" / method / "checkpoint.pt"
        mask_path = output / "methods" / method / "mask.pt"
        if train.digest_file(checkpoint_path) != checkpoint_records[method]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after freeze: {method}")
        results.append({
            "method": method, "raw_hard_accuracy": accuracy, "raw_hard_ce": ce,
            "test_samples": count, "test_evaluations": 1,
            "test_seconds": seconds, "pruning_compensation_seconds": method_seconds[method],
            "checkpoint_sha256": checkpoint_records[method]["checkpoint_sha256"],
            "mask_file_sha256": train.digest_file(mask_path),
        })
        print(json.dumps({"event": "final_test", **results[-1]}), flush=True)
    atomic_csv(output / "metrics_per_method.csv", results)
    train.atomic_json(output / "status.json", {
        "status": "COMPLETE", "experiment_id": EXPERIMENT, "seed": 0,
        "methods": list(METHODS), "test_accessed": True,
        "test_accessed_only_after_all_checkpoints_frozen": True,
        "test_evaluations_per_method": 1, "automatic_retry": False,
        "calibration_seconds": calibration_seconds,
        "total_experiment_seconds": time.perf_counter() - total_started,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "gpu": gpu, "source_hashes": source_hashes,
    })
    print(json.dumps({"event": "complete", "methods": len(METHODS)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            failure = directory / "failure.json"
            if not failure.exists():
                train.atomic_json(failure, {
                    "status": "FAILED", "type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "automatic_retry": False,
                })
        raise
