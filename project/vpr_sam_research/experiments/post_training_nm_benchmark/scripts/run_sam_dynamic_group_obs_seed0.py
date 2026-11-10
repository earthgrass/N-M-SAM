"""Dense-SAM Seed 0: dynamic four-column Group-OBS, no APBR geometry.

This is an isolated post-training control. The only Final Test happens after
all 19 masks and the final checkpoint have been frozen and hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_geo_obs_seed0 as common
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import matrix_to_weight, weight_to_matrix
from baseline_adapters.geo_dynamic_obs import dynamic_group_obs_prune
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers, validate_masks

PARENT_SHA256 = "3a0716555d33ec92efe8362e24abec9be27625d24ffb6dfb1d51fddb5b91bd13"
PROTOCOL = "cifar100_50k10k_final_v1"
METHOD = "dynamic_group_obs"


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    if common.digest(args.parent) != PARENT_SHA256:
        raise ValueError("Dense-SAM parent SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != PROTOCOL or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != "dense_sam" or identity.get("seed") != 0):
        raise ValueError("Dense-SAM parent identity mismatch")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL:
        raise ValueError("Protocol mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437 or common.digest(archive) != common.ARCHIVE_SHA256
            or common.digest(archive, "md5") != common.ARCHIVE_MD5):
        raise ValueError("Official CIFAR-100 archive mismatch")
    train_path = args.dataset_root / "extracted/cifar-100-python/train"
    train_data, train_labels = common.read_cifar(train_path, 50000)
    ids = np.random.default_rng(common.CALIBRATION_SEED).permutation(50000)[:1024].tolist()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    batches = DataLoader(common.CifarSubset(train_data, train_labels, ids, transform),
                         batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible layer contract mismatch")
    source = {
        "runner": common.digest(Path(__file__)),
        "dynamic_adapter": common.digest(ROOT / "baseline_adapters/geo_dynamic_obs/adapter.py"),
        "hessian_adapter": common.digest(ROOT / "baseline_adapters/sparsegpt/adapter.py"),
        "model": common.digest(ROOT / "baseline_adapters/cifar100_model.py"),
        "nm": common.digest(ROOT / "benchmark/nm.py"),
        "config": common.digest(args.config),
    }
    common.atomic_json(out / "preflight.json", {
        "status": "PASS", "experiment_id": "sam_dynamic_group_obs_seed0_20261102_v1",
        "parent_sha256": PARENT_SHA256, "protocol_id": PROTOCOL,
        "archive_sha256": common.ARCHIVE_SHA256, "train_pickle_sha256": common.digest(train_path),
        "calibration_ids_sha256": ids_sha, "calibration_ids": ids,
        "eligible_layers": list(layers), "eligible_positions": 11157504,
        "expected_deletions": 5578752, "source_sha256": source,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "test_accessed": False,
    })
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del parent
    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _m, values, item=adapter: item.add_batch(values[0])))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    calibration_start = time.perf_counter()
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in batches:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - calibration_start
    for hook in hooks:
        hook.remove()
    if seen != ids:
        raise ValueError("Calibration order mismatch")
    hessians = {name: adapter.hessians()[0] for name, adapter in adapters.items()}
    del adapters
    state = {key: value.clone() for key, value in original.items()}
    masks, rows = {}, []
    peak_allocated = peak_reserved = 0
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).float()
        h = hessians[name]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = dynamic_group_obs_prune(w, h, mode="std_group_obs", blocksize=128, percdamp=.01)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        keep, final = result.keep_mask, result.weight
        if not bool(torch.isfinite(final).all()) or int(torch.count_nonzero(final[~keep])):
            raise FloatingPointError(f"Invalid output at {name}")
        masks[key] = keep.reshape_as(layer.weight).detach().cpu()
        state[key] = matrix_to_weight(final.detach().cpu(), layer.weight.shape)
        delta = final - w
        physical = common.quadratic_cost(delta, h)
        row = {
            "layer_index": index, "layer": name, "rows": w.shape[0], "K": w.shape[1],
            "deleted": int((~keep).sum()), "physical_quadratic_cost": physical,
            "physical_relative_reconstruction_error": physical / common.quadratic_cost(w, h),
            "survivor_change_rms": float(delta[keep].square().mean().sqrt()),
            "decomposition_seconds": result.decomposition_seconds,
            "selection_compensation_seconds": result.selection_compensation_seconds,
            "wall_seconds": seconds, "finite": True, "strict_2of4": True,
        }
        rows.append(row)
        common.atomic_torch(out / "layers" / f"{index:02d}_{name.replace('.', '_')}.pt", {
            "layer": name, "weight": state[key], "mask": masks[key],
            "chosen_pairs": result.chosen_pairs.detach().cpu(),
            "candidate_costs": result.candidate_costs.detach().cpu(), "diagnostic": row,
        })
        common.atomic_json(out / "progress.json", {"completed_layers": index,
                                                   "last_layer": name, "test_accessed": False})
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        print(json.dumps({"event": "layer_complete", "index": index, "layer": name,
                          "test_accessed": False}), flush=True)
        torch.cuda.empty_cache()
    checked = build_cifar100_model()
    checked.load_state_dict(state, strict=True)
    validate_masks(checked, masks, n=2, m=4, complete=True)
    if sum(int((~mask).sum()) for mask in masks.values()) != 5578752:
        raise ValueError("Full-model deletion count mismatch")
    if not all(torch.equal(state[key], value) for key, value in original.items() if key not in masks):
        raise ValueError("Excluded weights or BN buffers changed")
    common.atomic_csv(out / "per_layer.csv", rows)
    mask_sha = common.atomic_torch(out / "mask.pt", masks)
    payload = {"schema": 1, "status": "healthy", "experiment_id": "sam_dynamic_group_obs_seed0_20261102_v1",
               "protocol_id": PROTOCOL, "stage": "post_obs_raw_hard", "method": METHOD,
               "seed": 0, "recovery_epochs": 0, "model": state, "masks": masks,
               "parent_sha256": PARENT_SHA256, "calibration_ids_sha256": ids_sha,
               "source_sha256": source, "test_accessed": False, "gradient_training": False,
               "bn_calibration": False}
    checkpoint_sha = common.atomic_torch(out / "checkpoint.pt", payload)
    common.atomic_json(out / "frozen_pretest.json", {"status": "CHECKPOINT_FROZEN",
                                                  "checkpoint_sha256": checkpoint_sha,
                                                  "mask_sha256": mask_sha, "test_accessed": False})
    # Test pickle is opened only after the complete output is frozen.
    test_data, test_labels = common.read_cifar(args.dataset_root / "extracted/cifar-100-python/test", 10000)
    test_loader = DataLoader(common.CifarSubset(test_data, test_labels, range(10000), transform),
                             batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    if common.digest(out / "checkpoint.pt") != checkpoint_sha:
        raise ValueError("Checkpoint changed after freeze")
    test_start = time.perf_counter()
    accuracy, ce, count = common.evaluate_once(state, test_loader, device)
    result = {"method": METHOD, "accuracy_percent": accuracy, "ce": ce,
              "test_samples": count, "test_evaluations": 1,
              "checkpoint_sha256": checkpoint_sha, "mask_sha256": mask_sha,
              "pruning_compensation_seconds": sum(r["wall_seconds"] for r in rows),
              "test_seconds": time.perf_counter() - test_start}
    common.atomic_json(out / "result.json", result)
    common.atomic_json(out / "status.json", {"status": "COMPLETE", "result": result,
                                          "calibration_seconds": calibration_seconds,
                                          "peak_allocated_bytes": peak_allocated,
                                          "peak_reserved_bytes": peak_reserved,
                                          "test_accessed_only_after_freeze": True, "recovery_epochs": 0})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{common.digest(p)}  {p.relative_to(out).as_posix()}\n" for p in files))
    print(json.dumps({"event": "final_test", **result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        if args.output.exists():
            common.atomic_json(args.output / "failure.json", {"status": "FAILED",
                                                              "traceback": traceback.format_exc()})
        raise
