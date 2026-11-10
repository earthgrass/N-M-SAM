"""Exploratory CPU comparison: SAM-parent GPT vs singleton conditional Group-OBS.

Both branches use the same SAM Seed-0 Dense parent, frozen Train-only
calibration, FP64 OBS, and one fixed Final Test each.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.fixedmask_obs import FixedMaskOBSAdapter
from baseline_adapters.sparsegpt import sparsegpt_matrix
from baseline_adapters.conditional_greedy_obs import conditional_greedy_obs_prune
from benchmark.nm import eligible_layers
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA, cifar, normalized, error, evaluate, write_new

OUTPUT = ROOT / "artifacts/sam_singleton_conditional_seed0_cpu_20261103"
METHODS = ("native_sparsegpt", "singleton_conditional_joint")


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("This initial real-model pilot is CPU-only")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    source_paths = {
        "runner": Path(__file__),
        "singleton_conditional_joint": ROOT / "baseline_adapters/conditional_greedy_obs/adapter.py",
        "native_sparsegpt": ROOT / "baseline_adapters/sparsegpt/adapter.py",
        "calibration_and_evaluation": ROOT / "scripts/run_apbr_gpt_fixedmask_cpu_full.py",
        "model": ROOT / "baseline_adapters/cifar100_model.py",
        "formal_config": ROOT / "configs/cifar100_50k10k_final_v1.yaml",
    }
    source_start = {name: file_sha(path) for name, path in source_paths.items()}
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    sam_root = ROOT / "artifacts/cifar100_50k10k_final_v1/seed0_remote_copy"
    parent_path = sam_root / "dense_sam/final_epoch_200.pt"
    raw_path = sam_root / "methods/sparsegpt/raw_hard.pt"
    parent_sha = file_sha(parent_path)
    if parent_sha != "3a0716555d33ec92efe8362e24abec9be27625d24ffb6dfb1d51fddb5b91bd13":
        raise ValueError("Frozen parent SHA256 mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("epoch") != 200 or parent.get("steps") != 78200 or parent.get("test_accessed") is not False:
        raise ValueError("Dense parent identity mismatch")
    formal_path = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    archive = DATA / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169001437 or file_sha(archive) != "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7":
        raise ValueError("Official CIFAR-100 archive mismatch")
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible-layer contract mismatch")
    train_data, _ = cifar(DATA / "extracted/cifar-100-python/train", 50000)
    ids = np.random.default_rng(20261027).permutation(50000)[:1024].tolist()
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw_sha = file_sha(raw_path)
    if raw_sha != "77c6ab8a4468842ba211dec422db81f14933ea1403b9c9c35883463e321b7bcf":
        raise ValueError("Historical native SparseGPT Raw hash mismatch")
    if raw["parent"]["sha256"] != parent_sha or raw["test_accessed"] is not False:
        raise ValueError("Historical GPT Raw parent/Test identity mismatch")
    if ids != raw["calibration_ids"]:
        raise ValueError("Calibration IDs mismatch")
    mean = torch.tensor(formal["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(formal["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    adapters = {name: FixedMaskOBSAdapter(layer, compute_dtype=torch.float32) for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _module, args, a=adapters[name]: a.add_batch(args[0])) for name, layer in layers.items()]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), 128):
                batch = ids[start:start + 128]
                model(normalized(train_data, batch, mean, std))
                print(f"calibration {start + len(batch)}/1024", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    calibration_seconds = time.perf_counter() - started
    states = {method: {key: value.detach().cpu().clone() for key, value in parent["model"].items()} for method in METHODS}
    masks = {method: {} for method in METHODS}
    rows = []
    for index, (layer_name, layer) in enumerate(layers.items(), 1):
        key = layer_name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).float()
        h = adapters.pop(layer_name).hessians()[0]
        row = {"index": index, "layer": layer_name, "K": w.shape[1], "methods": {}}
        for method in METHODS:
            started = time.perf_counter()
            result = (sparsegpt_matrix(w, h, keep_n=2, m=4, blocksize=128,
                                       percdamp=0.01, compute_dtype=torch.float64)
                      if method == "native_sparsegpt" else
                      conditional_greedy_obs_prune(w, h, blocksize=128, percdamp=0.01))
            elapsed = time.perf_counter() - started
            keep = result.keep_mask.reshape_as(layer.weight).detach().cpu()
            value = result.weight.reshape_as(layer.weight).detach().cpu()
            validate_keep_mask(result.keep_mask, 2, 4)
            if not bool(torch.isfinite(value).all()) or int(torch.count_nonzero(value[~keep])):
                raise RuntimeError(f"Invalid compensated weights: {layer_name}/{method}")
            masks[method][key] = keep
            states[method][key] = value
            row["methods"][method] = {
                "physical_quadratic_error": error(w, result.weight, h),
                "deleted": int((~keep).sum()),
                "decomposition_seconds": result.decomposition_seconds,
                "selection_compensation_seconds": (
                    result.compensation_seconds if method == "native_sparsegpt"
                    else result.selection_compensation_seconds),
                "wall_seconds": elapsed,
            }
        row["position_flips"] = int((masks[METHODS[0]][key] != masks[METHODS[1]][key]).sum())
        row["identical_groups"] = int((masks[METHODS[0]][key].reshape(-1, 4) == masks[METHODS[1]][key].reshape(-1, 4)).all(dim=1).sum())
        rows.append(row)
        write_new(OUTPUT / f"layer_{index:02d}.json", row)
        print(f"layer {index}/19 {layer_name} complete", flush=True)
    for method in METHODS:
        if sum(int((~mask).sum()) for mask in masks[method].values()) != 5578752:
            raise ValueError(f"Deletion count mismatch: {method}")
        for key, value in states[method].items():
            if key not in masks[method] and not torch.equal(value, parent["model"][key]):
                raise ValueError(f"Excluded/BN state changed: {method}/{key}")
    if {name: file_sha(path) for name, path in source_paths.items()} != source_start:
        raise RuntimeError("Source changed during pruning; refusing Final Test")
    frozen = {}
    for method in METHODS:
        path = OUTPUT / f"{method}.pt"
        payload = {"status": "fixed_before_test", "method": method, "seed": 0,
                   "protocol_id": parent["protocol_id"], "parent_sha256": parent_sha,
                   "model": states[method], "masks": masks[method],
                   "mask_sha256": tensor_sha(masks[method]), "calibration_ids": ids,
                   "test_accessed_during_pruning": False, "recovery_epochs": 0,
                   "algorithm_note": ("native_sparsegpt" if method == "native_sparsegpt"
                                      else "singleton_obs_conditional_selection_plus_joint_group_compensation")}
        write_new(path, payload)
        frozen[method] = file_sha(path)
    write_new(OUTPUT / "pre_test_freeze.json", {"checkpoint_sha256": frozen, "calibration_seconds": calibration_seconds})
    print("all outputs frozen; one fixed Test per method", flush=True)
    test_data, test_labels = cifar(DATA / "extracted/cifar-100-python/test", 10000)
    evaluations = {method: evaluate(states[method], test_data, test_labels, mean, std) for method in METHODS}
    summary = {"status": "COMPLETE_CPU_EXPLORATORY_SEED0", "parent_sha256": parent_sha,
               "historical_native_gpt_raw_sha256": raw_sha,
               "checkpoint_sha256": frozen, "mask_sha256": {m: tensor_sha(masks[m]) for m in METHODS},
               "calibration_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
               "source_sha256": source_start,
               "test_pickle_sha256": file_sha(DATA / "extracted/cifar-100-python/test"),
               "calibration_seconds": calibration_seconds, "layer_count": len(rows),
               "position_flips": sum(row["position_flips"] for row in rows),
               "quadratic_error_sum": {m: sum(row["methods"][m]["physical_quadratic_error"] for row in rows) for m in METHODS},
               "pruning_wall_sum": {m: sum(row["methods"][m]["wall_seconds"] for row in rows) for m in METHODS},
               "test": evaluations}
    write_new(OUTPUT / "results.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_sha256",)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if not (OUTPUT / "failure.json").exists():
            write_new(OUTPUT / "failure.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
