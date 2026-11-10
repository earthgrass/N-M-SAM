"""One fixed Final Test of four predeclared APBR near-optimal OBS full-model arms.

Exploratory APBR-VPR Seed 0 only. Calibration uses the historical 1024 Train
images. No Test file is opened until all 19-layer outputs are checked, saved,
and hashed. No training, BN calibration, or hyperparameter search is performed.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.apbr_nearoptimal_conditional_obs import apbr_nearoptimal_conditional_obs_prune
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers
from scripts.audit_apbr_gpt_mask_match_local import PAIRS, file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA, cifar, normalized, evaluate


OUTPUT = ROOT / "artifacts/apbr_nearoptimal_conditional_seed0_full_cpu_20261103_v1"
PARENT_SHA256 = "4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82"
ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
METHODS = ("physical", "unit", "true", "shuffled")
CALIBRATION_SEED = 20261027
SHUFFLE_SEED = 20261103
TAU = 0.01


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pt":
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def physical_error(w: torch.Tensor, candidate: torch.Tensor, h: torch.Tensor) -> float:
    delta = (w - candidate).double()
    return float(.5 * torch.einsum("ri,ij,rj->", delta, h.double(), delta))


def current_sources(formal_path: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__),
        "selector": ROOT / "baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py",
        "hessian_adapter": ROOT / "baseline_adapters/sparsegpt/adapter.py",
        "model": ROOT / "baseline_adapters/cifar100_model.py",
        "eligibility": ROOT / "benchmark/nm.py",
        "formal_config": formal_path,
    }
    return {name: file_sha(path) for name, path in files.items()}


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("This isolated full-model exploration is CPU-only")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    paths = PAIRS["apbr_vpr_seed0"]
    formal_path = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    source_sha = current_sources(formal_path)
    if file_sha(paths["parent"]) != PARENT_SHA256:
        raise ValueError("APBR-VPR Dense Final SHA256 mismatch")
    archive = DATA / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169001437 or file_sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Official CIFAR-100 archive mismatch")
    parent = torch.load(paths["parent"], map_location="cpu", weights_only=False)
    if (parent.get("protocol_id") != "cifar100_50k10k_final_v1"
            or parent.get("epoch") != 200 or parent.get("steps") != 78200
            or parent.get("test_accessed") is not False
            or parent.get("identity", {}).get("method") != "apbr_sam"
            or parent.get("identity", {}).get("seed") != 0):
        raise ValueError("Frozen parent identity mismatch")
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible-layer contract mismatch")
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if geometry.get("method") != "vpr" or mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
        raise ValueError("Parent geometry/parameter mapping mismatch")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    for layer_name, layer in layers.items():
        scale = logs[layer_name + ".weight"]
        if scale.dtype != torch.float64 or scale.shape != layer.weight.shape or not bool(torch.isfinite(scale).all()):
            raise ValueError(f"Geometry mismatch: {layer_name}")

    train_path = DATA / "extracted/cifar-100-python/train"
    train_data, _ = cifar(train_path, 50000)
    ids = np.random.default_rng(CALIBRATION_SEED).permutation(50000)[:1024].tolist()
    historical_raw = torch.load(paths["raw"], map_location="cpu", weights_only=False)
    if ids != historical_raw["calibration_ids"] or historical_raw["parent"]["sha256"] != PARENT_SHA256:
        raise ValueError("Historical calibration identity mismatch")
    mean = torch.tensor(formal["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(formal["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    write_new(OUTPUT / "preflight.json", {
        "status": "PREFLIGHT_PASS", "parent_sha256": PARENT_SHA256,
        "historical_raw_sha256": file_sha(paths["raw"]),
        "archive_sha256": ARCHIVE_SHA256, "train_pickle_sha256": file_sha(train_path),
        "calibration_ids": ids, "calibration_ids_sha256": ids_sha(ids),
        "protocol_id": parent["protocol_id"], "geometry_method": "vpr",
        "eligible_layers": list(layers), "eligible_positions": 11157504,
        "expected_deletions": 5578752, "methods": METHODS,
        "tau": TAU, "shuffle_seed": SHUFFLE_SEED,
        "damping": 0.01, "blocksize": 128,
        "hessian_dtype": "float32", "obs_compute_dtype": "float64",
        "source_sha256": source_sha, "test_accessed": False,
    })

    adapters = {name: SparseGPTAdapter(layer, compute_dtype=torch.float32)
                for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _module, args, a=adapters[name]: a.add_batch(args[0]))
             for name, layer in layers.items()]
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, 1024, 128):
                batch = ids[offset:offset + 128]
                model(normalized(train_data, batch, mean, std))
                print(f"calibration {offset + len(batch)}/1024", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    calibration_seconds = time.perf_counter() - start

    states = {method: {key: value.detach().cpu().clone() for key, value in parent["model"].items()}
              for method in METHODS}
    masks = {method: {} for method in METHODS}
    layer_rows = []
    for layer_index, (layer_name, layer) in enumerate(layers.items(), 1):
        key = layer_name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).clone()
        s = weight_to_matrix(logs[key]).clone()
        h = adapters.pop(layer_name).hessians()[0]
        if h.shape != (w.shape[1], w.shape[1]) or not bool(torch.isfinite(h).all()):
            raise ValueError(f"Invalid Hessian: {layer_name}")
        row = {"index": layer_index, "layer": layer_name,
               "shape": list(layer.weight.shape), "K": w.shape[1],
               "hessian_sha256": tensor_sha({"hessian": h}), "methods": {}}
        for method in METHODS:
            start = time.perf_counter()
            result = apbr_nearoptimal_conditional_obs_prune(
                w, h, s, tau=TAU, geometry_mode=method,
                shuffle_seed=SHUFFLE_SEED if method == "shuffled" else None,
                blocksize=128, percdamp=0.01,
            )
            elapsed = time.perf_counter() - start
            keep = result.keep_mask.reshape_as(layer.weight).detach().cpu().clone()
            value = result.weight.reshape_as(layer.weight).detach().cpu().clone()
            validate_keep_mask(result.keep_mask, 2, 4)
            if (keep.dtype != torch.bool or not bool(torch.isfinite(value).all())
                    or int(torch.count_nonzero(value[~keep]))):
                raise ValueError(f"Invalid 2:4 compensated output: {method}/{layer_name}")
            masks[method][key] = keep
            states[method][key] = value
            row["methods"][method] = {
                "deleted": int((~keep).sum()),
                "mask_sha256": tensor_sha({key: keep}),
                "physical_quadratic_error": physical_error(w, result.weight, h),
                "near_optimal_first_gt1": int((result.first_candidate_counts > 1).sum()),
                "near_optimal_second_gt1": int((result.second_candidate_counts > 1).sum()),
                "first_selected_cost_sum": float(result.first_selected_cost.sum()),
                "first_min_cost_sum": float(result.first_min_cost.sum()),
                "second_selected_cost_sum": float(result.second_selected_cost.sum()),
                "second_min_cost_sum": float(result.second_min_cost.sum()),
                "first_choices_sha256": tensor_sha({"first": result.first_choices}),
                "second_choices_sha256": tensor_sha({"second": result.second_choices}),
                "decomposition_seconds": result.decomposition_seconds,
                "selection_compensation_seconds": result.selection_compensation_seconds,
                "wall_seconds": elapsed,
            }
        for method in METHODS[1:]:
            row["methods"][method]["position_flips_vs_physical"] = int((masks[method][key] != masks["physical"][key]).sum())
            row["methods"][method]["position_flips_vs_unit"] = int((masks[method][key] != masks["unit"][key]).sum())
        layer_rows.append(row)
        write_new(OUTPUT / f"layer_{layer_index:02d}.json", row)
        print(f"layer {layer_index}/19 {layer_name} complete", flush=True)

    for method in METHODS:
        if sum(int((~keep).sum()) for keep in masks[method].values()) != 5578752:
            raise ValueError(f"Deletion count mismatch: {method}")
        for key, value in states[method].items():
            if key not in masks[method] and not torch.equal(value, parent["model"][key]):
                raise ValueError(f"Excluded/BN state changed: {method}/{key}")
    historical_path = ROOT / "artifacts/conditional_greedy_obs_apbr_seed0_cpu_20261103_v2/greedy_conditional.pt"
    historical_status = ROOT / "artifacts/conditional_greedy_obs_apbr_seed0_cpu_20261103_v2/results.json"
    old_summary = json.loads(historical_status.read_text(encoding="utf-8"))
    if (old_summary["parent_sha256"] != PARENT_SHA256
            or old_summary["checkpoint_sha256"]["greedy_conditional"] != file_sha(historical_path)):
        raise ValueError("Historical physical conditional replay identity mismatch")
    old = torch.load(historical_path, map_location="cpu", weights_only=False)
    if old["parent_sha256"] != PARENT_SHA256:
        raise ValueError("Historical physical conditional parent mismatch")
    for key, value in states["physical"].items():
        if not torch.equal(value, old["model"][key]):
            raise ValueError(f"Real-model physical conditional weight replay failed: {key}")
    for key, keep in masks["physical"].items():
        if not torch.equal(keep, old["masks"][key]):
            raise ValueError(f"Real-model physical conditional Mask replay failed: {key}")
    write_new(OUTPUT / "historical_physical_replay.json", {
        "status": "BITWISE_EQUAL_19_LAYERS", "historical_checkpoint_sha256": file_sha(historical_path),
        "historical_status_sha256": file_sha(historical_status),
        "parent_sha256": PARENT_SHA256, "test_accessed": False,
    })
    if current_sources(formal_path) != source_sha:
        raise RuntimeError("Source changed during execution; refusing Test")
    frozen_sha = {}
    frozen_masks = {}
    for method in METHODS:
        path = OUTPUT / f"{method}_compensated.pt"
        payload = {
            "status": "FROZEN_BEFORE_FINAL_TEST", "protocol_id": parent["protocol_id"],
            "method": method, "parent_sha256": PARENT_SHA256,
            "model": states[method], "masks": masks[method],
            "mask_sha256": tensor_sha(masks[method]),
            "calibration_ids_sha256": ids_sha(ids), "tau": TAU,
            "test_accessed_during_calibration_pruning": False,
            "recovery_epochs": 0, "bn_only_calibration": False,
        }
        write_new(path, payload)
        frozen_sha[method] = file_sha(path)
        frozen_masks[method] = payload["mask_sha256"]
    write_new(OUTPUT / "pre_test_freeze.json", {
        "status": "ALL_19_LAYER_CHECKPOINTS_FROZEN_BEFORE_TEST",
        "parent_sha256": PARENT_SHA256, "checkpoint_sha256": frozen_sha,
        "mask_sha256": frozen_masks, "source_sha256": source_sha,
        "calibration_ids_sha256": ids_sha(ids), "test_accessed": False,
    })
    print("all four checkpoint/mask hashes frozen; beginning fixed Final Test", flush=True)

    test_data, test_labels = cifar(DATA / "extracted/cifar-100-python/test", 10000)
    evaluations = {}
    for method in METHODS:
        start = time.perf_counter()
        evaluations[method] = evaluate(states[method], test_data, test_labels, mean, std)
        evaluations[method]["evaluation_seconds"] = time.perf_counter() - start
        print(f"Final Test {method}: {evaluations[method]['test_accuracy']:.4f}", flush=True)
    summary = {
        "status": "COMPLETE_CPU_EXPLORATORY_SEED0",
        "protocol_id": parent["protocol_id"], "parent_sha256": PARENT_SHA256,
        "parent_geometry_method": "vpr", "tau": TAU,
        "checkpoint_sha256": frozen_sha, "mask_sha256": frozen_masks,
        "source_sha256": source_sha, "calibration_ids_sha256": ids_sha(ids),
        "layer_count": len(layer_rows), "eligible_positions": 11157504,
        "deletions_per_method": 5578752, "calibration_seconds": calibration_seconds,
        "test_accessed_during_calibration_pruning": False,
        "final_test_count_per_method": 10000,
        "full_test_accessed_after_freeze": True,
        "raw_hard_test": evaluations,
        "physical_quadratic_error_sum": {
            method: sum(row["methods"][method]["physical_quadratic_error"] for row in layer_rows)
            for method in METHODS
        },
        "position_flips_vs_unit": {
            method: sum(row["methods"][method].get("position_flips_vs_unit", 0) for row in layer_rows)
            for method in METHODS
        },
        "near_optimal_first_gt1_sum": {
            method: sum(row["methods"][method]["near_optimal_first_gt1"] for row in layer_rows)
            for method in METHODS
        },
        "near_optimal_second_gt1_sum": {
            method: sum(row["methods"][method]["near_optimal_second_gt1"] for row in layer_rows)
            for method in METHODS
        },
        "pruning_wall_seconds_sum": {
            method: sum(row["methods"][method]["wall_seconds"] for row in layer_rows)
            for method in METHODS
        },
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    write_new(OUTPUT / "results.json", summary)
    print(json.dumps({"status": summary["status"], "accuracy": {
        method: evaluations[method]["test_accuracy"] for method in METHODS}},
        ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / "failure.json"
        if not path.exists():
            write_new(path, {"status": "FAILED", "traceback": traceback.format_exc(),
                             "automatic_retry": False})
        raise
