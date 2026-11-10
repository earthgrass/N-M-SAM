"""CPU-only same-parent, same-solver APBR vs GPT fixed-mask OBS audit."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.fixedmask_obs import FixedMaskOBSAdapter, fixedmask_obs_matrix
from benchmark.nm import eligible_layers
from scripts.audit_apbr_gpt_mask_match_local import PAIRS, file_sha, tensor_sha

DATA = Path("C:/Users/earthgrass73/Desktop/NM稀疏/data/cifar100_official_20261026")
OUTPUT = ROOT / "artifacts/apbr_gpt_mask_fixed_obs_cpu_seed0_20261103"


def write_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pt":
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def cifar(path: Path, count: int):
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data, labels = np.asarray(raw[b"data"]), np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if (data.shape != (count, 3072) or data.dtype != np.uint8 or labels.shape != (count,)
            or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, count // 100))):
        raise ValueError("CIFAR-100 identity mismatch")
    return data, labels


def normalized(data: np.ndarray, ids, mean, std) -> torch.Tensor:
    ids = np.asarray(ids, dtype=np.int64)
    x = torch.from_numpy(data[ids].copy()).reshape(len(ids), 3, 32, 32).float().div_(255)
    return (x - mean) / std


def error(w: torch.Tensor, candidate: torch.Tensor, h: torch.Tensor) -> float:
    delta = (w - candidate).double()
    return float(torch.einsum("ri,ij,rj->", delta, h.double(), delta))


def evaluate(state, data, labels, mean, std):
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(state, strict=True)
    count = correct = 0
    ce_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(data), 128):
            ids = np.arange(start, min(start + 128, len(data)))
            logits = model(normalized(data, ids, mean, std))
            targets = torch.from_numpy(labels[ids].copy())
            ce_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(1) == targets).sum())
            count += len(ids)
    return {"test_count": count, "test_accuracy": correct / count, "test_ce": ce_sum / count}


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("This experiment is explicitly CPU-only")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    paths = PAIRS["apbr_vpr_seed0"]
    formal_path = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    fixed_path = ROOT / "configs/apbr_seed0_fixedmask_obs_20261027_blackwell.yaml"
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    fixed = yaml.safe_load(fixed_path.read_text(encoding="utf-8"))
    if (DATA / "cifar-100-python.tar.gz").stat().st_size != 169001437 or file_sha(DATA / "cifar-100-python.tar.gz") != "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7":
        raise ValueError("Official archive identity mismatch")
    if file_sha(paths["parent"]) != "4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82":
        raise ValueError("Parent SHA256 mismatch")
    parent = torch.load(paths["parent"], map_location="cpu", weights_only=False)
    apbr_raw = torch.load(paths["raw"], map_location="cpu", weights_only=False)
    gpt_mask = torch.load(paths["gpt"], map_location="cpu", weights_only=False)
    masks = {"apbr": apbr_raw["masks"], "gpt": gpt_mask}
    if parent.get("test_accessed") is not False or apbr_raw["parent"]["sha256"] != file_sha(paths["parent"]):
        raise ValueError("Parent identity/test isolation mismatch")
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if set(masks["apbr"]) != set(masks["gpt"]) or set(masks["gpt"]) != {name + ".weight" for name in layers}:
        raise ValueError("Mask layer set mismatch")
    for name in masks:
        for key, keep in masks[name].items():
            if keep.shape != parent["model"][key].shape or keep.dtype != torch.bool:
                raise ValueError(f"Invalid Mask shape/dtype: {name}/{key}")
            validate_keep_mask(keep.reshape(keep.shape[0], -1), 2, 4)
    train_data, train_labels = cifar(DATA / "extracted/cifar-100-python/train", 50000)
    ids = np.random.default_rng(fixed["calibration"]["selection_seed"]).permutation(50000)[:1024].tolist()
    if ids != apbr_raw["calibration_ids"]:
        raise ValueError("Frozen calibration IDs mismatch")
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
    states = {name: {key: value.detach().cpu().clone() for key, value in parent["model"].items()} for name in masks}
    reports = []
    for layer_name, layer in layers.items():
        key = layer_name + ".weight"
        adapter = adapters.pop(layer_name)
        h = adapter.hessians()[0]
        original = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        row = {"layer": layer_name, "K": int(original.shape[1]), "methods": {}}
        for name in ("apbr", "gpt"):
            keep = masks[name][key].reshape_as(original)
            stamp = time.perf_counter()
            result = fixedmask_obs_matrix(original, h, keep, compute_dtype=torch.float32,
                                          percdamp=fixed["obs"]["percdamp"], blocksize=fixed["obs"]["blocksize"])
            candidate = result.weight.reshape_as(layer.weight).detach().clone()
            if not bool(torch.isfinite(candidate).all()) or int(torch.count_nonzero(candidate[~masks[name][key]])):
                raise RuntimeError(f"Nonfinite/regrowth at {layer_name} for {name}")
            states[name][key] = candidate
            row["methods"][name] = {
                "raw_physical_quadratic_error": error(original, original * keep, h),
                "obs_physical_quadratic_error": error(original, result.weight, h),
                "survivor_changed_count": result.survivor_changed_count,
                "decomposition_seconds": result.decomposition_seconds,
                "compensation_seconds": result.compensation_seconds,
                "wall_seconds": time.perf_counter() - stamp,
            }
        reports.append(row)
        write_new(OUTPUT / f"layer_{len(reports):02d}.json", row)
        print(f"layer {len(reports)}/19 {layer_name} complete", flush=True)
        del adapter, h
    for name in states:
        for key, value in states[name].items():
            if key not in masks[name] and not torch.equal(value, parent["model"][key]):
                raise RuntimeError(f"Excluded/BN state changed: {key}")
    frozen = {}
    for name in ("apbr", "gpt"):
        payload = {
            "status": "fixed_before_test", "protocol_id": parent["protocol_id"], "seed": 0,
            "method": f"{name}_mask_same_fixed_obs_cpu", "model": states[name],
            "masks": masks[name], "mask_sha256": tensor_sha(masks[name]),
            "parent_sha256": file_sha(paths["parent"]), "calibration_ids": ids,
            "test_accessed_during_compensation": False, "recovery_epochs": 0,
        }
        path = OUTPUT / f"{name}_fixed_obs.pt"
        write_new(path, payload)
        frozen[name] = file_sha(path)
    write_new(OUTPUT / "pre_test_freeze.json", {"checkpoint_sha256": frozen, "calibration_seconds": calibration_seconds})
    print("both checkpoints frozen; beginning one fixed Test evaluation each", flush=True)
    test_data, test_labels = cifar(DATA / "extracted/cifar-100-python/test", 10000)
    results = {name: evaluate(states[name], test_data, test_labels, mean, std) for name in ("apbr", "gpt")}
    summary = {
        "status": "COMPLETE_CPU_FULL_MODEL", "parent_sha256": file_sha(paths["parent"]),
        "checkpoint_sha256": frozen, "mask_sha256": {name: tensor_sha(masks[name]) for name in masks},
        "source_sha256": file_sha(Path(__file__)), "formal_config_sha256": file_sha(formal_path),
        "obs_config_sha256": file_sha(fixed_path), "calibration_seconds": calibration_seconds,
        "layer_count": len(reports), "results": results,
        "compensation_test_isolation": True,
    }
    write_new(OUTPUT / "results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        failure = OUTPUT / "failure.json"
        if not failure.exists():
            write_new(failure, {"status": "FAILED", "traceback": traceback.format_exc()})
        raise
