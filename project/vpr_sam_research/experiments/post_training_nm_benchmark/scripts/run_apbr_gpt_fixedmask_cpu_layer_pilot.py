"""Real Train-side single-layer comparison; never reads CIFAR-100 Test."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import grouped_patch_matrices, weight_to_matrix
from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from scripts.audit_apbr_gpt_mask_match_local import PAIRS, file_sha


class StopAtLayer(Exception):
    pass


def quadratic_error(weight: torch.Tensor, candidate: torch.Tensor, h: torch.Tensor) -> float:
    delta = (weight - candidate).double()
    return float(torch.einsum("ri,ij,rj->", delta, h.double(), delta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, choices=(32, 1024), default=32)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("This pilot is deliberately CPU-only")
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    layer_name = "layer1.0.conv1"
    key = layer_name + ".weight"
    paths = PAIRS["apbr_vpr_seed0"]
    frozen = yaml.safe_load((ROOT / "configs/apbr_seed0_fixedmask_obs_20261027_blackwell.yaml").read_text(encoding="utf-8"))
    formal = yaml.safe_load((ROOT / "configs/cifar100_50k10k_final_v1.yaml").read_text(encoding="utf-8"))
    parent = torch.load(paths["parent"], map_location="cpu", weights_only=False)
    raw = torch.load(paths["raw"], map_location="cpu", weights_only=False)
    gpt = torch.load(paths["gpt"], map_location="cpu", weights_only=False)
    if file_sha(paths["parent"]) != raw["parent"]["sha256"]:
        raise ValueError("Parent mismatch")
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"])
    layer = dict(model.named_modules())[layer_name]
    archive = Path("C:/Users/earthgrass73/Desktop/NM稀疏/data/cifar100_official_20261026")
    if file_sha(archive / "cifar-100-python.tar.gz") != "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7":
        raise ValueError("CIFAR-100 archive SHA256 mismatch")
    with (archive / "extracted/cifar-100-python/train").open("rb") as stream:
        train = pickle.load(stream, encoding="bytes")
    data = np.asarray(train[b"data"])
    if data.shape != (50000, 3072) or data.dtype != np.uint8:
        raise ValueError("Training data identity mismatch")
    ids = np.random.default_rng(frozen["calibration"]["selection_seed"]).permutation(50000)[:args.samples].copy()
    mean = torch.tensor(formal["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(formal["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    captured = {}

    def hook(_module, args):
        captured["input"] = args[0].detach().clone()
        raise StopAtLayer()

    handle = layer.register_forward_pre_hook(hook)
    h_sum = None
    vectors = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), 128):
                batch_ids = ids[start:start + 128]
                inputs = torch.from_numpy(data[batch_ids].copy()).reshape(len(batch_ids), 3, 32, 32).float().div_(255)
                inputs = (inputs - mean) / std
                try:
                    model(inputs)
                except StopAtLayer:
                    pass
                if "input" not in captured:
                    raise RuntimeError("Target input was not captured")
                patches = grouped_patch_matrices(layer, captured.pop("input"))[0].float()
                product = patches @ patches.T
                h_sum = product if h_sum is None else h_sum + product
                vectors += patches.shape[1]
    finally:
        handle.remove()
    h = 2 * h_sum / vectors
    weight = weight_to_matrix(layer.weight.detach()).float()
    results = {}
    for name, mask in (("apbr", raw["masks"][key]), ("gpt", gpt[key])):
        keep = mask.reshape_as(weight)
        started = time.perf_counter()
        result = fixedmask_obs_matrix(
            weight, h, keep, compute_dtype=torch.float32,
            percdamp=frozen["obs"]["percdamp"], blocksize=frozen["obs"]["blocksize"],
        )
        elapsed = time.perf_counter() - started
        raw_weight = weight * keep
        results[name] = {
            "raw_physical_quadratic_error": quadratic_error(weight, raw_weight, h),
            "obs_physical_quadratic_error": quadratic_error(weight, result.weight, h),
            "survivor_changed_count": result.survivor_changed_count,
            "deleted_exact_zero": bool(torch.count_nonzero(result.weight[~keep]) == 0),
            "finite": bool(torch.isfinite(result.weight).all()),
            "damp": result.damp,
            "decomposition_seconds": result.decomposition_seconds,
            "compensation_seconds": result.compensation_seconds,
            "wall_seconds": elapsed,
        }
    evidence = {
        "status": "CPU_SINGLE_LAYER_PILOT_ONLY",
        "layer": layer_name,
        "train_sample_ids": ids.tolist(),
        "train_sample_count": args.samples,
        "hessian_shape": list(h.shape),
        "parent_sha256": file_sha(paths["parent"]),
        "apbr_mask_file_sha256": file_sha(paths["raw"]),
        "gpt_mask_file_sha256": file_sha(paths["gpt"]),
        "obs_config": frozen["obs"],
        "results": results,
        "test_accessed": False,
        "full_model_accuracy_measured": False,
    }
    out = ROOT / f"docs/research/apbr_gpt_mask_compensation_match_20261103/cpu_layer_pilot_{args.samples}.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
