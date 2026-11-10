"""Fixed Seed 2/3 SAM conditional OBS CPU evaluation, one seed per invocation."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.apbr_nearoptimal_conditional_obs import apbr_nearoptimal_conditional_obs_prune
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA, cifar, normalized, evaluate

SYNC = ROOT / "artifacts/dense_sam_apbr_l2_seed1to4_20261103_local_copy"
CONFIG = ROOT / "configs/cifar100_50k10k_final_v1_sam_seed245_rtx5090_20261103.yaml"
CALIB_SHA = "3b7da27d79762235c87f18571bc73224d6e6f2b3ba663b7d9a4a3de312f23157"
TEST_SHA = "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864"
ARCHIVE_SHA = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def write_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pt":
        torch.save(obj, temp)
    else:
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    temp.replace(path)


def sources() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "selector": ROOT / "baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py",
        "hessian": ROOT / "baseline_adapters/sparsegpt/adapter.py",
        "model": ROOT / "baseline_adapters/cifar100_model.py",
        "eligibility": ROOT / "benchmark/nm.py",
        "calibration_evaluation": ROOT / "scripts/run_apbr_gpt_fixedmask_cpu_full.py",
        "formal_config": CONFIG,
    }
    return {name: file_sha(path) for name, path in paths.items()}


def main(seed: int) -> None:
    if seed not in (2, 3) or torch.cuda.is_available():
        raise ValueError("Require Seed 2/3 and CPU-only PyTorch")
    output = ROOT / f"artifacts/sam_conditional_seed{seed}_cpu_20261104_v1"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    source_sha = sources()
    local_dir = SYNC / f"seed{seed}/dense_sam"
    proof = json.loads((local_dir / "LOCAL_SHA256.json").read_text(encoding="utf-8"))
    expected = proof["job"]["checkpoint_sha256"]
    parent_path = local_dir / "final_epoch_200.pt"
    if (proof["job"]["state"] != "completed"
            or proof["file_sha256"]["final_epoch_200.pt"] != expected
            or file_sha(parent_path) != expected):
        raise ValueError("SAM Dense Final transfer SHA256 mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    identity = parent["identity"]
    if (parent["protocol_id"] != "cifar100_50k10k_final_v1"
            or parent["epoch"] != 200 or parent["steps"] != 78200
            or parent["test_accessed"] is not False or parent["status"] != "healthy"
            or identity["seed"] != seed or identity["method"] != "dense_sam"
            or identity["config_sha256"] != file_sha(CONFIG)
            or identity["initial_model_sha256"] != proof["job"]["initial_model_sha256"]):
        raise ValueError("SAM Dense Final identity mismatch")
    l2proof = json.loads((SYNC / f"seed{seed}/apbr_fixed_l2/LOCAL_SHA256.json")
                         .read_text(encoding="utf-8"))
    if identity["initial_model_sha256"] != l2proof["job"]["initial_model_sha256"]:
        raise ValueError("SAM/APBR-L2 paired initialization mismatch")
    archive = DATA / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169001437 or file_sha(archive) != ARCHIVE_SHA:
        raise ValueError("Official CIFAR-100 archive mismatch")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible-layer contract mismatch")
    train_path = DATA / "extracted/cifar-100-python/train"
    train_data, _ = cifar(train_path, 50000)
    ids = np.random.default_rng(20261027).permutation(50000)[:1024].tolist()
    if ids_sha(ids) != CALIB_SHA:
        raise ValueError("Frozen calibration IDs mismatch")
    mean = torch.tensor(config["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(config["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    write_new(output / "preflight.json", {
        "status": "PREFLIGHT_PASS", "seed": seed, "parent_sha256": expected,
        "paired_initial_model_sha256": identity["initial_model_sha256"],
        "archive_sha256": ARCHIVE_SHA, "train_pickle_sha256": file_sha(train_path),
        "calibration_ids_sha256": CALIB_SHA, "calibration_ids": ids,
        "eligible_layers": list(layers), "eligible_positions": 11157504,
        "tau": 0.01, "damping": 0.01, "blocksize": 128,
        "hessian_dtype": "float32", "obs_compute_dtype": "float64",
        "source_sha256": source_sha, "test_accessed": False,
    })
    adapters = {name: SparseGPTAdapter(layer, compute_dtype=torch.float32)
                for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(
        lambda _module, args, a=adapters[name]: a.add_batch(args[0]))
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
    state = {key: value.detach().cpu().clone() for key, value in parent["model"].items()}
    masks = {}
    rows = []
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        weight = weight_to_matrix(layer.weight.detach()).clone()
        hessian = adapters.pop(name).hessians()[0]
        if hessian.shape != (weight.shape[1], weight.shape[1]) or not bool(torch.isfinite(hessian).all()):
            raise ValueError(f"Invalid Hessian: {name}")
        start = time.perf_counter()
        result = apbr_nearoptimal_conditional_obs_prune(
            weight, hessian, None, tau=0.01, geometry_mode="physical",
            blocksize=128, percdamp=0.01)
        keep = result.keep_mask.reshape_as(layer.weight).detach().cpu().clone()
        value = result.weight.reshape_as(layer.weight).detach().cpu().clone()
        validate_keep_mask(result.keep_mask, 2, 4)
        if keep.dtype != torch.bool or not bool(torch.isfinite(value).all()) or int(torch.count_nonzero(value[~keep])):
            raise ValueError(f"Invalid compensated output: {name}")
        state[key], masks[key] = value, keep
        delta = (weight - result.weight).double()
        row = {"index": index, "layer": name, "K": weight.shape[1],
               "hessian_sha256": tensor_sha({"hessian": hessian}),
               "mask_sha256": tensor_sha({key: keep}), "deleted": int((~keep).sum()),
               "physical_quadratic_error": float(.5 * torch.einsum(
                   "ri,ij,rj->", delta, hessian.double(), delta)),
               "decomposition_seconds": result.decomposition_seconds,
               "selection_compensation_seconds": result.selection_compensation_seconds,
               "wall_seconds": time.perf_counter() - start}
        rows.append(row)
        write_new(output / f"layer_{index:02d}.json", row)
        print(f"layer {index}/19 {name} complete", flush=True)
    if sum(int((~mask).sum()) for mask in masks.values()) != 5578752:
        raise ValueError("Deletion count mismatch")
    for key, value in state.items():
        if key not in masks and not torch.equal(value, parent["model"][key]):
            raise ValueError(f"Excluded/BN state changed: {key}")
    if sources() != source_sha:
        raise RuntimeError("Source/config changed; refusing Test")
    payload = {"status": "FROZEN_BEFORE_FINAL_TEST", "protocol_id": parent["protocol_id"],
               "seed": seed, "method": "physical_conditional_joint_obs",
               "parent_sha256": expected, "model": state, "masks": masks,
               "mask_sha256": tensor_sha(masks), "calibration_ids_sha256": CALIB_SHA,
               "recovery_epochs": 0, "bn_only_calibration": False,
               "test_accessed_during_pruning": False}
    checkpoint_path = output / "physical_compensated.pt"
    write_new(checkpoint_path, payload)
    frozen_sha = file_sha(checkpoint_path)
    write_new(output / "pre_test_freeze.json", {
        "status": "ALL_19_LAYER_CHECKPOINTS_FROZEN_BEFORE_TEST",
        "checkpoint_sha256": frozen_sha, "mask_sha256": payload["mask_sha256"],
        "parent_sha256": expected, "source_sha256": source_sha,
        "calibration_ids_sha256": CALIB_SHA, "test_accessed": False})
    test_path = DATA / "extracted/cifar-100-python/test"
    if file_sha(test_path) != TEST_SHA:
        raise ValueError("Frozen official Test pickle identity mismatch")
    test_data, test_labels = cifar(test_path, 10000)
    evaluations = {"dense": evaluate(parent["model"], test_data, test_labels, mean, std),
                   "physical": evaluate(state, test_data, test_labels, mean, std)}
    write_new(output / "results.json", {
        "status": f"COMPLETE_CPU_EXPLORATORY_SEED{seed}", "seed": seed,
        "parent_sha256": expected, "checkpoint_sha256": frozen_sha,
        "mask_sha256": payload["mask_sha256"], "source_sha256": source_sha,
        "calibration_ids_sha256": CALIB_SHA, "test_pickle_sha256": TEST_SHA,
        "layer_count": len(rows), "eligible_positions": 11157504,
        "deletions": 5578752, "calibration_seconds": calibration_seconds,
        "pruning_wall_seconds_sum": sum(row["wall_seconds"] for row in rows),
        "physical_quadratic_error_sum": sum(row["physical_quadratic_error"] for row in rows),
        "test_accessed_during_calibration_pruning": False,
        "full_test_accessed_after_freeze": True, "test": evaluations})
    print(json.dumps({"status": f"COMPLETE_CPU_EXPLORATORY_SEED{seed}",
                      "dense": evaluations["dense"]["test_accuracy"],
                      "physical": evaluations["physical"]["test_accuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    try:
        main(args.seed)
    except Exception:
        output = ROOT / f"artifacts/sam_conditional_seed{args.seed}_cpu_20261104_v1"
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "failure.json").exists():
            write_new(output / "failure.json", {
                "status": "FAILED", "traceback": traceback.format_exc(),
                "automatic_retry": False})
        raise
