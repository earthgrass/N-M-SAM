"""Seed-1 SAM conditional OBS versus APBR geometric near-optimal OBS.

Both arms are frozen before one Final Test each. No Dense training or recovery.
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
from scripts.audit_apbr_gpt_mask_match_local import file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA, cifar, normalized, evaluate

SEED = 1
COHORT = ROOT / "artifacts/cifar100_50k10k_final_v1_blackwell_handoff/seed1_remote_copy"
OUT = ROOT / "artifacts/sam_apbr_conditional_seed1_cpu_20261103_v1"
EXPECTED = {
    "sam_conditional": ("dense_sam", "e1909b70c0574a17c0ba1e5558a738848e943a67eed2c2bc2bb150bd3c96da7b", "physical"),
    "apbr_geo_conditional": ("apbr_sam", "6571e30ed783508f28bc378aa3b22a739bf1ddbbb790db3d98c08858e126e10b", "true"),
}
ARCHIVE_SHA = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
TAU = 0.01


def write_new(path: Path, obj) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pt":
        torch.save(obj, temp)
    else:
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def ids_sha(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def source_sha(formal: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "selector": ROOT / "baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py",
        "hessian": ROOT / "baseline_adapters/sparsegpt/adapter.py",
        "model": ROOT / "baseline_adapters/cifar100_model.py",
        "eligibility": ROOT / "benchmark/nm.py",
        "formal_config": formal,
    }
    return {name: file_sha(path) for name, path in paths.items()}


def one_arm(method: str, parent_kind: str, parent_sha: str, mode: str,
            data: np.ndarray, ids: list[int], mean: torch.Tensor, std: torch.Tensor) -> tuple[dict, dict]:
    parent_path = COHORT / parent_kind / "final_epoch_200.pt"
    eval_path = COHORT / "test_final" / ("dense_final.json" if parent_kind == "dense_sam" else "apbr_final.json")
    eval_record = json.loads(eval_path.read_text(encoding="utf-8"))
    if file_sha(parent_path) != parent_sha or eval_record["checkpoint_sha256"] != parent_sha:
        raise ValueError(f"Seed-1 parent SHA256 mismatch: {method}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (parent["protocol_id"] != "cifar100_50k10k_final_v1" or parent["epoch"] != 200
            or parent["steps"] != 78200 or parent["test_accessed"] is not False
            or parent["identity"]["seed"] != SEED or parent["identity"]["method"] != parent_kind):
        raise ValueError(f"Seed-1 parent identity mismatch: {method}")
    raw_path = COHORT / "methods" / ("sparsegpt" if parent_kind == "dense_sam" else "apbr_sam") / "raw_hard.pt"
    historical_raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    if historical_raw["calibration_ids"] != ids or historical_raw["parent"]["sha256"] != parent_sha:
        raise ValueError(f"Historical calibration identity mismatch: {method}")
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible-layer identity mismatch")
    if mode == "true":
        geometry = parent["stepper"]["geometry"]
        mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
        if geometry["method"] != "vpr" or mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
            raise ValueError("APBR geometry/parameter mismatch")
        logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    else:
        logs = {}
    adapters = {name: SparseGPTAdapter(layer, compute_dtype=torch.float32) for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _m, args, a=adapters[name]: a.add_batch(args[0]))
             for name, layer in layers.items()]
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, 1024, 128):
                batch = ids[offset:offset + 128]
                model(normalized(data, batch, mean, std))
                print(f"{method} calibration {offset + len(batch)}/1024", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    calibration_seconds = time.perf_counter() - start
    state = {key: value.detach().cpu().clone() for key, value in parent["model"].items()}
    masks = {}
    layer_rows = []
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        w = weight_to_matrix(layer.weight.detach()).clone()
        s = None
        if mode == "true":
            if logs[key].dtype != torch.float64 or logs[key].shape != layer.weight.shape or not bool(torch.isfinite(logs[key]).all()):
                raise ValueError(f"Geometry tensor mismatch: {key}")
            s = weight_to_matrix(logs[key]).clone()
        h = adapters.pop(name).hessians()[0]
        if h.shape != (w.shape[1], w.shape[1]) or not bool(torch.isfinite(h).all()):
            raise ValueError(f"Invalid Hessian: {name}")
        start = time.perf_counter()
        result = apbr_nearoptimal_conditional_obs_prune(w, h, s, tau=TAU, geometry_mode=mode,
                                                        blocksize=128, percdamp=0.01)
        elapsed = time.perf_counter() - start
        keep = result.keep_mask.reshape_as(layer.weight).detach().cpu().clone()
        value = result.weight.reshape_as(layer.weight).detach().cpu().clone()
        validate_keep_mask(result.keep_mask, 2, 4)
        if keep.dtype != torch.bool or not bool(torch.isfinite(value).all()) or int(torch.count_nonzero(value[~keep])):
            raise ValueError(f"Invalid compensated weight: {method}/{name}")
        state[key] = value
        masks[key] = keep
        delta = (w - result.weight).double()
        row = {"index": index, "layer": name, "K": w.shape[1], "hessian_sha256": tensor_sha({"hessian": h}),
               "mask_sha256": tensor_sha({key: keep}), "deleted": int((~keep).sum()),
               "physical_quadratic_error": float(.5 * torch.einsum("ri,ij,rj->", delta, h.double(), delta)),
               "near_optimal_first_gt1": int((result.first_candidate_counts > 1).sum()),
               "near_optimal_second_gt1": int((result.second_candidate_counts > 1).sum()),
               "decomposition_seconds": result.decomposition_seconds,
               "selection_compensation_seconds": result.selection_compensation_seconds,
               "wall_seconds": elapsed}
        layer_rows.append(row)
        write_new(OUT / method / f"layer_{index:02d}.json", row)
        print(f"{method} layer {index}/19 {name}", flush=True)
    if sum(int((~mask).sum()) for mask in masks.values()) != 5578752:
        raise ValueError(f"Deletion count mismatch: {method}")
    for key, value in state.items():
        if key not in masks and not torch.equal(value, parent["model"][key]):
            raise ValueError(f"Excluded/BN state changed: {method}/{key}")
    payload = {"status": "FROZEN_BEFORE_TEST", "protocol_id": parent["protocol_id"],
               "seed": SEED, "method": method, "parent_sha256": parent_sha,
               "model": state, "masks": masks, "mask_sha256": tensor_sha(masks),
               "calibration_ids_sha256": ids_sha(ids), "tau": TAU, "recovery_epochs": 0,
               "test_accessed_during_pruning": False}
    ckpt_path = OUT / f"{method}.pt"
    write_new(ckpt_path, payload)
    info = {"method": method, "parent_sha256": parent_sha, "parent_dense_test_accuracy": eval_record["test_accuracy"],
            "parent_dense_test_ce": eval_record["test_loss"], "historical_raw_sha256": file_sha(raw_path),
            "checkpoint_sha256": file_sha(ckpt_path), "mask_sha256": payload["mask_sha256"],
            "calibration_seconds": calibration_seconds, "layer_count": len(layer_rows),
            "deletions": 5578752, "pruning_wall_seconds_sum": sum(row["wall_seconds"] for row in layer_rows),
            "physical_quadratic_error_sum": sum(row["physical_quadratic_error"] for row in layer_rows)}
    return info, state


def main() -> None:
    if torch.cuda.is_available() or OUT.exists():
        raise RuntimeError("This is a one-shot local CPU run; output must not exist")
    OUT.mkdir(parents=True)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    formal_path = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    source = source_sha(formal_path)
    archive = DATA / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169001437 or file_sha(archive) != ARCHIVE_SHA:
        raise ValueError("Official CIFAR-100 archive mismatch")
    train_path = DATA / "extracted/cifar-100-python/train"
    data, _ = cifar(train_path, 50000)
    ids = np.random.default_rng(20261027).permutation(50000)[:1024].tolist()
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    mean = torch.tensor(formal["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(formal["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    write_new(OUT / "preflight.json", {"status": "PREFLIGHT_PASS", "seed": SEED,
        "protocol_id": "cifar100_50k10k_final_v1", "parent_sha256": {k: v[1] for k, v in EXPECTED.items()},
        "archive_sha256": ARCHIVE_SHA, "train_pickle_sha256": file_sha(train_path),
        "calibration_ids": ids, "calibration_ids_sha256": ids_sha(ids),
        "tau": TAU, "source_sha256": source, "test_accessed": False})
    info, states = {}, {}
    for method, (parent_kind, parent_sha, mode) in EXPECTED.items():
        info[method], states[method] = one_arm(method, parent_kind, parent_sha, mode, data, ids, mean, std)
    if source_sha(formal_path) != source:
        raise RuntimeError("Source changed; refusing Test")
    write_new(OUT / "pre_test_freeze.json", {"status": "BOTH_CHECKPOINTS_FROZEN_BEFORE_TEST",
        "checkpoint_sha256": {m: info[m]["checkpoint_sha256"] for m in EXPECTED},
        "mask_sha256": {m: info[m]["mask_sha256"] for m in EXPECTED},
        "source_sha256": source, "test_accessed": False})
    print("both checkpoints frozen; beginning fixed Final Test", flush=True)
    test_data, test_labels = cifar(DATA / "extracted/cifar-100-python/test", 10000)
    test_sha = file_sha(DATA / "extracted/cifar-100-python/test")
    evaluations = {}
    for method in EXPECTED:
        start = time.perf_counter()
        evaluations[method] = evaluate(states[method], test_data, test_labels, mean, std)
        evaluations[method]["evaluation_seconds"] = time.perf_counter() - start
        print(f"{method} Test accuracy {evaluations[method]['test_accuracy']:.4f}", flush=True)
    write_new(OUT / "results.json", {"status": "COMPLETE_CPU_EXPLORATORY_SEED1", "seed": SEED,
        "protocol_id": "cifar100_50k10k_final_v1", "methods": info, "test": evaluations,
        "test_pickle_sha256": test_sha, "calibration_ids_sha256": ids_sha(ids),
        "source_sha256": source, "test_accessed_during_pruning": False,
        "test_accessed_after_freeze": True, "pytorch": torch.__version__,
        "python": platform.python_version(), "cuda_available": False})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        if not (OUT / "failure.json").exists():
            write_new(OUT / "failure.json", {"status": "FAILED", "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
