"""Train-only, one-layer APBR geometry tie-break pilot on a frozen Dense parent.

This is an exploratory CPU diagnostic. It never opens the CIFAR-100 Test file.
All arms use the same Dense weights, train calibration Hessian, and physical
joint OBS compensation; only near-optimal selection tie-breaks differ.
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
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.apbr_nearoptimal_conditional_obs import apbr_nearoptimal_conditional_obs_prune
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.common.nm import validate_keep_mask
from baseline_adapters.exactobs_reference import run_official_exactobs_row
from baseline_adapters.sparsegpt import SparseGPTAdapter
from benchmark.nm import eligible_layers
from scripts.audit_apbr_gpt_mask_match_local import PAIRS, file_sha, tensor_sha
from scripts.run_apbr_gpt_fixedmask_cpu_full import DATA, cifar, normalized


OUTPUT = ROOT / "artifacts/apbr_nearoptimal_conditional_seed0_train_pilot_20261103_v2"
REPORT = ROOT / "docs/research/apbr_nearoptimal_conditional_seed0_20261103_v2"
PARENT_SHA256 = "4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82"
ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
LAYER = "layer1.0.conv1"
SELECTION_SEED = 20261027
SHUFFLE_SEED = 20261103
SAMPLES = 1024
TAU = 0.01
METHODS = ("physical", "unit", "true", "shuffled")


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


def local_quadratic(w: torch.Tensor, candidate: torch.Tensor, h: torch.Tensor) -> float:
    delta = (w - candidate).double()
    return float(.5 * torch.einsum("ri,ij,rj->", delta, h.double(), delta))


def train_diagnostic(model: nn.Module, data: np.ndarray, labels: np.ndarray,
                     ids: list[int], mean: torch.Tensor, std: torch.Tensor) -> dict:
    correct = 0
    ce_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            batch_ids = ids[start:start + 128]
            logits = model(normalized(data, batch_ids, mean, std))
            targets = torch.from_numpy(labels[np.asarray(batch_ids)].copy())
            ce_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(1) == targets).sum())
    return {"train_diagnostic_count": len(ids),
            "train_diagnostic_accuracy": correct / len(ids),
            "train_diagnostic_ce": ce_sum / len(ids)}


def source_manifest(formal_path: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__),
        "new_adapter": ROOT / "baseline_adapters/apbr_nearoptimal_conditional_obs/adapter.py",
        "old_conditional_adapter": ROOT / "baseline_adapters/conditional_greedy_obs/adapter.py",
        "hessian_adapter": ROOT / "baseline_adapters/sparsegpt/adapter.py",
        "model": ROOT / "baseline_adapters/cifar100_model.py",
        "eligibility": ROOT / "benchmark/nm.py",
        "official_exactobs_reference": ROOT / "baseline_adapters/exactobs_reference.py",
        "official_exactobs_wrapper": ROOT / "benchmark/obc.py",
        "pinned_trueobs": ROOT / "artifacts/upstream/OBC/trueobs.py",
        "frozen_formal_config": formal_path,
    }
    return {name: file_sha(path) for name, path in files.items()}


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("This diagnostic was frozen as a local CPU-only pilot")
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Pilot outputs already exist; never overwrite an earlier result")
    OUTPUT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    torch.set_num_threads(min(torch.get_num_threads(), 16))

    parent_path = PAIRS["apbr_vpr_seed0"]["parent"]
    raw_path = PAIRS["apbr_vpr_seed0"]["raw"]
    formal_path = ROOT / "configs/cifar100_50k10k_final_v1.yaml"
    train_path = DATA / "extracted/cifar-100-python/train"
    archive_path = DATA / "cifar-100-python.tar.gz"
    if file_sha(parent_path) != PARENT_SHA256:
        raise ValueError("Frozen APBR-VPR Seed0 parent SHA256 mismatch")
    if archive_path.stat().st_size != 169001437 or file_sha(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Official CIFAR-100 archive mismatch")
    source_sha = source_manifest(formal_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (parent.get("protocol_id") != "cifar100_50k10k_final_v1"
            or parent.get("epoch") != 200 or parent.get("steps") != 78200
            or parent.get("test_accessed") is not False
            or parent.get("identity", {}).get("method") != "apbr_sam"
            or parent.get("identity", {}).get("seed") != 0):
        raise ValueError("Parent training identity mismatch")
    model = build_cifar100_model().cpu().eval()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Frozen eligible-layer identity mismatch")
    layer = layers[LAYER]
    geometry = parent["stepper"]["geometry"]
    mapping = [(name, tuple(shape)) for name, shape in geometry["mapping"]]
    if geometry.get("method") != "vpr" or mapping != [(name, tuple(p.shape)) for name, p in model.named_parameters()]:
        raise ValueError("Geometry mapping does not match Dense checkpoint")
    logs = {name: value for (name, _), value in zip(mapping, geometry["logs"])}
    s_tensor = logs[LAYER + ".weight"]
    if s_tensor.shape != layer.weight.shape or s_tensor.dtype != torch.float64 or not bool(torch.isfinite(s_tensor).all()):
        raise ValueError("APBR log-scale tensor identity mismatch")
    s = weight_to_matrix(s_tensor).contiguous()
    # A contiguous view still aliases the model parameter. Each diagnostic
    # temporarily swaps the layer weight, so the immutable Dense reference
    # must own separate storage.
    w = weight_to_matrix(layer.weight.detach()).clone()
    if w.data_ptr() == layer.weight.data_ptr():
        raise RuntimeError("Dense reference weight aliases the live model")
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    mean = torch.tensor(formal["dataset"]["normalize_mean"]).reshape(1, 3, 1, 1)
    std = torch.tensor(formal["dataset"]["normalize_std"]).reshape(1, 3, 1, 1)
    train_data, train_labels = cifar(train_path, 50000)
    permutation = np.random.default_rng(SELECTION_SEED).permutation(50000)
    calibration_ids = permutation[:SAMPLES].tolist()
    diagnostic_ids = permutation[SAMPLES:2 * SAMPLES].tolist()
    if set(calibration_ids) & set(diagnostic_ids):
        raise ValueError("Calibration/diagnostic train IDs overlap")
    historical_raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    if historical_raw["calibration_ids"] != calibration_ids or historical_raw["parent"]["sha256"] != PARENT_SHA256:
        raise ValueError("Historical calibration or parent mismatch")
    preflight = {
        "status": "PREFLIGHT_PASS", "parent_kind": "apbr_vpr_seed0",
        "parent_sha256": PARENT_SHA256, "historical_raw_sha256": file_sha(raw_path),
        "archive_sha256": ARCHIVE_SHA256, "train_pickle_sha256": file_sha(train_path),
        "source_sha256": source_sha, "protocol_id": parent["protocol_id"],
        "layer": LAYER, "weight_shape": list(layer.weight.shape),
        "geometry_sha256": tensor_sha({LAYER + ".weight": s_tensor}),
        "calibration_ids": calibration_ids, "calibration_ids_sha256": ids_sha(calibration_ids),
        "diagnostic_ids": diagnostic_ids, "diagnostic_ids_sha256": ids_sha(diagnostic_ids),
        "tau": TAU, "shuffle_seed": SHUFFLE_SEED, "damping": 0.01,
        "blocksize": 128, "compute_dtype": "float64", "hessian_dtype": "float32",
        "cpu_only": True, "test_accessed": False,
    }
    write_new(OUTPUT / "preflight.json", preflight)

    adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
    hook = layer.register_forward_pre_hook(lambda _module, inputs: adapter.add_batch(inputs[0]))
    calibration_start = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(calibration_ids), 128):
                batch = calibration_ids[start:start + 128]
                model(normalized(train_data, batch, mean, std))
                print(f"train calibration {start + len(batch)}/{SAMPLES}", flush=True)
    finally:
        hook.remove()
    calibration_seconds = time.perf_counter() - calibration_start
    h = adapter.hessians()[0].cpu()
    if h.shape != (w.shape[1], w.shape[1]) or not bool(torch.isfinite(h).all()):
        raise ValueError("Invalid layer Hessian")
    h_sha = tensor_sha({"hessian": h})
    dense_diagnostic = train_diagnostic(model, train_data, train_labels, diagnostic_ids, mean, std)
    baseline_error = float(.5 * torch.einsum("ri,ij,rj->", w.double(), h.double(), w.double()))
    if baseline_error <= 0:
        raise ValueError("Nonpositive dense quadratic baseline")

    results = {}
    masks = {}
    for method in METHODS:
        start = time.perf_counter()
        result = apbr_nearoptimal_conditional_obs_prune(
            w, h, log_scale=s, tau=TAU, geometry_mode=method,
            shuffle_seed=SHUFFLE_SEED if method == "shuffled" else None,
            blocksize=128, percdamp=0.01,
        )
        elapsed = time.perf_counter() - start
        keep = result.keep_mask.detach().cpu()
        candidate = result.weight.detach().cpu()
        validate_keep_mask(keep, 2, 4)
        if (keep.dtype != torch.bool or int((~keep).sum()) != w.numel() // 2
                or not bool(torch.isfinite(candidate).all()) or int(torch.count_nonzero(candidate[~keep]))):
            raise ValueError(f"Invalid 2:4 compensated result: {method}")
        if source_manifest(formal_path) != source_sha:
            raise RuntimeError("Source changed during pilot")
        local_error = local_quadratic(w, candidate, h)
        layer_weight = candidate.reshape_as(layer.weight).contiguous()
        output_path = OUTPUT / f"{method}_layer.pt"
        write_new(output_path, {
            "status": "FROZEN_TRAIN_ONLY", "parent_sha256": PARENT_SHA256,
            "method": method, "layer": LAYER, "weight": layer_weight,
            "keep_mask": keep.reshape_as(layer.weight).contiguous(),
            "tau": TAU, "hessian_sha256": h_sha,
            "calibration_ids_sha256": ids_sha(calibration_ids),
            "test_accessed": False, "recovery_epochs": 0,
        })
        with torch.no_grad():
            layer.weight.copy_(layer_weight)
        diagnostic = train_diagnostic(model, train_data, train_labels, diagnostic_ids, mean, std)
        with torch.no_grad():
            layer.weight.copy_(w.reshape_as(layer.weight))
        first_counts = result.first_candidate_counts.detach().cpu()
        second_counts = result.second_candidate_counts.detach().cpu()
        masks[method] = keep
        results[method] = {
            "layer_checkpoint_sha256": file_sha(output_path),
            "mask_sha256": tensor_sha({LAYER + ".weight": keep.reshape_as(layer.weight)}),
            "physical_quadratic_error": local_error,
            "physical_relative_quadratic_error": local_error / baseline_error,
            "train_diagnostic": diagnostic,
            "near_optimal_first_gt1": int((first_counts > 1).sum()),
            "near_optimal_second_gt1": int((second_counts > 1).sum()),
            "first_candidate_count_mean": float(first_counts.float().mean()),
            "second_candidate_count_mean": float(second_counts.float().mean()),
            "first_choices_sha256": tensor_sha({"choices": result.first_choices}),
            "second_choices_sha256": tensor_sha({"choices": result.second_choices}),
            "decomposition_seconds": result.decomposition_seconds,
            "selection_compensation_seconds": result.selection_compensation_seconds,
            "wall_seconds": elapsed,
            "deleted": int((~keep).sum()),
        }
        print(f"{method}: train diagnostic {diagnostic['train_diagnostic_accuracy']:.4f}, "
              f"physical error {local_error:.4g}", flush=True)
    if not torch.equal(layer.weight.detach(), w.reshape_as(layer.weight)):
        raise RuntimeError("Model weight not restored after diagnostic")
    for method in METHODS[1:]:
        results[method]["mask_flips_vs_physical"] = int((masks[method] != masks["physical"]).sum())
        results[method]["mask_flips_vs_unit"] = int((masks[method] != masks["unit"]).sum())
    exact_reference = run_official_exactobs_row(layer, h, 0, relative_damping=0.01)
    exact_receipt = {
        "scope": "output_row_0_only", "algorithm": "pinned_official_TrueOBS",
        "comparison_limit": "official TrueOBS globally interleaves groups; conditional pilot processes quartets in fixed order",
        "physical_quadratic_cost": exact_reference.physical_quadratic_cost,
        "damped_quadratic_cost": exact_reference.damped_quadratic_cost,
        "runtime_seconds": exact_reference.runtime_seconds,
        "relative_damping": exact_reference.relative_damping,
        "damping_value": exact_reference.damping_value,
        "cholesky_info": exact_reference.cholesky_info,
        "dead_columns": exact_reference.dead_columns,
        "fallback_used": exact_reference.fallback_used,
        "mask_sha256": tensor_sha({"row0": exact_reference.keep_mask}),
        "cost_by_pilot_method_row0": {
            method: local_quadratic(w[0:1], torch.load(OUTPUT / f"{method}_layer.pt", map_location="cpu", weights_only=False)["weight"].reshape_as(w)[0:1], h)
            for method in METHODS
        },
        "mask_flips_by_pilot_method_row0": {
            method: int((exact_reference.keep_mask != masks[method][0]).sum()) for method in METHODS
        },
    }
    write_new(OUTPUT / "official_exactobs_row0.json", exact_receipt)
    summary = {
        "status": "CPU_SINGLE_LAYER_TRAIN_DIAGNOSTIC_COMPLETE",
        "not_full_model": True, "not_final_test": True, "test_accessed": False,
        "parent_sha256": PARENT_SHA256, "parent_geometry_method": "vpr",
        "layer": LAYER, "K": w.shape[1], "rows": w.shape[0],
        "group_count": w.numel() // 4, "tau": TAU,
        "calibration_seconds": calibration_seconds,
        "dense_train_diagnostic": dense_diagnostic,
        "hessian_sha256": h_sha, "source_sha256": source_sha,
        "official_exactobs_row0": exact_receipt,
        "methods": results,
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    write_new(OUTPUT / "results.json", summary)
    report_lines = [
        "# APBR near-optimal conditional OBS: Seed 0 train-only pilot", "",
        "v1 因稠密权重视图与模型参数共享存储而作废；该失败产物保留，v2 使用独立复制的稠密权重。", "",
        "本次是 APBR-VPR Seed 0 同父模型的真实 CIFAR-100 单层 CPU 诊断；仅使用官方 Train，未读取 Test。",
        f"层：`{LAYER}`，K={w.shape[1]}；冻结的近优窗口 τ={TAU:.2%}。",
        "四臂共用物理 Hessian 和联合 OBS 补偿，只改变近优窗口内的选择。",
        "`physical` 直接取最低物理 OBS 分数；`unit` 用 |w|² 裁决；`true` 用 |w exp(-s)|²；`shuffled` 在每个四元组内固定打乱 s。",
        "第二次选择使用虚拟删除后的权重与条件逆曲率。各步近优界限不保证整对总代价或最终准确率的相同界限。", "",
        "| 选择 | 训练侧诊断准确率 | CE | 物理二次误差 | 相对 physical Mask 翻转位置 | 两步近优候选>1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = results[method]
        diag = item["train_diagnostic"]
        report_lines.append(
            f"| {method} | {diag['train_diagnostic_accuracy']:.2%} | {diag['train_diagnostic_ce']:.4f} | "
            f"{item['physical_quadratic_error']:.6g} | {item.get('mask_flips_vs_physical', 0):,} | "
            f"{item['near_optimal_first_gt1']:,}/{item['near_optimal_second_gt1']:,} |"
        )
    report_lines += [
        "", f"稠密模型在同一训练诊断子集：{dense_diagnostic['train_diagnostic_accuracy']:.2%}，CE={dense_diagnostic['train_diagnostic_ce']:.4f}。",
        "本机无 CUDA；没有全模型准确率或 GPU 资源结论。APBR-VPR 的结果也不能代替 APBR-L2 结论。",
        "真实几何优于打乱或单位几何的判断须同时看训练侧分类指标、物理重建误差和多层/多 Seed 稳定性。",
        "附加一条输出行的锁定官方 TrueOBS 参照；它跨组全行选择，和本 Pilot 的固定四元组处理顺序不同，不能直接归因方法优劣。",
        f"证据：`{OUTPUT}`。", "",
    ]
    report_path = REPORT / "PILOT_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    digest_lines = []
    for path in sorted((*OUTPUT.iterdir(), report_path)):
        if path.is_file():
            digest_lines.append(f"{file_sha(path)}  {path.relative_to(ROOT)}")
    (REPORT / "SHA256SUMS.txt").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report": str(report_path),
                      "methods": {m: results[m]["train_diagnostic"] for m in METHODS}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        failure_path = OUTPUT / "failure.json"
        if not failure_path.exists():
            write_new(failure_path, {"status": "FAILED", "traceback": traceback.format_exc(),
                                     "test_accessed": False})
        raise
