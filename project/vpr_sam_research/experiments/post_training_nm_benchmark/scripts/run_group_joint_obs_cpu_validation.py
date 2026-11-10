"""Deterministic CPU-only numerical gate for Group-level Joint OBS.

This runner uses synthetic positive-definite systems only.  It never loads
CIFAR data, a Test split, a checkpoint, CUDA, or a remote resource.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.fixedmask_obs import fixedmask_obs_matrix
from baseline_adapters.group_joint_obs import (
    APBR_SHUFFLE_SEEDS,
    DELETION_PAIRS,
    apbr_pair_costs,
    prepare_obs_system,
    select_group_mask,
    static_group_obs_costs,
)
from baseline_adapters.sparsegpt import sparsegpt_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "cpu_validation_results.json"
    candidate_path = output / "candidate_costs.csv"
    if result_path.exists() or candidate_path.exists():
        raise FileExistsError("CPU validation outputs already exist")

    seed = 20261031
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, columns = 3, 8
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    root = torch.randn(columns, columns, generator=generator, dtype=torch.float64)
    hessian = root.T @ root + 0.3 * torch.eye(columns, dtype=torch.float64)
    geometry = 0.1 * torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    system = prepare_obs_system(hessian, percdamp=0.01, compute_dtype=torch.float64)
    direct_inverse = torch.linalg.inv(system.damped_hessian)
    inverse_max_abs = float((system.inverse_hessian - direct_inverse).abs().max())
    inverse_relative = float((system.inverse_hessian - direct_inverse).norm() / direct_inverse.norm())
    factor_reconstruction = system.inverse_factor.T @ system.inverse_factor
    factor_max_abs = float((factor_reconstruction - direct_inverse).abs().max())
    factor_relative = float((factor_reconstruction - direct_inverse).norm() / direct_inverse.norm())

    costs, conditions = static_group_obs_costs(weight, system.inverse_hessian)
    max_cost_error = 0.0
    max_constraint_error = 0.0
    max_qdd_error = 0.0
    candidate_rows = []
    for row in range(rows):
        for group in range(columns // 4):
            for candidate, pair in enumerate(DELETION_PAIRS):
                indices = torch.tensor([4 * group + item for item in pair])
                qdd = system.inverse_hessian[indices][:, indices]
                reference_qdd = direct_inverse[indices][:, indices]
                max_qdd_error = max(max_qdd_error, float((qdd - reference_qdd).abs().max()))
                deleted_weight = weight[row, indices]
                multiplier = torch.linalg.solve(reference_qdd, deleted_weight)
                delta = -direct_inverse[:, indices] @ multiplier
                objective = 0.5 * delta @ system.damped_hessian @ delta
                cost_error = abs(float(costs[row, group, candidate] - objective))
                max_cost_error = max(max_cost_error, cost_error)
                constraint_error = float((delta[indices] + deleted_weight).abs().max())
                max_constraint_error = max(max_constraint_error, constraint_error)
                candidate_rows.append({
                    "row": row,
                    "group": group,
                    "candidate_index": candidate,
                    "deleted_local_i": pair[0],
                    "deleted_local_j": pair[1],
                    "joint_obs_cost": float(costs[row, group, candidate]),
                    "explicit_objective": float(objective),
                    "absolute_error": cost_error,
                    "qdd_condition": float(conditions[group, candidate]),
                })

    apbr_cost = apbr_pair_costs(weight, geometry)
    max_apbr_error = 0.0
    for row in range(rows):
        for group in range(columns // 4):
            for candidate, pair in enumerate(DELETION_PAIRS):
                expected = sum(
                    weight[row, 4 * group + item].square()
                    * torch.exp(-2 * geometry[row, 4 * group + item])
                    for item in pair
                )
                max_apbr_error = max(
                    max_apbr_error, abs(float(apbr_cost[row, group, candidate] - expected))
                )

    # Frozen near-optimal and C_min=0 cases.
    near_weight = torch.tensor([[1.0, 1.0, 1.004, 10.0]], dtype=torch.float64)
    identity_system = prepare_obs_system(torch.eye(4, dtype=torch.float64), percdamp=0.0)
    near_geometry = torch.tensor([[0.0, 0.0, 4.0, 0.0]], dtype=torch.float64)
    group_only = select_group_mask(near_weight, identity_system)
    zero_tau = select_group_mask(near_weight, identity_system, log_radius=near_geometry, tau=0.0)
    hybrid = select_group_mask(near_weight, identity_system, log_radius=near_geometry, tau=0.01)
    zero_weight = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    zero_result = select_group_mask(
        zero_weight, identity_system, log_radius=torch.zeros_like(zero_weight), tau=0.02
    )

    shuffle_a = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[0])
    shuffle_a_repeat = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[0])
    shuffle_b = apbr_pair_costs(weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[1])
    # With nonuniform weights, reassigning geometry legitimately changes APBR
    # costs.  Unit weights isolate the geometry multiset: summing all six pair
    # costs counts every block coordinate exactly three times.
    unit_weight = torch.ones_like(weight)
    unit_original = apbr_pair_costs(unit_weight, geometry)
    unit_shuffled = apbr_pair_costs(
        unit_weight, geometry, shuffled_seed=APBR_SHUFFLE_SEEDS[0]
    )
    shuffle_sum_error = float((unit_shuffled.sum(-1) - unit_original.sum(-1)).abs().max())

    mask_result = select_group_mask(weight, system, log_radius=geometry, tau=0.02)
    strict_2of4 = bool((mask_result.keep_mask.reshape(-1, 4).sum(1) == 2).all())

    replay = {}
    for dtype in (torch.float32, torch.float64):
        native = sparsegpt_matrix(
            weight, hessian, blocksize=8, percdamp=0.01, compute_dtype=dtype
        )
        fixed = fixedmask_obs_matrix(
            weight, hessian, native.keep_mask,
            blocksize=8, percdamp=0.01, compute_dtype=dtype,
        )
        difference = fixed.weight - native.weight
        replay[str(dtype).removeprefix("torch.")] = {
            "mask_equal": bool(torch.equal(fixed.keep_mask, native.keep_mask)),
            "weight_bitwise_equal": bool(torch.equal(fixed.weight, native.weight)),
            "max_absolute_error": float(difference.abs().max()),
            "relative_l2_error": float(difference.norm() / native.weight.norm()),
            "deleted_exact_zero": bool((fixed.weight[~fixed.keep_mask] == 0).all()),
            "survivor_changed_count": fixed.survivor_changed_count,
        }

    ill_conditioned_hard_failure = False
    try:
        static_group_obs_costs(
            torch.ones(1, 4, dtype=torch.float64),
            torch.diag(torch.tensor([1e-12, 1.0, 1.0, 1.0], dtype=torch.float64)),
        )
    except RuntimeError:
        ill_conditioned_hard_failure = True

    checks = {
        "six_candidates": len(DELETION_PAIRS) == 6 and len(set(DELETION_PAIRS)) == 6,
        "inverse_matches_direct": inverse_max_abs < 1e-12,
        "upper_factor_reconstructs_q": factor_max_abs < 1e-12,
        "qdd_matches_direct_inverse": max_qdd_error < 1e-12,
        "joint_cost_matches_explicit_optimum": max_cost_error < 1e-11,
        "constraints_satisfied": max_constraint_error < 1e-11,
        "apbr_cost_matches_scalar_reference": max_apbr_error < 1e-12,
        "tau_zero_degenerates": (
            group_only.selected_pair_index.item() == zero_tau.selected_pair_index.item() == 0
        ),
        "near_optimal_set_correct": (
            hybrid.candidate_count.item() == 3
            and hybrid.selected_pair_index.item() == 1
            and hybrid.relative_gpt_increase.item() <= 0.01 + 1e-12
        ),
        "zero_minimum_not_artificially_expanded": (
            zero_result.candidate_count.item() == 1
            and zero_result.selected_pair_index.item() == 0
        ),
        "shuffle_deterministic": bool(torch.equal(shuffle_a, shuffle_a_repeat)),
        "shuffle_seed_changes_mapping": not bool(torch.equal(shuffle_a, shuffle_b)),
        "shuffle_preserves_block_multiset": shuffle_sum_error < 1e-12,
        "strict_2of4": strict_2of4,
        "all_finite": bool(
            torch.isfinite(costs).all()
            and torch.isfinite(apbr_cost).all()
            and torch.isfinite(mask_result.selected_gpt_cost).all()
        ),
        "native_replay_fp32_bitwise": replay["float32"]["weight_bitwise_equal"],
        "native_replay_fp64_bitwise": replay["float64"]["weight_bitwise_equal"],
        "ill_conditioned_qdd_hard_failure": ill_conditioned_hard_failure,
    }
    status = "CPU_NUMERICAL_GATE_PASS" if all(checks.values()) else "CPU_NUMERICAL_GATE_FAIL"
    evidence = {
        "status": status,
        "seed": seed,
        "synthetic_only": True,
        "device": "cpu",
        "cuda_used": False,
        "dataset_loaded": False,
        "test_accessed": False,
        "server_contacted": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "percdamp": 0.01,
            "blocksize": 8,
            "keep_n": 2,
            "m": 4,
            "condition_limit": 1e8,
            "pair_order": list(DELETION_PAIRS),
            "shuffle_seeds": list(APBR_SHUFFLE_SEEDS),
        },
        "errors": {
            "inverse_max_absolute": inverse_max_abs,
            "inverse_relative_l2": inverse_relative,
            "factor_reconstruction_max_absolute": factor_max_abs,
            "factor_reconstruction_relative_l2": factor_relative,
            "qdd_max_absolute": max_qdd_error,
            "joint_cost_max_absolute": max_cost_error,
            "constraint_max_absolute": max_constraint_error,
            "apbr_cost_max_absolute": max_apbr_error,
            "shuffle_pair_sum_max_absolute": shuffle_sum_error,
        },
        "native_mask_replay": replay,
        "checks": checks,
        "source_sha256": {
            "adapter": sha256(ROOT / "baseline_adapters" / "group_joint_obs" / "adapter.py"),
            "runner": sha256(Path(__file__)),
            "sparsegpt_adapter": sha256(ROOT / "baseline_adapters" / "sparsegpt" / "adapter.py"),
            "fixedmask_adapter": sha256(ROOT / "baseline_adapters" / "fixedmask_obs" / "adapter.py"),
        },
    }
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    write_new(result_path, json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(evidence, ensure_ascii=False, allow_nan=False))
    if status != "CPU_NUMERICAL_GATE_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
