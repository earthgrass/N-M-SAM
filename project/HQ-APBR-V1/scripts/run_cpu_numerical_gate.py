"""Write deterministic CPU numerical errors for HQ-APBR Phases 1-2.

Synthetic matrices only. This does not train CIFAR-100, access Test, or time GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hq_apbr_v1 import (
    GeometrySpec, final_obs_keep_mask, hq_global_l2_perturbation,
    historical_geometry_update, inverse_diagonal_cholesky,
    obs_singleton_cost, target_from_inverse_diagonals,
)

OUTPUT = ROOT / "reports/hq_apbr_v1_cpu_numerical_metrics_v2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(value: dict) -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(OUTPUT)


def main() -> None:
    if torch.cuda.is_available():
        raise RuntimeError("This is the CPU-only numerical gate")
    torch.manual_seed(20261105)
    torch.set_num_threads(2)
    a = torch.randn(6, 6, dtype=torch.float64)
    gram = a.T @ a + .25 * torch.eye(6, dtype=torch.float64)
    qdiag, damped, damping = inverse_diagonal_cholesky(gram, percdamp=.01)
    symmetry_correction = float((damped - torch.diag_embed(torch.full((6,), damping, dtype=gram.dtype)) - gram).abs().max())
    direct_q = torch.linalg.inv(damped)
    w = torch.linspace(-.7, .8, 6, dtype=torch.float64)
    i = 2
    k = [j for j in range(6) if j != i]
    delta = torch.empty_like(w)
    delta[i] = -w[i]
    delta[k] = torch.linalg.solve(damped[k][:, k], damped[k, i] * w[i])
    cost_direct = .5 * (delta @ damped @ delta)
    cost_formula = obs_singleton_cost(w[i:i + 1], qdiag[i:i + 1])[0]
    q_error = float((qdiag - direct_q.diag()).abs().max())
    cost_error = float((cost_direct - cost_formula).abs())
    raw_vs_damped_gap = float((cost_direct - .5 * delta @ gram @ delta).abs())

    specs = [
        GeometrySpec("first.weight", (2, 4, 1, 1), 1, torch.tensor([[1., 2., 4., 8.]], dtype=torch.float64)),
        GeometrySpec("second.weight", (6, 1, 1, 2), 2, torch.tensor([[3., 5.], [7., 11.]], dtype=torch.float64)),
    ]
    target, center, coordinates = target_from_inverse_diagonals(specs)
    center_error = float(abs(sum(x.sum() for x in target.values())))
    q_reconstruction_error = float((torch.exp(2 * (target["first.weight"][0, :, 0, 0] + center))
                                    - specs[0].q_diag[0]).abs().max())
    eta = .3
    history = []
    state = {name: torch.zeros_like(value) for name, value in target.items()}
    for factor in (1., 1.5, .7):
        scaled_specs = [GeometrySpec(spec.name, spec.weight_shape, spec.groups, spec.q_diag * factor)
                        for spec in specs]
        point, _, _ = target_from_inverse_diagonals(scaled_specs)
        history.append(point)
        state = historical_geometry_update(state, point, eta)
    explicit = {name: sum(eta * (1 - eta) ** (2 - k) * point[name]
                          for k, point in enumerate(history)) for name in state}
    ema_error = max(float((state[name] - explicit[name]).abs().max()) for name in state)
    ema_center_error = float(abs(sum(value.sum() for value in state.values())))

    params = {"first.weight": torch.empty((2, 4, 1, 1)), "head.bias": torch.empty(2)}
    grads = {"first.weight": torch.arange(1, 9, dtype=torch.float32).reshape(2, 4, 1, 1),
             "head.bias": torch.tensor([.5, -1.])}
    epsilon, actual_norm, zero = hq_global_l2_perturbation(
        params, grads, {"first.weight": state["first.weight"]}, .05)
    radius_relative_error = abs(actual_norm - .05) / .05
    # Direct-Q identity holds in the eligible-only normalized subspace;
    # the noneligible coordinates alter the full-model normalization.
    eligible = grads["first.weight"].double() * torch.exp(2 * state["first.weight"])
    direct_eligible = .05 * eligible / eligible.norm()
    eligible_actual = .05 * epsilon["first.weight"].double() / epsilon["first.weight"].double().norm()
    eligible_direction_error = float((eligible_actual - direct_eligible).abs().max())
    mask = final_obs_keep_mask(torch.tensor([[1., 1., 1., 1., 7.]]), torch.ones(5))
    mask_correct = mask.tolist() == [[False, False, True, True, True]]
    passed = (q_error <= 1e-12 and cost_error <= 1e-12 and raw_vs_damped_gap > 1e-7
              and symmetry_correction <= 1e-12
              and center_error <= 1e-12 and q_reconstruction_error <= 1e-12
              and ema_error <= 1e-12 and ema_center_error <= 1e-12
              and radius_relative_error <= 2e-5 and eligible_direction_error <= 1e-8
              and mask_correct and not zero and coordinates == 20)
    report = {
        "status": "CPU_NUMERICAL_GATE_PASS" if passed else "CPU_NUMERICAL_GATE_FAIL",
        "synthetic_only": True, "test_accessed": False, "cuda_available": False,
        "python": platform.python_version(), "pytorch": torch.__version__,
        "source_sha256": {"math_core": digest(ROOT / "hq_apbr_v1/math_core.py"),
                          "gate_runner": digest(Path(__file__))},
        "test_a": {"inverse_diag_max_abs_error": q_error,
                   "obs_cost_abs_error": cost_error,
                   "undamped_vs_damped_cost_gap": raw_vs_damped_gap,
                   "symmetrization_max_abs_correction": symmetry_correction,
                   "relative_damping": damping},
        "test_b_d": {"global_center_sum_abs_error": center_error,
                     "exp_2u_reconstruction_max_abs_error": q_reconstruction_error,
                     "eligible_coordinate_count_in_toy": coordinates},
        "test_c": {"ema_explicit_max_abs_error": ema_error,
                   "ema_center_sum_abs_error": ema_center_error},
        "test_e_f": {"global_radius_relative_error": radius_relative_error,
                     "eligible_direction_max_abs_error": eligible_direction_error,
                     "zero_direction": zero},
        "test_g": {"stable_2_of_4_with_dense_tail": mask_correct},
        "phase_3_gpu_gate": "NOT_RUN_CPU_ONLY",
    }
    write_once(report)
    if not passed:
        raise AssertionError("Numerical tolerances failed; see metrics JSON")
    print(json.dumps({"status": report["status"], "errors": report["test_a"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if not OUTPUT.exists():
            write_once({"status": "CPU_NUMERICAL_GATE_FAIL", "traceback": traceback.format_exc(),
                        "test_accessed": False, "cuda_available": False})
        raise
