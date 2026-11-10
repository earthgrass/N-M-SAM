"""Deterministic CPU-only interface calibration; never opens CIFAR/Test data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from joint_optimization.calibration.calibration_report import write_summary
from joint_optimization.calibration.eta_scale_calibrator import calibrate_eta_s
from joint_optimization.calibration.lambda_calibrator import calibrate_lambda_p


def digest_tensors(values) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["phase"] != "local_cpu_calibration" or config["final_test_access"] is not False:
        raise ValueError("Wrong calibration phase")
    if torch.cuda.is_available():
        raise RuntimeError("This runner is deliberately CPU-only")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    shapes = ((8, 16), (4, 12), (3, 8))
    names = tuple(f"synthetic.layer{index}.weight" for index in range(len(shapes)))
    weights = [torch.randn(shape, generator=generator, dtype=torch.float32) for shape in shapes]
    before = digest_tensors(zip(names, weights))
    ce = [torch.randn(shape, generator=generator, dtype=torch.float32) for shape in shapes]
    prune = [0.2 * torch.randn(shape, generator=generator, dtype=torch.float32) for shape in shapes]
    rule = config["lambda_p"]
    lambda_result = calibrate_lambda_p(
        ce, prune, names, target_ratio=rule["target_gradient_ratio"],
        lower_bound=rule["lower_bound"], upper_bound=rule["upper_bound"],
        ratio_tolerance=rule["ratio_absolute_tolerance"],
    )

    scales = [torch.zeros(shape, dtype=torch.float64) for shape in shapes]
    epsilon = [torch.randn(shape, generator=generator, dtype=torch.float64) for shape in shapes]
    norm = sum(item.square().sum() for item in epsilon).sqrt()
    epsilon = [item * (config["formal_training"]["rho"] / norm) for item in epsilon]
    total_coordinates = sum(item.numel() for item in epsilon)
    denominator = sum(item.square().sum() for item in epsilon)
    gamma = 5.482145661475941e-8
    energy = [gamma * (1 - total_coordinates * item.square() / denominator) for item in epsilon]
    scale_gradients = [torch.randn(shape, generator=generator, dtype=torch.float64) for shape in shapes]
    eta_rule = config["eta_s"]
    eta_result = calibrate_eta_s(
        scales, energy, scale_gradients, lambda_p=lambda_result.lambda_p,
        target_ratio=eta_rule["target_projected_update_ratio"],
        relative_tolerance=eta_rule["relative_tolerance"],
    )
    after = digest_tensors(zip(names, weights))
    if before != after:
        raise RuntimeError("Calibration mutated synthetic model parameters")

    source = "SYNTHETIC_INTERFACE_ONLY"
    lambda_payload = {**lambda_result.to_dict(), "calibration_source": source,
                      "model_sha256_before": before, "model_sha256_after": after,
                      "optimizer_step": False, "bn_updated": False,
                      "final_test_accessed": False}
    eta_payload = {**eta_result.to_dict(), "calibration_source": source,
                   "scale_state_mutated": False, "final_test_accessed": False}
    atomic_json(output / "lambda_calibration.json", lambda_payload)
    atomic_json(output / "eta_s_calibration.json", eta_payload)
    write_summary(output / "calibration_summary.md", lambda_payload, eta_payload, source=source)
    with (output / "calibration_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value", "acceptance"))
        writer.writeheader()
        writer.writerows([
            {"metric": "lambda_p", "value": lambda_result.lambda_p, "acceptance": "1e-6_to_1e6"},
            {"metric": "lambda_gradient_ratio", "value": lambda_result.achieved_ratio, "acceptance": "0.05_plusminus_5e-5"},
            {"metric": "eta_s", "value": eta_result.eta_s, "acceptance": "finite_positive"},
            {"metric": "eta_projected_update_ratio", "value": eta_result.achieved_ratio, "acceptance": "0.05_relative_0.1pct"},
            {"metric": "model_unchanged", "value": before == after, "acceptance": "true"},
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
