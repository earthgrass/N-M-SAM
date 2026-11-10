"""Real rebased-model, train-only, CPU calibration for lambda_p and eta_s."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(PROJECT / "src")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.apbr_gpt_joint import candidate_costs, softmin_pruning_regularizer
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.calibration import calibrate_eta_s, calibrate_lambda_p
from joint_optimization.hessian_cache import build_training_factors, capture_layer_inputs
from joint_optimization.training_objective import differentiable_apbr_perturbations
from vpr_sam.optim.geometry import Geometry
from vpr_sam.optim.step import freeze_bn


def state_hash(items) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        x = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(x.dtype).encode())
        digest.update(str(tuple(x.shape)).encode()); digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def rng_hash(python_state, numpy_state, torch_state: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(repr(python_state).encode())
    digest.update(numpy_state[0].encode())
    digest.update(np.asarray(numpy_state[1], dtype="<u4").tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode())
    digest.update(torch_state.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def independent_augmented_batch(data: np.ndarray, labels: np.ndarray, ids: list[int], config: dict, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.from_numpy(data[np.asarray(ids)].copy()).float().reshape(-1, 3, 32, 32) / 255
    images = F.pad(images, (4, 4, 4, 4), mode="constant", value=0)
    result = torch.empty(len(ids), 3, 32, 32)
    offsets = torch.randint(0, 9, (len(ids), 2), generator=generator)
    flips = torch.rand(len(ids), generator=generator) < 0.5
    for index in range(len(ids)):
        y, x = map(int, offsets[index])
        crop = images[index, :, y:y + 32, x:x + 32]
        result[index] = crop.flip(-1) if bool(flips[index]) else crop
    mean = torch.tensor(config["dataset"]["normalize_mean"])[None, :, None, None]
    std = torch.tensor(config["dataset"]["normalize_std"])[None, :, None, None]
    return (result - mean) / std, torch.from_numpy(labels[np.asarray(ids)].copy()).long()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--batch-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Phase 3 calibration is CPU-only")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError("Refusing nonempty real calibration output")
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    calibration_config = json.loads(args.calibration_config.read_text(encoding="utf-8"))
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if archive.stat().st_size != 169001437 or train.digest_file(archive, "md5") != protocol["dataset"]["archive_md5"] or train.digest_file(archive) != protocol["dataset"]["archive_sha256"]:
        raise ValueError("Official CIFAR-100 archive mismatch")
    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream: raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8); labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    ids_payload = json.loads(args.batch_ids.read_text(encoding="utf-8")); ids = ids_payload["sample_ids"]
    if len(ids) != 128 or len(set(ids)) != 128: raise ValueError("Fixed calibration batch identity")

    initial = torch.load(args.initial_asset, map_location="cpu", weights_only=False)
    if initial.get("cohort") != "apbr_gpt_joint_seed0_rebased": raise ValueError("Wrong initial cohort")
    model = build_cifar100_model().cpu(); model.load_state_dict(initial["model"])
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise RuntimeError("Eligible layer contract")
    geometry = Geometry(model.named_parameters(), dict(protocol["apbr_sam"]["geometry"]), "vpr", 0)
    geometry.load_state_dict(initial["apbr_geometry"])
    params_before = state_hash(model.named_parameters()); buffers_before = state_hash(model.named_buffers())
    geometry_before = state_hash((name, value) for (name, _), value in zip(geometry.mapping, geometry.logs))
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()
    rng_before = rng_hash(python_rng_before, numpy_rng_before, torch_rng_before)
    inputs, targets = independent_augmented_batch(data, labels, ids, protocol, ids_payload["augmentation_seed"])

    captured = capture_layer_inputs(model, layers, inputs)
    factor_cache = {}; factor_rows = []
    for index, (name, layer) in enumerate(layers.items()):
        item = build_training_factors(
            layer, captured.pop(name), sample_seed=20261101 + index,
            max_patches=calibration_config["formal_training"]["max_patches_per_layer"],
            percdamp=calibration_config["formal_training"]["percdamp"], compute_dtype=torch.float32,
        )
        factor_cache[name] = item
        factor_rows.append({"layer": name, "k": layer.weight[0].numel(),
                            "available_patches": item.available_patches, "sampled_patches": item.sampled_patches,
                            "patch_ids_sha256": item.sampled_patch_ids_sha256,
                            "gram_seconds": item.gram_seconds, "factor_seconds": item.factor_seconds,
                            "condition_max": item.factors.condition_max, "damp": item.factors.damp})

    model.train(); model.zero_grad(set_to_none=True)
    center = F.cross_entropy(model(inputs), targets); center.backward()
    first_gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in geometry.params]
    actual_epsilon, epsilon_stats = geometry.perturbations(protocol["training"]["sam"]["rho"])
    geometry.prepare(actual_epsilon, 0, epsilon_stats["zero_perturbation"])
    energy_candidate = geometry.candidate[0]
    if energy_candidate is None: raise RuntimeError("Original APBR energy update unavailable")
    energy_raw = [candidate - scale for candidate, scale in zip(energy_candidate, geometry.logs)]
    geometry.discard()
    saved = [p.detach().clone() for p in geometry.params]
    with torch.no_grad():
        for p, epsilon in zip(geometry.params, actual_epsilon): p.add_(epsilon)
    model.zero_grad(set_to_none=True)
    with freeze_bn(model):
        adversarial_ce = F.cross_entropy(model(inputs), targets)

    scale_proxies = [scale.detach().clone().requires_grad_(True) for scale in geometry.logs]
    proxy_epsilon = differentiable_apbr_perturbations(first_gradients, scale_proxies, rho=protocol["training"]["sam"]["rho"])
    proxy_max_error = max(float((proxy.double() - actual.double()).abs().max().detach()) for proxy, actual in zip(proxy_epsilon, actual_epsilon))
    name_to_index = {name: index for index, (name, _shape) in enumerate(geometry.mapping)}
    eligible_names = tuple(name + ".weight" for name in layers)
    eligible_parameters = [geometry.params[name_to_index[name]] for name in eligible_names]
    layer_regularizers = []; total_blocks = 0; references = {}
    for name, layer in layers.items():
        parameter_index = name_to_index[name + ".weight"]
        base_weight = (geometry.params[parameter_index] - actual_epsilon[parameter_index]).reshape(layer.weight.shape[0], -1)
        probe = proxy_epsilon[parameter_index].reshape_as(base_weight)
        factors = factor_cache[name].factors.inverse_qdd.detach()
        gpt, _apbr = candidate_costs(base_weight.detach(), probe.detach()[None], factors)
        reference = float(gpt.mean())
        if not np.isfinite(reference) or reference <= 0: raise RuntimeError(f"Invalid reference cost: {name}")
        references[name] = reference
        value = softmin_pruning_regularizer(base_weight, probe, factors, reference, temperature=calibration_config["joint_objective"]["temperature"])
        blocks = base_weight.shape[0] * (base_weight.shape[1] // 4)
        layer_regularizers.append(value * blocks); total_blocks += blocks
    regularizer = sum(layer_regularizers) / total_blocks
    ce_gradients = torch.autograd.grad(adversarial_ce, eligible_parameters, retain_graph=True)
    prune_gradients = torch.autograd.grad(regularizer, eligible_parameters, retain_graph=True)
    lambda_rule = calibration_config["lambda_p"]
    lambda_result = calibrate_lambda_p(ce_gradients, prune_gradients, eligible_names,
        target_ratio=lambda_rule["target_gradient_ratio"], lower_bound=lambda_rule["lower_bound"],
        upper_bound=lambda_rule["upper_bound"], ratio_tolerance=lambda_rule["ratio_absolute_tolerance"])
    scale_gradients = torch.autograd.grad(regularizer, scale_proxies)
    eta_rule = calibration_config["eta_s"]
    eta_result = calibrate_eta_s(geometry.logs, energy_raw, scale_gradients, lambda_p=lambda_result.lambda_p,
        target_ratio=eta_rule["target_projected_update_ratio"], relative_tolerance=eta_rule["relative_tolerance"])

    with torch.no_grad():
        for p, before in zip(geometry.params, saved): p.copy_(before)
        for name, buffer in model.named_buffers():
            pass
    # BatchNorm buffers were snapshotted by hash and are restored from initial asset model state.
    model.load_state_dict(initial["model"])
    model.zero_grad(set_to_none=True)
    random.setstate(python_rng_before)
    np.random.set_state(numpy_rng_before)
    torch.set_rng_state(torch_rng_before)
    params_after = state_hash(model.named_parameters()); buffers_after = state_hash(model.named_buffers())
    geometry_after = state_hash((name, value) for (name, _), value in zip(geometry.mapping, geometry.logs))
    # Augmentation and patch sampling use private Generators, so global RNG is
    # expected to remain byte-identical without a restorative mutation.
    rng_after = rng_hash(random.getstate(), np.random.get_state(), torch.get_rng_state())
    if (params_after, buffers_after, geometry_after) != (params_before, buffers_before, geometry_before):
        raise RuntimeError("Real calibration mutated frozen model/BN/geometry")
    if rng_before != rng_after:
        raise RuntimeError("Real calibration changed global RNG state")

    train.atomic_json(output / "lambda_calibration.json", {**lambda_result.to_dict(), "source": "REAL_REBASED_MODEL_TRAIN_BATCH", "regularizer": float(regularizer.detach()), "adversarial_ce": float(adversarial_ce.detach()), "final_test_accessed": False})
    train.atomic_json(output / "eta_s_calibration.json", {**eta_result.to_dict(), "source": "REAL_REBASED_MODEL_TRAIN_BATCH", "proxy_epsilon_max_abs_error": proxy_max_error, "final_test_accessed": False})
    train.atomic_json(output / "calibration_identity.json", {"status": "REAL_CALIBRATION_PASS", "cohort": initial["cohort"], "initial_checkpoint_sha256": train.digest_file(args.initial_asset), "initial_model_tensor_sha256": initial["model_tensor_sha256"], "train_pickle_sha256": train.digest_file(train_file), "batch_ids": ids, "parameters_unchanged": params_before == params_after, "buffers_unchanged": buffers_before == buffers_after, "geometry_unchanged": geometry_before == geometry_after, "rng_unchanged": rng_before == rng_after, "optimizer_constructed": False, "final_test_accessed": False, "references": references})
    with (output / "hessian_refresh_log.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(factor_rows[0])); writer.writeheader(); writer.writerows(factor_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
