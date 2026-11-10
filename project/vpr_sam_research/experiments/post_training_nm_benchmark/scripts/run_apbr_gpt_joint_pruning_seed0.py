"""One-shot dynamic Groupwise OBS pruning for the rebased Joint checkpoint.

Prepared locally in Phase 3.  This script reads CIFAR-100 train calibration
only and deliberately performs no Final Test.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT / "src")]

from baseline_adapters.apbr_gpt_joint import dynamic_joint_obs_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.hessian_cache import build_training_factors, capture_layer_inputs
from joint_optimization.orchestration import AtomicCheckpointManager, audit_nm_masks, write_manifest
from vpr_sam.optim.geometry import Geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-asset", type=Path, required=True)
    parser.add_argument("--calibration-identity", type=Path, required=True)
    parser.add_argument("--train-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pruning runner requires exactly one CUDA device")
    device = torch.device("cuda:0"); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite pruning output")
    manager = AtomicCheckpointManager(output)
    trained = torch.load(args.training_checkpoint, map_location="cpu", weights_only=False)
    initial = torch.load(args.initial_asset, map_location="cpu", weights_only=False)
    identity = json.loads(args.calibration_identity.read_text(encoding="utf-8"))
    if trained.get("epoch") != 200 or trained["stepper"]["steps"] != 78_200:
        raise RuntimeError("Joint parent is not fixed epoch 200")
    with args.train_pickle.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8); labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    ids = np.asarray(identity["batch_ids"], dtype=np.int64)
    x = torch.from_numpy(data[ids].copy()).reshape(-1, 3, 32, 32).float().div_(255)
    mean = torch.tensor((0.507075159237132, 0.486548873314951, 0.440917843367034)).view(1,3,1,1)
    std = torch.tensor((0.267334285879241, 0.256438462917088, 0.276150471325684)).view(1,3,1,1)
    x = ((x - mean) / std).to(device); y = torch.from_numpy(labels[ids].copy()).to(device)
    model = build_cifar100_model().to(device); model.load_state_dict(trained["model"])
    geometry = Geometry(model.named_parameters(), initial["apbr_geometry"]["config"], "vpr", 0)
    geometry.load_state_dict(trained["stepper"]["geometry"])
    param_names = [name for name, _shape in geometry.mapping]

    # Eight global APBR probes from eight disjoint chunks of the fixed train batch.
    probe_by_name = {name: [] for name in param_names}
    model.train()
    for probe_index in range(8):
        begin = probe_index * (len(x) // 8); end = (probe_index + 1) * (len(x) // 8)
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x[begin:end]), y[begin:end]); loss.backward()
        epsilon, _stats = geometry.perturbations(0.05)
        for name, value in zip(param_names, epsilon):
            probe_by_name[name].append(value.detach().clone())
    model.zero_grad(set_to_none=True); model.eval()

    layers = eligible_layers(model)
    captured = capture_layer_inputs(model, layers, x, storage_device=device)
    masks, layer_rows = {}, []
    with torch.no_grad():
        for index, (name, layer) in enumerate(layers.items()):
            training_factors = build_training_factors(layer, captured.pop(name),
                sample_seed=20261101 + index, max_patches=8192, percdamp=0.01,
                compute_dtype=torch.float32)
            probes = torch.stack(probe_by_name[name + ".weight"]).reshape(8, layer.weight.shape[0], -1)
            original = layer.weight.detach().reshape(layer.weight.shape[0], -1)
            result = dynamic_joint_obs_matrix(original, training_factors.factors.damped_hessian,
                probes, blocksize=128, percdamp=0.0, compute_dtype=torch.float32)
            layer.weight.copy_(result.weight.reshape_as(layer.weight))
            masks[name] = result.keep_mask.reshape_as(layer.weight).cpu()
            layer_rows.append({"layer": name, "damp_already_embedded": training_factors.factors.damp,
                "selected_gpt_cost": float(result.selected_gpt_cost.sum()),
                "selected_apbr_cost": float(result.selected_apbr_cost.sum()),
                "decomposition_seconds": result.decomposition_seconds,
                "selection_compensation_seconds": result.selection_compensation_seconds})
    audit = audit_nm_masks(model, layers, {name: mask.to(device) for name, mask in masks.items()})
    payload = {"method": "joint_apbr_gpt_dynamic_groupwise_obs", "model": model.state_dict(),
        "masks": masks, "parent": str(args.training_checkpoint.name), "layer_audit": layer_rows,
        "mask_audit": audit, "recovery_epochs": 0, "test_accessed": False}
    checkpoint_path, checkpoint_sha = manager.save("raw_hard.pt", payload)
    mask_path, mask_sha = manager.save("masks.pt", {"masks": masks, "test_accessed": False})
    write_manifest(output / "checkpoint_freeze.json", {"status": "CHECKPOINTS_FROZEN",
        "checkpoint_sha256": {checkpoint_path.name: checkpoint_sha, mask_path.name: mask_sha},
        "test_accessed": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
