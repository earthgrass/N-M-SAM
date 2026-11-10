"""Phase-3 CPU orchestration smoke; opens CIFAR-100 train only, never Test."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.apbr_gpt_joint import dynamic_joint_obs_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers, magnitude_mask
from joint_optimization.hessian_cache import build_training_factors, capture_layer_inputs
from joint_optimization.orchestration import AtomicCheckpointManager, audit_nm_masks, write_manifest


MEAN = torch.tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1)
STD = torch.tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1)


def tensor_hash(state) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        item = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode()); digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--train-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("Phase-3 smoke is deliberately CPU-only")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite CPU smoke evidence")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    model = build_cifar100_model().cpu(); model.load_state_dict(checkpoint["model"])
    initial_hash = tensor_hash(model.state_dict())
    with args.train_pickle.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,):
        raise ValueError("Official CIFAR-100 train identity failed")
    ids = np.asarray((39328, 14404, 39830, 46380), dtype=np.int64)
    x = torch.from_numpy(data[ids].copy()).reshape(-1, 3, 32, 32).float().div_(255)
    x = (x - MEAN) / STD
    y = torch.from_numpy(labels[ids].copy())

    # One real-model batch: forward, backward, and exactly one optimizer step.
    model.train(); optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad(set_to_none=True); started = time.perf_counter()
    logits = model(x); loss = nn.functional.cross_entropy(logits, y); loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("Non-finite real-model gradient")
    optimizer.step(); training_seconds = time.perf_counter() - started
    if tensor_hash(model.state_dict()) == initial_hash:
        raise RuntimeError("One training iteration did not change the model")

    # Small real-activation Hessian refresh using the untouched initial model.
    hmodel = build_cifar100_model().cpu(); hmodel.load_state_dict(checkpoint["model"])
    layer_map = eligible_layers(hmodel); selected_name = "layer1.0.conv1"
    captured = capture_layer_inputs(hmodel, {selected_name: layer_map[selected_name]}, x[:2])
    factors = build_training_factors(layer_map[selected_name], captured[selected_name],
                                     sample_seed=20261101, max_patches=64)

    # Exercise the complete groupwise algorithm 19 times on deterministic
    # small SPD systems.  This validates network orchestration without
    # pretending that a full 11M-coordinate CPU prune was run.
    generator = torch.Generator().manual_seed(20261101); group_rows = []
    for index, name in enumerate(layer_map):
        width = 8 + 4 * (index % 3); rows = 2 + index % 2
        weight = torch.randn(rows, width, generator=generator)
        a = torch.randn(width, width, generator=generator)
        hessian = a @ a.T + 0.5 * torch.eye(width)
        probes = 0.01 * torch.randn(2, rows, width, generator=generator)
        result = dynamic_joint_obs_matrix(weight, hessian, probes, compute_dtype=torch.float64,
                                          blocksize=4)
        if torch.count_nonzero(result.weight[~result.keep_mask]):
            raise RuntimeError("Groupwise OBS zero invariant failed")
        group_rows.append({"layer": name, "rows": rows, "columns": width,
                           "deleted": int((~result.keep_mask).sum())})

    # Full real-model mask scope/count audit on a disposable model.
    audit_model = build_cifar100_model().cpu(); audit_model.load_state_dict(checkpoint["model"])
    audit_layers = eligible_layers(audit_model)
    masks = {name: magnitude_mask(layer.weight) for name, layer in audit_layers.items()}
    with torch.no_grad():
        for name, layer in audit_layers.items():
            layer.weight.mul_(masks[name])
    mask_audit = audit_nm_masks(audit_model, audit_layers, masks)

    # Atomic checkpoint save/restore includes optimizer and independent RNG.
    manager = AtomicCheckpointManager(output / "checkpoint_roundtrip")
    path, digest = manager.save("smoke.pt", {"model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "completed_steps": 1, "final_test_accessed": False})
    loaded = manager.load(path.name, expected_sha256=digest)
    restored = build_cifar100_model().cpu(); restored.load_state_dict(loaded["model"])
    if tensor_hash(restored.state_dict()) != tensor_hash(model.state_dict()):
        raise RuntimeError("Checkpoint roundtrip changed tensors")

    evidence = {
        "status": "CPU_ORCHESTRATION_PASS", "cpu_only": True,
        "real_train_batch_ids": ids.tolist(), "real_forward_backward": True,
        "one_optimizer_iteration": True, "loss": float(loss.detach()),
        "training_iteration_seconds": training_seconds,
        "small_real_hessian_layer": selected_name,
        "small_real_hessian_sampled_patches": factors.sampled_patches,
        "small_real_hessian_condition_max": factors.factors.condition_max,
        "groupwise_small_network_layers": group_rows,
        "mask_audit": mask_audit, "checkpoint_sha256": digest,
        "checkpoint_roundtrip": True, "final_test_accessed": False,
    }
    write_manifest(output / "cpu_orchestration_evidence.json", evidence)
    print(json.dumps(evidence, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
