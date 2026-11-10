"""Produce the frozen CIFAR-100 ResNet18 identity manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_adapters.cifar100_model import CIFAR100_BLOCKS, CIFAR100_MODEL_ID, build_cifar100_model
from benchmark.model import build_model
from benchmark.nm import CONVENTION, eligible_layers, magnitude_mask, validate_layer_mask

SEED = 20_261_026


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        item = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(item.dtype).encode())
        h.update(str(tuple(item.shape)).encode())
        h.update(memoryview(item.numpy()))
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = build_cifar100_model()
    initialized_hash = state_hash(model)
    state_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    architecture_sha256 = hashlib.sha256(
        json.dumps(state_shapes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([0, 99])
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    gradients_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    c10 = build_model()
    c10_shapes = {name: list(value.shape) for name, value in c10.state_dict().items()}
    shape_differences = {
        name: {"cifar10": c10_shapes[name], "cifar100": state_shapes[name]}
        for name in state_shapes if state_shapes[name] != c10_shapes[name]
    }
    pruning_layers = eligible_layers(model)
    exclusions = {
        "first_conv": "conv1",
        "last_linear": "linear",
        "batch_norm_affine": [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)],
        "all_bias_parameters": [name for name, _ in model.named_parameters() if name.endswith(".bias")],
    }
    probe = torch.tensor([[4.0, 3.0, 2.0, 1.0, 9.0, 9.0]])
    probe_mask = magnitude_mask(probe, n=2, m=4)
    validate_layer_mask(probe, probe_mask, n=2, m=4, check_zeros=False)
    tie_mask = magnitude_mask(torch.ones(1, 4), n=2, m=4)
    checks = {
        "output_shape_batch2_100": list(output.shape) == [2, 100],
        "forward_finite": bool(torch.isfinite(output).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "all_parameter_gradients_present_and_finite": gradients_finite,
        "only_cifar10_shape_differences_are_classifier": set(shape_differences) == {"linear.weight", "linear.bias"},
        "first_and_last_excluded": "conv1" not in pruning_layers and "linear" not in pruning_layers,
        "bn_and_bias_excluded": all(not isinstance(module, nn.BatchNorm2d) for module in pruning_layers.values()),
        "nm_mask_bool": probe_mask.dtype == torch.bool,
        "nm_complete_group_2_of_4": int(probe_mask[0, :4].sum()) == 2,
        "nm_dense_tail": bool(probe_mask[0, 4:].all()),
        "nm_magnitude_tie_stable_lower_indices_kept": tie_mask.tolist() == [[True, True, False, False]],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Model identity checks failed: {checks}")
    data_identity_pass = False
    dataset_link: dict[str, object] = {"audit_path": None, "audit_sha256": None, "archive_sha256": None}
    if args.dataset_audit is not None and args.dataset_audit.exists():
        dataset_bytes = args.dataset_audit.read_bytes()
        dataset_audit = json.loads(dataset_bytes)
        data_identity_pass = dataset_audit.get("status") == "DATA_IDENTITY_PASS"
        dataset_link = {
            "audit_path": str(args.dataset_audit.resolve()),
            "audit_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "archive_sha256": dataset_audit.get("archive", {}).get("sha256"),
        }
    manifest = {
        "status": "CIFAR100_MODEL_IDENTITY_PASS" if data_identity_pass else "MODEL_STRUCTURE_PASS_DATA_IDENTITY_PENDING",
        "structure_status": "PASS",
        "data_identity_prerequisite_pass": data_identity_pass,
        "dataset_identity_link": dataset_link,
        "model_id": CIFAR100_MODEL_ID,
        "model_initialization_seed": SEED,
        "randomly_initialized_not_trained": True,
        "factory": "baseline_adapters.cifar100_model.build_cifar100_model",
        "legacy_cifar10_factory_unchanged": "benchmark.model.build_model",
        "blocks": list(CIFAR100_BLOCKS),
        "stem": {"conv": [3, 64, 3, 3], "stride": 1, "padding": 1, "bias": False, "maxpool": False},
        "basic_block": "two 3x3 convolutions with BN; projection shortcut is 1x1 Conv2d+BN on stride/channel change",
        "stage_channels": [64, 128, 256, 512],
        "stage_strides": [1, 2, 2, 2],
        "pooling": "F.avg_pool2d(kernel_size=4)",
        "classifier": {"in_features": 512, "out_features": 100, "bias": True},
        "initialization": "unmodified torch.nn module reset_parameters defaults; no project custom initializer",
        "torch_version": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_dict_shapes": state_shapes,
        "state_dict_key_order": list(state_shapes),
        "architecture_sha256": architecture_sha256,
        "initialized_state_sha256": initialized_hash,
        "cifar10_to_cifar100_shape_differences": shape_differences,
        "pruning": {
            "convention": CONVENTION,
            "eligible_module_names": list(pruning_layers),
            "excluded": exclusions,
            "n": 2, "m": 4, "complete_group_rule": "keep exactly 2", "tail_rule": "dense",
            "mask_semantics": "bool keep-mask (True means retained)",
            "magnitude_tie_rule": "stable descending argsort; lower contiguous index is retained first on exact ties",
            "sparsegpt_tie_rule": "stable ascending pruning metric; lower contiguous index is removed first on exact ties",
        },
        "checks": checks,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "architecture_sha256": architecture_sha256, "initialized_state_sha256": initialized_hash}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
