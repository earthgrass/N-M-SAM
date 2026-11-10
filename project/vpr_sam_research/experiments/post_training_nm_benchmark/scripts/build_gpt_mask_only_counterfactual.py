"""Build a no-compensation counterfactual from a dense parent and frozen GPT mask."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_apbr_weights_original_sparsegpt_one as common
import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers, validate_layer_mask, validate_masks

PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT = "apbr_sparsegpt_compensation_attribution_geometry_20261031"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--mask-source", choices=("original_sparsegpt", "radius_guided_sparsegpt"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--parent-method", choices=("apbr_sam", "dense_sam"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {output}")
    parent_sha = common.sha_file(args.parent)
    mask_file_sha = common.sha_file(args.mask)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != PROTOCOL or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != args.parent_method or identity.get("seed") != args.seed):
        raise ValueError("Dense parent identity mismatch")
    masks = torch.load(args.mask, map_location="cpu", weights_only=False)
    if not isinstance(masks, dict) or train.tensor_sha(masks) == "":
        raise ValueError("Invalid frozen mask payload")

    model = build_cifar100_model()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if set(masks) != {name + ".weight" for name in layers}:
        raise ValueError("Frozen mask scope mismatch")
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        validate_layer_mask(parameters[name], mask, n=2, m=4, check_zeros=False)
    dense_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for name, mask in masks.items():
            if mask.dtype != torch.bool or mask.shape != parameters[name].shape:
                raise ValueError(f"Mask coordinate mismatch: {name}")
            parameters[name].mul_(mask)
    validate_masks(model, masks, n=2, m=4, complete=True)
    for name, value in model.state_dict().items():
        if name not in masks and not torch.equal(value, dense_state[name]):
            raise ValueError(f"Noneligible model/BN state changed: {name}")

    eligible_positions = sum(mask.numel() for mask in masks.values())
    removed_positions = sum(int((~mask).sum()) for mask in masks.values())
    if eligible_positions != 11_157_504 or removed_positions != 5_578_752:
        raise ValueError("Formal 2:4 counts mismatch")
    payload = {
        "schema": 1, "status": "healthy", "protocol_id": PROTOCOL,
        "experiment_id": EXPERIMENT, "stage": "raw_hard",
        "method": args.mask_source + "_mask_only_counterfactual",
        "seed": args.seed, "parent_method": args.parent_method,
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "masks": masks, "mask_sha256": train.tensor_sha(masks),
        "parent": {"path": str(args.parent), "sha256": parent_sha},
        "mask_source": {"path": str(args.mask), "file_sha256": mask_file_sha},
        "algorithm": {
            "mask_reselected": False, "compensation": False, "training": False,
            "bn_update": False, "recovery_epochs": 0,
        },
        "test_accessed": False,
    }
    checkpoint_sha = common.atomic_torch(output / "mask_only.pt", payload)
    status = {
        "status": "completed", "protocol_id": PROTOCOL, "experiment_id": EXPERIMENT,
        "seed": args.seed, "parent_method": args.parent_method, "mask_source": args.mask_source,
        "parent_sha256": parent_sha, "mask_file_sha256": mask_file_sha,
        "mask_tensor_sha256": payload["mask_sha256"], "checkpoint_sha256": checkpoint_sha,
        "eligible_layers": len(layers), "eligible_positions": eligible_positions,
        "removed_positions": removed_positions, "compensation": False,
        "training": False, "bn_update": False, "test_accessed": False,
    }
    common.atomic_json(output / "status.json", status)
    print(json.dumps(status, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            target = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            common.atomic_json(target / "failure.json", {
                "status": "failed", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
            })
        raise
