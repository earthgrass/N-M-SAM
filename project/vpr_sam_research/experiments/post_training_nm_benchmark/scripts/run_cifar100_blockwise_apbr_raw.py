"""Generate the frozen APBR geometry 2:4 Raw-Hard endpoint; no Recovery/OBS."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as formal
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers, project, validate_masks

EXPERIMENT = "apbr_blockwise_perturbation_cifar100_20261028_v2"


def atomic_checkpoint(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return formal.digest_file(path)


def geometry_masks(model, geometry: dict) -> dict[str, torch.Tensor]:
    mapping = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if geometry.get("method") != "vpr" or geometry.get("mapping") != mapping:
        raise ValueError("Matching VPR geometry required")
    logs = {name: value for (name, _), value in zip(geometry["mapping"], geometry["logs"])}
    masks: dict[str, torch.Tensor] = {}
    for name, layer in eligible_layers(model).items():
        key = name + ".weight"
        weight = layer.weight.detach()
        log_radius = logs[key].to(weight.device)
        if log_radius.dtype != torch.float64 or log_radius.shape != weight.shape or not bool(torch.isfinite(log_radius).all()):
            raise ValueError(f"Geometry tensor mismatch: {key}")
        score = weight.abs().double() * torch.exp(-log_radius)
        flat = score.reshape(score.shape[0], -1)
        complete = flat.shape[1] // 4 * 4
        groups = flat[:, :complete].reshape(flat.shape[0], -1, 4)
        keep = groups.argsort(dim=-1, descending=True, stable=True)[..., :2]
        head = torch.zeros_like(groups, dtype=torch.bool)
        head.scatter_(-1, keep, True)
        mask = torch.ones_like(flat, dtype=torch.bool)
        mask[:, :complete] = head.reshape(flat.shape[0], complete)
        masks[key] = mask.reshape_as(weight)
    return masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme", choices=("global_block_feedback", "equal_block_energy"), required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite Raw-Hard output")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT or config.get("scope") != "seed0_only":
        raise ValueError("Wrong blockwise v2 config")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    formal.validate_checkpoint(parent)
    if (parent.get("experiment_id") != EXPERIMENT or parent.get("scheme") != args.scheme
            or parent.get("epoch") != 200 or parent.get("steps") != 78_200
            or parent["identity"].get("seed") != args.seed or parent.get("test_accessed") is not False):
        raise ValueError("Final parent identity mismatch")

    model = build_cifar100_model().cuda()
    model.load_state_dict(parent["model"])
    before_excluded = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    masks = geometry_masks(model, parent["stepper"]["geometry"])
    if len(masks) != 19:
        raise ValueError("Expected 19 eligible masks")
    positions = sum(mask.numel() for mask in masks.values())
    deleted = sum(int((~mask).sum()) for mask in masks.values())
    if positions != 11_157_504 or deleted != 5_578_752:
        raise ValueError("Frozen 2:4 position/deletion count mismatch")
    project(model, masks)
    validate_masks(model, masks, n=2, m=4, complete=True)
    for name, parameter in model.named_parameters():
        if name in masks:
            mask = masks[name].to(parameter.device)
            if torch.count_nonzero(parameter.detach()[~mask]):
                raise ValueError(f"Deleted coordinate is not zero: {name}")
        elif not torch.equal(parameter.detach().cpu(), before_excluded[name]):
            raise ValueError(f"Excluded parameter changed: {name}")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {
        "schema": 1, "status": "healthy", "protocol_id": formal.PROTOCOL_ID,
        "experiment_id": EXPERIMENT, "stage": "raw_hard", "scheme": args.scheme,
        "method": args.scheme, "seed": args.seed, "epoch": 0, "model": state,
        "model_sha256": formal.tensor_sha(state), "masks": {name: mask.cpu() for name, mask in masks.items()},
        "mask_sha256": formal.tensor_sha(masks), "eligible_layers": len(masks),
        "eligible_positions": positions, "deleted_positions": deleted,
        "parent": {"path": str(args.parent.resolve()), "sha256": formal.digest_file(args.parent)},
        "recovery_epochs": 0, "second_order_compensation": False, "test_accessed": False,
    }
    raw_path = output / "raw_hard.pt"
    raw_sha = atomic_checkpoint(raw_path, payload)
    formal.atomic_json(output / "status.json", {
        "status": "completed", "experiment_id": EXPERIMENT, "scheme": args.scheme,
        "seed": args.seed, "raw_hard_sha256": raw_sha, "mask_sha256": payload["mask_sha256"],
        "eligible_layers": len(masks), "eligible_positions": positions,
        "deleted_positions": deleted, "recovery_epochs": 0,
        "second_order_compensation": False, "test_accessed": False,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            formal.atomic_json(out / "failure.json", {"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc(), "automatic_retry": False})
            formal.atomic_json(out / "status.json", {"status": "failed", "message": str(error)})
        raise
