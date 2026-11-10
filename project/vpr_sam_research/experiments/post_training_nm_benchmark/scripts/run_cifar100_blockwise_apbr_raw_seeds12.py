"""Seeds 1/2 Raw-Hard extension using the frozen Seed-0 v2 implementation."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as formal
import run_cifar100_blockwise_apbr_raw as base
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import project, validate_masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme", choices=("global_block_feedback", "equal_block_energy"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite Raw-Hard output")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != base.EXPERIMENT or config.get("scope") != "seeds1_2_extension":
        raise ValueError("Wrong Seeds-1/2 extension config")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    formal.validate_checkpoint(parent)
    if (parent.get("experiment_id") != base.EXPERIMENT or parent.get("scheme") != args.scheme
            or parent.get("epoch") != 200 or parent.get("steps") != 78_200
            or parent["identity"].get("seed") != args.seed or parent.get("test_accessed") is not False):
        raise ValueError("Final parent identity mismatch")
    model = build_cifar100_model().cuda(); model.load_state_dict(parent["model"])
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    masks = base.geometry_masks(model, parent["stepper"]["geometry"])
    positions = sum(mask.numel() for mask in masks.values())
    deleted = sum(int((~mask).sum()) for mask in masks.values())
    if len(masks) != 19 or positions != 11_157_504 or deleted != 5_578_752:
        raise ValueError("Frozen mask count mismatch")
    project(model, masks); validate_masks(model, masks, n=2, m=4, complete=True)
    for name, value in model.state_dict().items():
        if name not in masks and not torch.equal(value.detach().cpu(), before[name]):
            raise ValueError(f"Excluded state changed: {name}")
    for name, parameter in model.named_parameters():
        if name in masks and torch.count_nonzero(parameter.detach()[~masks[name].to(parameter.device)]):
            raise ValueError(f"Deleted coordinate is not zero: {name}")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {"schema": 1, "status": "healthy", "protocol_id": formal.PROTOCOL_ID,
               "experiment_id": base.EXPERIMENT, "stage": "raw_hard", "scheme": args.scheme,
               "method": args.scheme, "seed": args.seed, "epoch": 0, "model": state,
               "model_sha256": formal.tensor_sha(state), "masks": {name: mask.cpu() for name, mask in masks.items()},
               "mask_sha256": formal.tensor_sha(masks), "eligible_layers": len(masks),
               "eligible_positions": positions, "deleted_positions": deleted,
               "parent": {"path": str(args.parent.resolve()), "sha256": formal.digest_file(args.parent)},
               "recovery_epochs": 0, "second_order_compensation": False, "test_accessed": False}
    raw_path = output / "raw_hard.pt"; raw_sha = base.atomic_checkpoint(raw_path, payload)
    formal.atomic_json(output / "status.json", {"status": "completed", "experiment_id": base.EXPERIMENT,
                       "scheme": args.scheme, "seed": args.seed, "raw_hard_sha256": raw_sha,
                       "mask_sha256": payload["mask_sha256"], "eligible_layers": len(masks),
                       "eligible_positions": positions, "deleted_positions": deleted,
                       "recovery_epochs": 0, "second_order_compensation": False, "test_accessed": False})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if "--output-dir" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
            formal.atomic_json(out / "failure.json", {"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc(), "automatic_retry": False})
            formal.atomic_json(out / "status.json", {"status": "failed", "message": str(error)})
        raise
