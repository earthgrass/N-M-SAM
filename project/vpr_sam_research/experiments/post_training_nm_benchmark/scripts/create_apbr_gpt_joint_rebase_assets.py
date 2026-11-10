"""Create one frozen CPU initial state shared by all rebased Seed-0 branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(PROJECT / "src")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import CIFAR100_MODEL_ID, build_cifar100_model
from vpr_sam.optim.geometry import Geometry


COHORT = "apbr_gpt_joint_seed0_rebased"
BRANCHES = ("dense_sam", "apbr_sam", "joint_apbr_gpt")


def tensor_sequence_sha(values) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a rebased asset directory")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train.seed_all(0)
    rng_before_model = torch.get_rng_state().clone()
    model = build_cifar100_model().cpu()
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model_sha = train.tensor_sha(model_state)
    geometry_config = dict(config["apbr_sam"]["geometry"])
    geometry = Geometry(model.named_parameters(), geometry_config, method="vpr", seed=0)
    geometry_state = geometry.state_dict()
    geometry_sha = tensor_sequence_sha(
        (name, value) for (name, _shape), value in zip(geometry_state["mapping"], geometry_state["logs"])
    )
    if any(torch.count_nonzero(value) for value in geometry_state["logs"]):
        raise RuntimeError("Initial APBR geometry is not all-zero")
    payload = {
        "schema": 1,
        "status": "frozen_initial_asset",
        "cohort": COHORT,
        "seed": 0,
        "model_id": CIFAR100_MODEL_ID,
        "model": model_state,
        "model_tensor_sha256": model_sha,
        "apbr_geometry": geometry_state,
        "geometry_tensor_sha256": geometry_sha,
        "rng_before_model_initialization": rng_before_model,
        "rng_after_asset_creation": train.rng_state(),
        "branches": BRANCHES,
        "all_branches_load_identical_model_state": True,
        "random_initialization": True,
        "historical_checkpoint_used_as_initialization": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_used": False,
        "final_test_accessed": False,
        "source_sha256": {
            "creator": train.digest_file(Path(__file__)),
            "model": train.digest_file(ROOT / "baseline_adapters" / "cifar100_model.py"),
            "benchmark_model": train.digest_file(ROOT / "benchmark" / "model.py"),
            "geometry": train.digest_file(PROJECT / "src" / "vpr_sam" / "optim" / "geometry.py"),
            "config": train.digest_file(args.config),
        },
    }
    path = output / "initial_common_seed0.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if train.tensor_sha(loaded["model"]) != model_sha:
        raise RuntimeError("Rebased model tensor hash changed on reload")
    temporary.replace(path)
    file_sha = train.digest_file(path)
    references = {}
    for branch in BRANCHES:
        reference = {
            "schema": 1, "cohort": COHORT, "branch": branch, "seed": 0,
            "initial_checkpoint": "initial_common_seed0.pt",
            "initial_checkpoint_sha256": file_sha,
            "initial_model_tensor_sha256": model_sha,
            "initial_geometry_tensor_sha256": geometry_sha if branch != "dense_sam" else None,
            "formal_training_started": False, "final_test_accessed": False,
        }
        train.atomic_json(output / f"{branch}_initial_reference.json", reference)
        references[branch] = reference
    train.atomic_json(output / "asset_manifest.json", {
        "status": "REBASED_COHORT_ASSET_FROZEN",
        "cohort": COHORT,
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha,
        "model_tensor_sha256": model_sha,
        "geometry_tensor_sha256": geometry_sha,
        "branches": references,
        "formal_training_started": False,
        "cuda_used": False,
        "final_test_accessed": False,
    })
    print(json.dumps({"checkpoint_sha256": file_sha, "model_tensor_sha256": model_sha,
                      "geometry_tensor_sha256": geometry_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
