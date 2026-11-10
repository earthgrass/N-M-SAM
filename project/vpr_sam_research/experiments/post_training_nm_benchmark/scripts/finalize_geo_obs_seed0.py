"""Finalize an interrupted Geo-OBS run from immutable 19-layer artifacts.

The computation failure occurred only in CSV assembly. This entry point does
not recalibrate, select masks, compensate weights, or change any layer output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_geo_obs_seed0 as geo
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers, validate_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-kind", choices=tuple(geo.PARENTS), required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output
    if (out / "frozen_pretest.json").exists() or (out / "results.csv").exists():
        raise FileExistsError("Results already frozen or evaluated")
    preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    expected_sha, protocol, identity_method = geo.PARENTS[args.parent_kind]
    if (progress["completed_layers"] != 19 or progress["test_accessed"] is not False
            or preflight["parent_sha256"] != expected_sha
            or preflight["config_sha256"] != geo.digest(args.config)
            or geo.digest(args.parent) != expected_sha):
        raise ValueError("Incomplete or mismatched preprocessing evidence")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != protocol:
        raise ValueError("Protocol mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    identity = parent.get("identity", {})
    if (parent.get("protocol_id") != protocol or parent.get("epoch") != 200
            or parent.get("steps") != 78200 or parent.get("test_accessed") is not False
            or identity.get("method") != identity_method or identity.get("seed") != 0):
        raise ValueError("Dense parent mismatch")
    model = build_cifar100_model()
    model.load_state_dict(parent["model"], strict=True)
    layers = eligible_layers(model)
    if list(layers) != preflight["eligible_layers"] or len(layers) != 19:
        raise ValueError("Layer list mismatch")
    states = {method: {k: v.detach().clone() for k, v in model.state_dict().items()}
              for method in geo.METHODS}
    masks = {method: {} for method in geo.METHODS}
    rows = []
    layer_hashes = {}
    for index, (name, layer) in enumerate(layers.items(), 1):
        key = name + ".weight"
        for method in geo.METHODS:
            path = out / "layers" / f"{index:02d}_{name.replace('.', '_')}_{method}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            if artifact["layer"] != name or artifact["method"] != method:
                raise ValueError("Layer artifact identity mismatch")
            mask, weight, costs, chosen = (artifact[k] for k in
                                           ("mask", "weight", "candidate_costs", "chosen_pair"))
            expected_mask, expected_chosen = geo.six_pair_keep(costs, weight[0].numel())
            if (not torch.equal(chosen, expected_chosen)
                    or not torch.equal(mask.reshape_as(expected_mask), expected_mask)
                    or mask.dtype != torch.bool or mask.shape != layer.weight.shape
                    or weight.shape != layer.weight.shape or not bool(torch.isfinite(weight).all())
                    or int(torch.count_nonzero(weight[~mask]))):
                raise ValueError(f"Layer integrity failure: {name}/{method}")
            states[method][key] = weight
            masks[method][key] = mask
            rows.append(artifact["diagnostic"])
            layer_hashes[str(path.relative_to(out))] = geo.digest(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    for row in rows:
        for field in fields:
            row.setdefault(field, None)
    geo.atomic_csv(out / "per_layer.csv", rows)
    for method in geo.METHODS:
        audit_model = build_cifar100_model()
        audit_model.load_state_dict(states[method], strict=True)
        validate_masks(audit_model, masks[method], n=2, m=4, complete=True)
        if sum(int((~x).sum()) for x in masks[method].values()) != 5578752:
            raise ValueError("Deletion count mismatch")
        for key, value in parent["model"].items():
            if key not in masks[method] and not torch.equal(states[method][key], value):
                raise ValueError(f"Unpruned state changed: {key}")
    frozen = {}
    for method in geo.METHODS:
        method_dir = out / "methods" / method
        payload = {"schema": 1, "status": "healthy", "protocol_id": protocol,
                   "experiment_id": "geo_obs_seed0_20261102_v1", "parent_kind": args.parent_kind,
                   "stage": "post_obs_raw_hard", "method": method, "seed": 0, "recovery_epochs": 0,
                   "model": states[method], "masks": masks[method], "parent_sha256": expected_sha,
                   "config_sha256": preflight["config_sha256"],
                   "calibration_ids_sha256": preflight["calibration_ids_sha256"],
                   "source_sha256": preflight["source_sha256"],
                   "finalizer_sha256": geo.digest(Path(__file__)),
                   "test_accessed": False, "gradient_training": False, "bn_calibration": False}
        mask_sha = geo.atomic_torch(method_dir / "masks.pt", masks[method])
        checkpoint_sha = geo.atomic_torch(method_dir / "checkpoint.pt", payload)
        frozen[method] = {"checkpoint_sha256": checkpoint_sha, "mask_sha256": mask_sha}
    geo.atomic_json(out / "frozen_pretest.json", {"status": "ALL_CHECKPOINTS_FROZEN",
                                                 "methods": frozen, "layer_sha256": layer_hashes,
                                                 "test_accessed": False})
    # Opening the official Test pickle happens only after both checkpoints are frozen.
    test_path = args.dataset_root / "extracted/cifar-100-python/test"
    data, labels = geo.read_cifar(test_path, 10000)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(geo.CifarSubset(data, labels, range(10000), transform),
                        batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    results = []
    for method in geo.METHODS:
        if geo.digest(out / "methods" / method / "checkpoint.pt") != frozen[method]["checkpoint_sha256"]:
            raise ValueError("Checkpoint altered after freeze")
        started = time.perf_counter()
        accuracy, ce, count = geo.evaluate_once(states[method], loader, device)
        results.append({"method": method, "accuracy_percent": accuracy, "ce": ce,
                        "test_samples": count, "test_evaluations": 1,
                        "checkpoint_sha256": frozen[method]["checkpoint_sha256"],
                        "mask_sha256": frozen[method]["mask_sha256"],
                        "pruning_compensation_seconds": sum(r["method_layer_seconds"] for r in rows if r["method"] == method),
                        "test_seconds": time.perf_counter() - started})
        print(json.dumps({"event": "final_test", **results[-1]}), flush=True)
    geo.atomic_csv(out / "results.csv", results)
    geo.atomic_json(out / "status.json", {"status": "COMPLETE", "parent_kind": args.parent_kind,
                                         "test_accessed_only_after_freeze": True, "recovery_epochs": 0,
                                         "results": results, "completed_layers": 19,
                                         "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                                         "initial_csv_failure_preserved": True})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{geo.digest(p)}  {p.relative_to(out).as_posix()}\n" for p in files))


if __name__ == "__main__":
    main()
