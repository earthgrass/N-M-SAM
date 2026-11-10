"""Replay a rolling checkpoint and locate exact zero-energy 2:4 feedback blocks."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
VPR_SRC = ROOT.parents[1] / "src"
sys.path.insert(0, str(VPR_SRC))

import run_cifar100_50k_formal_training as formal
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from blockwise_apbr import BlockwiseStepper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-repaired", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    formal.validate_checkpoint(checkpoint)
    seed = int(checkpoint["identity"]["seed"])
    scheme = checkpoint["scheme"]
    formal.seed_all(seed)

    with (args.dataset_root / "extracted/cifar-100-python/train").open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    dataset = formal.OfficialTrain(data, labels, transform)
    model = build_cifar100_model().cuda()
    eligible_names = {name + ".weight" for name in eligible_layers(model)}
    method_config = {
        "seed": seed,
        "optimizer": {"method": "vpr", "lr": 0.1, "momentum": 0.9,
                      "weight_decay": 0.0005, "nesterov": False, "rho": config["apbr"]["rho"]},
        "geometry": {"gamma": config["apbr"]["gamma"], "dtype": config["apbr"]["geometry_dtype"],
                     "warmup_steps": config["apbr"]["warmup_steps"],
                     "volume_atol": config["apbr"]["volume_atol"],
                     "volume_rtol": config["apbr"]["volume_rtol"], "static_std": 0.1,
                     "block_degenerate_threshold": config["blockwise"]["degenerate_threshold"]},
    }
    stepper = BlockwiseStepper(model, method_config, scheme, eligible_names)
    model.load_state_dict(checkpoint["model"])
    stepper.load_state_dict(checkpoint["stepper"])
    formal.restore_rng(checkpoint["rng"])

    original_prepare = stepper.geometry.prepare
    evidence: dict = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
                      "checkpoint_steps": checkpoint["steps"], "scheme": scheme, "seed": seed,
                      "first_degenerate_batch": None, "layers": []}
    current_batch = -1

    def diagnostic_prepare(eps, successful_steps, zero=False):
        layers = []
        total = 0
        for index in sorted(stepper.geometry.eligible_indices):
            value = eps[index]
            groups = value.double().square().reshape(value.shape[0], -1)
            end = groups.shape[1] // 4 * 4
            energies = groups[:, :end].reshape(groups.shape[0], -1, 4).sum(-1)
            bad = (~torch.isfinite(energies)) | (energies <= 0)
            count = int(bad.sum())
            if count:
                coordinates = torch.nonzero(bad, as_tuple=False)[:16].cpu().tolist()
                parameter = stepper.geometry.params[index]
                grad_groups = parameter.grad.detach().double().reshape(parameter.shape[0], -1)
                grad_groups = grad_groups[:, :end].reshape(parameter.shape[0], -1, 4)
                log_groups = stepper.geometry.logs[index].double().reshape(parameter.shape[0], -1)
                log_groups = log_groups[:, :end].reshape(parameter.shape[0], -1, 4)
                samples = []
                for out_channel, group_index in coordinates:
                    samples.append({
                        "out_channel": out_channel, "group_index": group_index,
                        "epsilon": value.reshape(value.shape[0], -1)[out_channel, group_index * 4:(group_index + 1) * 4].cpu().tolist(),
                        "gradient": grad_groups[out_channel, group_index].cpu().tolist(),
                        "log_radius": log_groups[out_channel, group_index].cpu().tolist(),
                    })
                layers.append({"parameter": stepper.geometry.mapping[index][0],
                               "zero_or_nonfinite_blocks": count, "samples": samples})
                total += count
        if total and evidence["first_degenerate_batch"] is None:
            evidence["first_degenerate_batch"] = current_batch
            evidence["global_step_before_failure"] = successful_steps
            evidence["total_zero_or_nonfinite_blocks"] = total
            evidence["layers"] = layers
            args.output.parent.mkdir(parents=True, exist_ok=True)
            formal.atomic_json(args.output, evidence)
        return original_prepare(eps, successful_steps, zero)

    stepper.geometry.prepare = diagnostic_prepare
    try:
        for current_batch, (inputs, targets, _) in enumerate(
                formal.training_loader(dataset, 128, seed, checkpoint["epoch"])):
            stepper.step(inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True))
    except FloatingPointError as error:
        evidence["observed_exception"] = str(error)
        formal.atomic_json(args.output, evidence)
        print(json.dumps(evidence, indent=2), flush=True)
        return 0 if evidence["first_degenerate_batch"] is not None else 2
    if args.expect_repaired and evidence["first_degenerate_batch"] is not None:
        evidence["repair_replay_status"] = "PASS"
        evidence["completed_replayed_epoch"] = checkpoint["epoch"] + 1
        evidence["steps_after_replay"] = stepper.steps
        formal.atomic_json(args.output, evidence)
        print(json.dumps(evidence, indent=2), flush=True)
        return 0
    raise RuntimeError("Checkpoint replay did not reproduce the recorded failure")


if __name__ == "__main__":
    raise SystemExit(main())
