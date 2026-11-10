"""One-shot exploratory APBR geometry mask + fixed-mask OBS experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_cifar100_50k_formal_training as train
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.fixedmask_obs import FixedMaskOBSAdapter
from benchmark.nm import eligible_layers, validate_masks


FORMAL_PROTOCOL = "cifar100_50k10k_final_v1"
EXPERIMENT_ID = "apbr_seed0_fixedmask_obs_20261027"


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, value) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return train.digest_file(path)


def gpu_pids() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def require_safe_gpu(config: dict, formal_queue_status: Path | None) -> dict:
    safety = config["safety"]
    if safety.get("require_formal_queue_complete"):
        if formal_queue_status is None:
            raise ValueError("Formal queue status is required by this hardware binding")
        queue = json.loads(formal_queue_status.read_text(encoding="utf-8"))
        if queue.get("status") != "COMPLETE" or queue.get("seeds") != [0, 1, 2]:
            raise RuntimeError("Formal three-seed queue is not COMPLETE; refusing GPU experiment")
        isolation = "formal_queue_complete"
    else:
        if not safety.get("dedicated_gpu_no_formal_queue_present"):
            raise ValueError("Dedicated-GPU isolation is not asserted")
        isolation = "dedicated_gpu"
    occupied = gpu_pids()
    if occupied:
        raise RuntimeError(f"GPU has compute processes before exploratory launch: {occupied}")
    fields = [item.strip() for item in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")]
    if len(fields) != 5 or fields[1] != config["gpu_uuid"]:
        raise RuntimeError("Frozen GPU identity mismatch")
    return {
        "name": fields[0], "uuid": fields[1], "memory_total_mib": int(fields[2]),
        "memory_free_mib_before": int(fields[3]), "driver": fields[4], "isolation": isolation,
    }


class CalibrationSubset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, ids: list[int], transform) -> None:
        self.data, self.labels, self.ids, self.transform = data, labels, ids, transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sample_id = int(self.ids[index])
        image = self.data[sample_id].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[sample_id]), sample_id


def geometry_masks(model, checkpoint: dict) -> dict[str, torch.Tensor]:
    geometry = checkpoint["stepper"]["geometry"]
    mapping = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
    if geometry.get("method") != "vpr" or geometry.get("mapping") != mapping:
        raise ValueError("APBR geometry does not match the parent checkpoint/model")
    logs = {name: value for (name, _), value in zip(geometry["mapping"], geometry["logs"])}
    masks = {}
    for name, layer in eligible_layers(model).items():
        key = name + ".weight"
        weight = layer.weight.detach()
        state = logs[key].to(weight.device)
        if state.dtype != torch.float64 or state.shape != weight.shape or not bool(torch.isfinite(state).all()):
            raise ValueError(f"Geometry tensor mismatch: {key}")
        score = weight.abs().double() * torch.exp(-state)
        flat = score.reshape(score.shape[0], -1)
        end = flat.shape[1] // 4 * 4
        keep = torch.ones_like(flat, dtype=torch.bool)
        groups = flat[:, :end].reshape(flat.shape[0], -1, 4)
        selected = groups.argsort(dim=-1, descending=True, stable=True)[..., :2]
        head = torch.zeros_like(groups, dtype=torch.bool)
        head.scatter_(-1, selected, True)
        keep[:, :end] = head.reshape(flat.shape[0], end)
        masks[key] = keep.reshape_as(weight).detach().cpu()
    return masks


def state_dict_cpu(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def quadratic_error(weight: torch.Tensor, candidate: torch.Tensor, hessian: torch.Tensor) -> float:
    delta = (weight - candidate).double()
    h = hessian.double()
    return float(torch.einsum("ri,ij,rj->", delta, h, delta))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--historical-raw", type=Path, required=True)
    parser.add_argument("--formal-queue-status", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Refusing nonempty exploratory output")
    experiment_config = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))
    formal_config = yaml.safe_load(args.formal_config.read_text(encoding="utf-8"))
    if experiment_config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong exploratory experiment identity")
    if formal_config["protocol_id"] != FORMAL_PROTOCOL:
        raise ValueError("Wrong formal source protocol")
    gpu = require_safe_gpu(experiment_config, args.formal_queue_status)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device is required; CPU fallback forbidden")
    train.seed_all(0)

    checkpoint_load_started = time.perf_counter()
    parent_sha = train.digest_file(args.parent)
    raw_sha = train.digest_file(args.historical_raw)
    if parent_sha != experiment_config["parent"]["sha256"]:
        raise ValueError("APBR Dense parent SHA256 mismatch")
    if raw_sha != experiment_config["historical_raw"]["checkpoint_sha256"]:
        raise ValueError("Historical APBR Raw SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    historical = torch.load(args.historical_raw, map_location="cpu", weights_only=False)
    train.validate_checkpoint(parent)
    if parent.get("epoch") != 200 or parent["identity"].get("method") != "apbr_sam" or parent["identity"].get("seed") != 0:
        raise ValueError("APBR Dense parent identity mismatch")
    if parent.get("test_accessed") is not False or historical.get("test_accessed") is not False:
        raise ValueError("Test isolation metadata mismatch")
    checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started

    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169_001_437
            or train.digest_file(archive, "md5") != formal_config["dataset"]["archive_md5"]
            or train.digest_file(archive) != formal_config["dataset"]["archive_sha256"]):
        raise ValueError("Official CIFAR-100 archive identity mismatch")
    train_file = args.dataset_root / "extracted" / "cifar-100-python" / "train"
    with train_file.open("rb") as stream:
        raw_data = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw_data[b"data"], dtype=np.uint8)
    labels = np.asarray(raw_data[b"fine_labels"], dtype=np.int64)
    if data.shape != (50_000, 3072) or labels.shape != (50_000,) or not bool(np.all(np.bincount(labels, minlength=100) == 500)):
        raise ValueError("Official training data identity mismatch")

    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"])
    parent_state = {name: value.detach().cpu() for name, value in parent["model"].items()}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    total_started = time.perf_counter()

    mask_started = time.perf_counter()
    masks = geometry_masks(model, parent)
    mask_sha = train.tensor_sha(masks)
    if mask_sha != experiment_config["historical_raw"]["mask_sha256"] or mask_sha != historical["mask_sha256"]:
        raise ValueError("Generated APBR geometry Mask SHA256 mismatch")
    if set(masks) != set(historical["masks"]):
        raise ValueError("Historical APBR Mask layer set mismatch")
    if any(not torch.equal(mask, historical["masks"][name]) for name, mask in masks.items()):
        raise ValueError("Generated APBR geometry Mask differs from historical Mask")
    total_parameters = sum(mask.numel() for mask in masks.values())
    removed_parameters = sum(int((~mask).sum()) for mask in masks.values())
    if total_parameters != 11_157_504 or removed_parameters != 5_578_752:
        raise ValueError("APBR Mask count mismatch")
    for key, mask in masks.items():
        reconstructed = parent_state[key] * mask
        if not torch.equal(reconstructed, historical["model"][key]):
            raise ValueError(f"Mask-only reconstruction differs from historical Raw: {key}")
    for key, value in parent_state.items():
        if key not in masks and not torch.equal(value, historical["model"][key]):
            raise ValueError(f"Historical Raw changed excluded state: {key}")
    mask_seconds = time.perf_counter() - mask_started

    calibration_ids = np.random.default_rng(experiment_config["calibration"]["selection_seed"]).permutation(50_000)[:1024].tolist()
    if calibration_ids != historical["calibration_ids"]:
        raise ValueError("Calibration IDs differ from historical APBR/SparseGPT configuration")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(formal_config["dataset"]["normalize_mean"], formal_config["dataset"]["normalize_std"]),
    ])
    batches = DataLoader(
        CalibrationSubset(data, labels, calibration_ids, transform), batch_size=128, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=True,
    )
    layers = eligible_layers(model)
    adapters = {name: FixedMaskOBSAdapter(layer, compute_dtype=torch.float32) for name, layer in layers.items()}
    hooks = [layer.register_forward_pre_hook(lambda _module, values, item=adapters[name]: item.add_batch(values[0]))
             for name, layer in layers.items()]
    model.eval()
    torch.cuda.synchronize(device)
    calibration_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for inputs, _, _ in batches:
                model(inputs.to(device, non_blocking=True))
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize(device)
    calibration_seconds = time.perf_counter() - calibration_started
    hessian_build_seconds = sum(adapter.hessian_build_seconds() for adapter in adapters.values())

    layer_reports = []
    decomposition_seconds = compensation_seconds = validation_seconds = 0.0
    torch.cuda.synchronize(device)
    pruning_started = time.perf_counter()
    for name, layer in layers.items():
        key = name + ".weight"
        before = layer.weight.detach().clone()
        adapter = adapters.pop(name)
        hessians = adapter.hessians().detach().clone()
        returned, results = adapter.prune(
            masks[key].to(device), keep_n=2, m=4,
            blocksize=experiment_config["obs"]["blocksize"],
            percdamp=experiment_config["obs"]["percdamp"],
        )
        if not torch.equal(returned.cpu(), masks[key]):
            raise RuntimeError(f"Fixed Mask changed: {key}")
        after = layer.weight.detach()
        if not bool(torch.isfinite(after).all()) or torch.count_nonzero(after[~returned]):
            raise RuntimeError(f"Invalid compensated weights: {key}")
        survivor_changed = int(torch.count_nonzero(after[returned] != before[returned]))
        torch.cuda.synchronize(device)
        validation_started = time.perf_counter()
        raw_candidate = before * returned
        rows_per_group = before.shape[0] // layer.groups
        raw_error = obs_error = 0.0
        before_matrix = before.reshape(before.shape[0], -1)
        raw_matrix = raw_candidate.reshape(raw_candidate.shape[0], -1)
        after_matrix = after.reshape(after.shape[0], -1)
        for group, hessian in enumerate(hessians):
            rows = slice(group * rows_per_group, (group + 1) * rows_per_group)
            raw_error += quadratic_error(before_matrix[rows], raw_matrix[rows], hessian)
            obs_error += quadratic_error(before_matrix[rows], after_matrix[rows], hessian)
        torch.cuda.synchronize(device)
        layer_validation_seconds = time.perf_counter() - validation_started
        validation_seconds += layer_validation_seconds
        layer_decomposition = sum(result.decomposition_seconds for result in results)
        layer_compensation = sum(result.compensation_seconds for result in results)
        decomposition_seconds += layer_decomposition
        compensation_seconds += layer_compensation
        layer_reports.append({
            "layer": name, "weight_shape": list(before.shape), "K": int(before[0].numel()),
            "mask_equal_historical": True, "deleted_zero": True,
            "survivor_changed_count": survivor_changed,
            "finite": True, "damp": [result.damp for result in results],
            "decomposition_seconds": layer_decomposition,
            "compensation_seconds": layer_compensation,
            "validation_seconds": layer_validation_seconds,
            "local_quadratic_error_raw": raw_error,
            "local_quadratic_error_obs": obs_error,
            "local_error_ratio_obs_over_raw": obs_error / raw_error if raw_error else None,
        })
        del adapter, hessians, before, raw_candidate
    torch.cuda.synchronize(device)
    pruning_wall_seconds = time.perf_counter() - pruning_started
    validate_masks(model, {key: value.to(device) for key, value in masks.items()}, n=2, m=4, complete=True)
    if not any(report["survivor_changed_count"] > 0 for report in layer_reports):
        raise RuntimeError("OBS did not update any surviving weight")

    final_state = state_dict_cpu(model)
    excluded_unchanged = all(torch.equal(value, parent_state[key]) for key, value in final_state.items() if key not in masks)
    bn_unchanged = all(
        torch.equal(value, parent_state[key]) for key, value in final_state.items()
        if key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    )
    if not excluded_unchanged or not bn_unchanged:
        raise RuntimeError("Calibration/compensation changed excluded state or BN buffers")
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - total_started
    timing = {
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "geometry_mask_build_and_identity_seconds": mask_seconds,
        "calibration_wall_seconds_including_hessian": calibration_seconds,
        "hessian_gpu_event_seconds": hessian_build_seconds,
        "decomposition_seconds": decomposition_seconds,
        "obs_compensation_seconds": compensation_seconds,
        "pruning_wall_seconds": pruning_wall_seconds,
        "post_compensation_validation_seconds": validation_seconds,
        "pruning_wall_excluding_validation_seconds": pruning_wall_seconds - validation_seconds,
        "calibration_plus_pruning_wall_excluding_validation_seconds": (
            calibration_seconds + pruning_wall_seconds - validation_seconds
        ),
        "total_mask_calibration_pruning_compensation_seconds": total_seconds,
        "test_evaluation_seconds": None,
    }
    payload = {
        "schema": 1, "status": "healthy", "protocol_id": FORMAL_PROTOCOL,
        "experiment_id": EXPERIMENT_ID, "stage": "raw_hard", "method": "apbr_fixedmask_obs",
        "seed": 0, "epoch": 0, "recovery_epochs": 0, "model": final_state,
        "model_sha256": train.tensor_sha(final_state), "masks": masks, "mask_sha256": mask_sha,
        "parent": {"path": str(args.parent), "sha256": parent_sha},
        "historical_raw": {"path": str(args.historical_raw), "sha256": raw_sha},
        "calibration_ids": calibration_ids, "test_accessed": False,
        "dynamic_mask_selection": False, "gradient_training": False,
        "bn_calibration": False, "bn_buffers_parent_identical": True,
        "timing": timing, "layer_reports": layer_reports,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu": gpu,
        "source_hashes": {
            "formal_config": train.digest_file(args.formal_config),
            "experiment_config": train.digest_file(args.experiment_config),
            "runner": train.digest_file(Path(__file__)),
            "fixedmask_adapter": train.digest_file(ROOT / "baseline_adapters" / "fixedmask_obs" / "adapter.py"),
        },
    }
    checkpoint_path = output / "apbr_seed0_fixedmask_obs.pt"
    checkpoint_sha = atomic_checkpoint(checkpoint_path, payload)
    evidence = {
        "status": "COMPENSATION_PASS", "experiment_id": EXPERIMENT_ID,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": parent_sha, "historical_raw_sha256": raw_sha,
        "mask_sha256": mask_sha, "total_parameters": total_parameters,
        "removed_parameters": removed_parameters, "mask_matches_historical_all_19_layers": True,
        "mask_only_reconstruction_matches_historical": True,
        "excluded_state_parent_identical": excluded_unchanged,
        "bn_buffers_parent_identical": bn_unchanged,
        "survivor_changed_count": sum(report["survivor_changed_count"] for report in layer_reports),
        "all_finite": all(report["finite"] for report in layer_reports),
        "all_deleted_zero": all(report["deleted_zero"] for report in layer_reports),
        "dynamic_mask_selection": False, "gradient_training": False, "test_accessed": False,
        "timing": timing, "peak_allocated_bytes": payload["peak_allocated_bytes"],
        "peak_reserved_bytes": payload["peak_reserved_bytes"], "gpu": gpu,
        "calibration_ids_sha256": hashlib.sha256(np.asarray(calibration_ids, dtype="<i8").tobytes()).hexdigest(),
        "layer_reports": layer_reports, "source_hashes": payload["source_hashes"],
    }
    atomic_json(output / "compensation_evidence.json", evidence)
    atomic_json(output / "mask_reference.json", {
        "mask_sha256": mask_sha, "historical_raw_checkpoint_sha256": raw_sha,
        "mask_storage": "embedded in apbr_seed0_fixedmask_obs.pt",
        "layers": sorted(masks), "total_parameters": total_parameters,
        "removed_parameters": removed_parameters,
    })
    atomic_json(output / "status.json", {
        "status": "COMPENSATION_PASS", "experiment_id": EXPERIMENT_ID,
        "checkpoint_sha256": checkpoint_sha, "test_accessed": False,
        "automatic_retry": False,
    })
    print(json.dumps(evidence, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            directory = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "FAILED", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
            }
            failure_path = directory / "failure.json"
            if not failure_path.exists():
                atomic_json(failure_path, failure)
        raise
