"""Same-GPU cost-only native SparseGPT timing against SAM dynamic Group-OBS.

Uses the frozen Dense-SAM parent and 1,024 Train calibration images. It never
constructs or opens the Test dataset and writes no accuracy result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_geo_obs_seed0 as common
from baseline_adapters.cifar100_model import build_cifar100_model
from baseline_adapters.common.conv2d import weight_to_matrix
from baseline_adapters.sparsegpt import SparseGPTAdapter, sparsegpt_matrix
from benchmark.nm import eligible_layers

PARENT_SHA256 = "3a0716555d33ec92efe8362e24abec9be27625d24ffb6dfb1d51fddb5b91bd13"


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA GPU required")
    if common.digest(args.parent) != PARENT_SHA256:
        raise ValueError("Dense-SAM checkpoint SHA256 mismatch")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (parent.get("protocol_id") != "cifar100_50k10k_final_v1"
            or parent.get("epoch") != 200 or parent.get("steps") != 78200
            or parent.get("test_accessed") is not False
            or parent.get("identity", {}).get("method") != "dense_sam"
            or parent.get("identity", {}).get("seed") != 0):
        raise ValueError("Dense-SAM identity mismatch")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != "cifar100_50k10k_final_v1":
        raise ValueError("Config mismatch")
    archive = args.dataset_root / "cifar-100-python.tar.gz"
    if (archive.stat().st_size != 169001437 or common.digest(archive) != common.ARCHIVE_SHA256
            or common.digest(archive, "md5") != common.ARCHIVE_MD5):
        raise ValueError("Official archive mismatch")
    data, labels = common.read_cifar(args.dataset_root / "extracted/cifar-100-python/train", 50000)
    ids = np.random.default_rng(common.CALIBRATION_SEED).permutation(50000)[:1024].tolist()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"]),
    ])
    loader = DataLoader(common.CifarSubset(data, labels, ids, transform), batch_size=128,
                        shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda:0")
    model = build_cifar100_model().to(device)
    model.load_state_dict(parent["model"], strict=True)
    model.eval()
    layers = eligible_layers(model)
    if len(layers) != 19 or sum(layer.weight.numel() for layer in layers.values()) != 11157504:
        raise ValueError("Eligible-layer mismatch")
    common.atomic_json(out / "preflight.json", {
        "status": "PASS", "parent_sha256": PARENT_SHA256,
        "archive_sha256": common.ARCHIVE_SHA256, "calibration_ids_sha256": ids_sha,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "test_accessed": False,
        "source_sha256": {"runner": common.digest(Path(__file__)),
                          "native_adapter": common.digest(ROOT / "baseline_adapters/sparsegpt/adapter.py")},
    })
    adapters, hooks = {}, []
    for name, layer in layers.items():
        adapter = SparseGPTAdapter(layer, compute_dtype=torch.float32)
        adapters[name] = adapter
        hooks.append(layer.register_forward_pre_hook(lambda _m, values, item=adapter: item.add_batch(values[0])))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    seen = []
    with torch.inference_mode():
        for inputs, _targets, sample_ids in loader:
            model(inputs.to(device, non_blocking=True))
            seen.extend(map(int, sample_ids.tolist()))
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    if seen != ids:
        raise ValueError("Calibration IDs/order mismatch")
    rows = []
    peak_allocated = peak_reserved = 0
    for name, layer in layers.items():
        w = weight_to_matrix(layer.weight.detach())
        h = adapters[name].hessians()[0]
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = sparsegpt_matrix(w, h, keep_n=2, m=4, blocksize=128,
                                  percdamp=.01, compute_dtype=torch.float32)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not bool(torch.isfinite(result.weight).all()) or int(torch.count_nonzero(result.weight[~result.keep_mask])):
            raise FloatingPointError(f"Invalid native OBS output: {name}")
        rows.append({"layer": name, "K": w.shape[1], "deleted": int((~result.keep_mask).sum()),
                     "decomposition_seconds": result.decomposition_seconds,
                     "compensation_seconds": result.compensation_seconds,
                     "wall_seconds": seconds})
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        print(json.dumps({"event": "layer_timed", "layer": name, "seconds": seconds,
                          "test_accessed": False}), flush=True)
        torch.cuda.empty_cache()
    if sum(r["deleted"] for r in rows) != 5578752:
        raise ValueError("2:4 deletion count mismatch")
    common.atomic_csv(out / "per_layer.csv", rows)
    common.atomic_json(out / "status.json", {
        "status": "COMPLETE", "method": "native_sparsegpt_cost_only", "seed": 0,
        "parent_sha256": PARENT_SHA256, "calibration_ids_sha256": ids_sha,
        "calibration_seconds": calibration_seconds,
        "pruning_compensation_seconds": sum(r["wall_seconds"] for r in rows),
        "decomposition_seconds": sum(r["decomposition_seconds"] for r in rows),
        "compensation_seconds": sum(r["compensation_seconds"] for r in rows),
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "test_accessed": False, "accuracy_measured": False,
    })
    print(json.dumps({"event": "complete", "calibration_seconds": calibration_seconds,
                      "pruning_compensation_seconds": sum(r["wall_seconds"] for r in rows)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        main(args)
    except BaseException:
        if args.output.exists():
            common.atomic_json(args.output / "failure.json", {"traceback": traceback.format_exc(),
                                                              "test_accessed": False})
        raise
