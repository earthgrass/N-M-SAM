"""Independent final-only CIFAR-100 test evaluator.

This program performs no training, pruning, checkpoint selection, or tuning.
Its output is bound to the exact input checkpoint SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_adapters.cifar100_model import build_cifar100_model

PROTOCOL_ID = "cifar100_50k10k_final_v1"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class OfficialTest(Dataset):
    def __init__(self, data, labels, transform):
        self.data, self.labels, self.transform = data, labels, transform

    def __len__(self):
        return 10_000

    def __getitem__(self, index):
        image = self.data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(image), int(self.labels[index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-role", choices=("dense_final", "raw_hard", "recovery_final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Final evaluation requires CUDA; CPU fallback forbidden")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["protocol_id"] != PROTOCOL_ID or config["evaluation"]["primary_checkpoint"] != "final":
        raise ValueError("Wrong evaluation protocol")
    checkpoint_sha = sha_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.checkpoint_role == "dense_final":
        if checkpoint.get("protocol_id") != PROTOCOL_ID or checkpoint.get("epoch") != 200:
            raise ValueError("Dense evaluation requires epoch-200 protocol checkpoint")
    elif args.checkpoint_role == "recovery_final":
        if checkpoint.get("protocol_id") != PROTOCOL_ID or checkpoint.get("epoch") != 20:
            raise ValueError("Recovery evaluation requires epoch-20 protocol checkpoint")
    else:
        if checkpoint.get("protocol_id") != PROTOCOL_ID or checkpoint.get("stage") != "raw_hard":
            raise ValueError("Raw-Hard checkpoint identity mismatch")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint has no model state")
    model = build_cifar100_model().cuda()
    model.load_state_dict(model_state)
    model.eval()

    test_file = args.dataset_root / "extracted" / "cifar-100-python" / "test"
    with test_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="bytes")
    data = np.asarray(raw[b"data"], dtype=np.uint8)
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if data.shape != (10_000, 3072) or labels.shape != (10_000,) or not bool(np.all(np.bincount(labels, minlength=100) == 100)):
        raise ValueError("Official test identity/shape/class distribution mismatch")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        config["dataset"]["normalize_mean"], config["dataset"]["normalize_std"])])
    batches = DataLoader(OfficialTest(data, labels, transform), batch_size=128, shuffle=False,
                         drop_last=False, num_workers=2, pin_memory=True)
    loss_sum, correct, count = 0.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in batches:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite final test output")
            loss_sum += float(loss)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
    if count != 10_000:
        raise ValueError("Final test coverage mismatch")
    torch.cuda.synchronize()
    result = {"status": "PASS", "protocol_id": PROTOCOL_ID,
              "checkpoint_role": args.checkpoint_role, "checkpoint_path": str(args.checkpoint),
              "checkpoint_sha256": checkpoint_sha, "test_pickle_sha256": sha_file(test_file),
              "test_count": count, "test_accuracy": correct / count, "test_loss": loss_sum / count,
              "selection_performed": False, "hyperparameter_change_performed": False,
              "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    atomic_json(args.output, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
