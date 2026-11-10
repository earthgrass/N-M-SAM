"""One authorized V2 trajectory and explicit cross-snapshot identity checks."""
import hashlib
import json
from pathlib import Path
import torch
from .repro import seed_for, file_hash

V1_COMMIT = "d4c58d66b248b620e0e058b92e79100f173970a6"
GPU_UUID = "fc8784a7-85a2-2bcd-d2e2-f36ef021616c"
CONFIG = "configs/approved_vpr_l2/seed0.json"
OUTPUT = "runs/vpr_l2_seed0_v2/fixed_l2_seed0"


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for key in sorted(a.keys() | b.keys())
                for row in differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip("."))]
    return [] if a == b else [{"field": prefix, "old": a, "new": b}]


def validate_protocol(root):
    ref = Path(root)/"docs/vpr_l2_reference"
    receipts = json.loads((ref/"REFERENCE_SHA256.json").read_text())
    for name, expected in receipts.items():
        if file_hash(ref/name) != expected:
            raise ValueError(f"Historical evidence changed: {name}")
    old = json.loads((ref/"v1_vpr_seed0.json").read_text())
    new = json.loads((Path(root)/CONFIG).read_text())
    diff = differences(old, new)
    expected = [{"field": "optimizer.method", "old": "vpr", "new": "fixed_l2"},
                {"field": "output", "old": old["output"], "new": OUTPUT}]
    if diff != expected:
        raise ValueError(f"Unapproved configuration differences: {diff}")
    return new, diff


def full_order_hash(manifest, epochs=200, batch_size=128):
    import numpy as np
    ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    value = hashlib.sha256(b"vpr-order-v1").hexdigest()
    for epoch in range(epochs):
        gen = torch.Generator().manual_seed(seed_for(0, "order", epoch))
        order = ids[torch.randperm(len(ids), generator=gen).numpy()]
        for start in range(0, len(order), batch_size):
            value = hashlib.sha256(bytes.fromhex(value)+order[start:start+batch_size].tobytes()).hexdigest()
    return value


def verify_final(state, gates):
    if state["stage"] != "train" or state["epoch"] != 200 or state["next_batch"] != 0:
        raise ValueError("Not an exact final dense checkpoint")
    if state["stepper"]["steps"] != 70400 or len(state["step_rows"]) != 70400 or len(state["epoch_rows"]) != 200:
        raise ValueError("Incorrect formal step/epoch budget")
    if state["initialization_hash"] != gates["initialization_hash"] or state["order_hash"] != gates["order_hash"]:
        raise ValueError("V1/V2 paired identity mismatch")
    if state["manifest"]["fingerprint"] != gates["dataset_fingerprint"]:
        raise ValueError("V1/V2 split mismatch")
    geom = state["stepper"]["geometry"]
    if geom["method"] != "fixed_l2" or geom["updates"]+geom["zero_skips"] != 70400:
        raise ValueError("Geometry method/update mismatch")
    if not all(s.dtype == torch.float64 and bool(torch.isfinite(s).all()) for s in geom["logs"]):
        raise ValueError("Invalid FP64 geometry")
    for row in state["step_rows"]:
        for value in row.values():
            if isinstance(value, float) and not __import__('math').isfinite(value):
                raise FloatingPointError("Non-finite history")
        if not row.get("zero_perturbation", False) and abs(row["epsilon_l2_over_rho"]-1) > 2e-5:
            raise FloatingPointError("Radius history mismatch")
    if not all(bool(torch.isfinite(v).all()) for v in state["model"].values()):
        raise FloatingPointError("Non-finite model")
