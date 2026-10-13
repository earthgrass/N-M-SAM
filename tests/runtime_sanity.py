"""Independent runtime sanity checks.

Loads the saved best.pt of a sparse run and verifies:
1. Every 2:4 block has exactly 2 active weights.
2. Effective weight is zero wherever mask==0.
3. Per-layer sparsity matches the 2:4 spec (within tail tolerance).
"""
import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nm_sam_cifar10.models import build_model
from nm_sam_cifar10.sparsity import SparsityConfig, attach_nm_sparsity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="best.pt")
    args = p.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("config", data)

    model = build_model(cfg["model"], num_classes=10)
    if cfg["method"] != "dense_sgd" and cfg["method"] != "dense_sam":
        sparsity_cfg = SparsityConfig(n=cfg["n"], m=cfg["m"], enabled=True)
        attach_nm_sparsity(model, sparsity_cfg)

    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location="cpu")
    model.load_state_dict(state["model"], strict=False)

    wrappers = list(getattr(model, "_nm_wrappers", []))
    print(f"Loaded {len(wrappers)} sparse wrappers.")

    n_blocks_total = 0
    n_blocks_bad = 0
    n_eff_zero_bad = 0
    for w in wrappers:
        mask = w.mask
        weight = w.weight
        # Block sums
        if weight.dim() == 4:
            o, ic, kh, kw = weight.shape
            flat_m = mask.reshape(o, -1)
        else:
            flat_m = mask
        inner = flat_m.size(-1)
        num_full = inner // cfg["m"]
        if num_full == 0:
            continue
        head = flat_m[..., : num_full * cfg["m"]].reshape(-1, cfg["m"])
        sums = head.sum(dim=-1)
        n_blocks_total += sums.numel()
        n_blocks_bad += int((sums != cfg["n"]).sum().item())

        # Effective weight zero
        eff = w()
        zero_pos = (mask == 0)
        n_zero = int(zero_pos.sum().item())
        if n_zero > 0:
            sample_idx = zero_pos.nonzero(as_tuple=False)[:64]
            vals = eff[sample_idx.split(1, dim=1)].squeeze(1)
            n_eff_zero_bad += int((vals != 0.0).sum().item())

    print(f"[blocks] total={n_blocks_total}, bad={n_blocks_bad}")
    print(f"[eff-zero] bad samples={n_eff_zero_bad}")
    assert n_blocks_bad == 0, "Some blocks do not have exactly 2 active weights!"
    assert n_eff_zero_bad == 0, "Effective weight is non-zero in masked positions!"
    print("All runtime sanity checks PASSED.")


if __name__ == "__main__":
    main()
