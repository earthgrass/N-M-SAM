"""Audit mask identity and overlap for the completed Shared-Q Seed-0 screen."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def flatten(path: Path) -> torch.Tensor:
    masks = torch.load(path, map_location="cpu", weights_only=False)
    return torch.cat([masks[name].bool().flatten() for name in sorted(masks)])


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError("Mask coordinate mismatch")
    reference_pruned = ~reference
    candidate_pruned = ~candidate
    intersection = int((reference_pruned & candidate_pruned).sum())
    union = int((reference_pruned | candidate_pruned).sum())
    return {
        "coordinates": reference.numel(),
        "changed_coordinates": int((reference != candidate).sum()),
        "keep_agreement": float((reference == candidate).double().mean()),
        "pruned_jaccard": intersection / union,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--original-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = flatten(args.original_mask)
    hh = flatten(args.screen_dir / "HH_alpha_0" / "mask.pt")
    rows = []
    for label, candidate in [
        ("Hfixed_vs_original", flatten(args.screen_dir / "Hfixed_alpha_0" / "mask.pt")),
        ("HH_vs_original", hh),
    ]:
        rows.append({"comparison": label, "lambda": 0.0, **compare(original, candidate)})
    for alpha in (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0):
        qh = flatten(args.screen_dir / f"QH_alpha_{alpha:g}" / "mask.pt")
        qq = flatten(args.screen_dir / f"QQ_alpha_{alpha:g}" / "mask.pt")
        rows.append({"comparison": "Q_selection_vs_HH", "lambda": alpha, **compare(hh, qh)})
        rows.append({"comparison": "QH_vs_QQ_identity", "lambda": alpha, **compare(qh, qq)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
