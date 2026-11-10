"""Recompute observed training costs from copied, immutable CSV evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


out = {}
for branch in ("sam", "apbr"):
    epoch = rows(f"{branch}_seed1_epoch_metrics.csv")
    if len(epoch) != 200 or int(epoch[-1]["steps"]) != 78200:
        raise RuntimeError(f"Incomplete {branch} training evidence")
    warm = [x for x in epoch if x["pruning_active"] == "False"]
    guided = [x for x in epoch if x["pruning_active"] == "True"]
    if len(warm) != 5 or len(guided) != 195:
        raise RuntimeError(f"Unexpected {branch} schedule")
    seconds = [float(x["epoch_seconds"]) for x in epoch]
    elapsed = float(epoch[-1]["elapsed_seconds"])
    epoch_sum = sum(seconds)
    item = {
        "epoch_count": len(epoch),
        "warmup_count": len(warm),
        "guided_count": len(guided),
        "total_elapsed_seconds": elapsed,
        "epoch_loop_seconds": epoch_sum,
        "outside_epoch_loop_seconds": elapsed - epoch_sum,
        "warmup_mean_epoch_seconds": sum(float(x["epoch_seconds"]) for x in warm) / len(warm),
        "guided_mean_epoch_seconds": sum(float(x["epoch_seconds"]) for x in guided) / len(guided),
        "peak_allocated_gib": max(int(x["peak_allocated_bytes"]) for x in epoch) / 2**30,
        "peak_reserved_gib": max(int(x["peak_reserved_bytes"]) for x in epoch) / 2**30,
    }
    if branch == "apbr":
        refresh = rows("apbr_seed1_hessian_refresh.csv")
        item["refresh_layer_records"] = len(refresh)
        item["refresh_rounds"] = len({int(x["epoch"]) for x in refresh})
        item["gram_seconds_sum"] = sum(float(x["gram_seconds"]) for x in refresh)
        item["factor_seconds_sum"] = sum(float(x["factor_seconds"]) for x in refresh)
    out[branch] = item

ROOT.joinpath("cost_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
