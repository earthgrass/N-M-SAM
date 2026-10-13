"""Metrics logging: CSV + JSON + per-method sanity checks."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_time: float
    cumulative_time: float
    generalization_gap: Optional[float] = None
    center_loss: Optional[float] = None
    sharpness: Optional[float] = None
    worst_case_loss: Optional[float] = None
    mask_change_ratio: Optional[float] = None
    active_weight_ratio: Optional[float] = None
    layer_sparsity: Optional[str] = None  # JSON-encoded per-layer dict
    configured_sparsity: Optional[float] = None
    actual_sparsity: Optional[float] = None
    total_nm_blocks: Optional[int] = None
    bad_block_count: Optional[int] = None
    layer_mask_change_ratio: Optional[str] = None
    mask_stability: Optional[float] = None
    iterations_per_second: Optional[float] = None
    images_per_second: Optional[float] = None
    grad1_norm: Optional[float] = None
    grad2_norm: Optional[float] = None
    epsilon_norm: Optional[float] = None
    epsilon_norm_over_rho: Optional[float] = None
    sam_skipped: Optional[bool] = None
    max_abs_logit: Optional[float] = None
    nan_detected: Optional[bool] = None
    inf_detected: Optional[bool] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class CSVLogger:
    """Append-only logger for per-epoch metrics."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._rows: List[Dict[str, Any]] = []
        self._columns: Optional[List[str]] = None

    def append(self, record: EpochRecord) -> None:
        d = asdict(record)
        row: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        self._rows.append(row)
        if self._columns is None:
            self._columns = list(row.keys())
        else:
            for k in row.keys():
                if k not in self._columns:
                    self._columns.append(k)

    def flush(self) -> None:
        if not self._rows:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        cols = self._columns or list(self._rows[0].keys())
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class JSONConfigLogger:
    """Persists the run config and the final summary."""

    def __init__(self, path: str, config: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.config = dict(config)

    def save_config(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"config": self.config}, f, indent=2, ensure_ascii=False, default=_json_safe)

    def append_summary(self, summary: Dict[str, Any]) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"config": self.config}
        data["summary"] = summary
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_safe)


def _json_safe(obj: Any) -> Any:
    try:
        return json.dumps(obj)
    except TypeError:
        return str(obj)


class FileLogger:
    """Plain ``print``-style logger mirrored to ``train.log``."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("")

    def write(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
