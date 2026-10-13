"""YAML mapping -> strict DenseConfig; returns original text for provenance."""
from pathlib import Path
import yaml
from .schema import DenseConfig


def load_dense_config(path: str | Path) -> tuple[DenseConfig, str]:
    raw = Path(path).read_text(encoding="utf-8")
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("config must be a YAML mapping")
    return DenseConfig(**value), raw

