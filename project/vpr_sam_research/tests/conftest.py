import copy
import json
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/smoke/vpr.json").read_text())


@pytest.fixture
def isolated_root(tmp_path):
    # Test IO is redirected under project/artifacts by pytest's basetemp option.
    root = tmp_path / "independent"
    (root / "src/vpr_sam").mkdir(parents=True)
    (root / "pyproject.toml").write_text("# test project marker\n")
    return root
