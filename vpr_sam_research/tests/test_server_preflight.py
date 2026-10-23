import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

spec = importlib.util.spec_from_file_location("preflight", Path(__file__).parents[1]/"scripts/preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_preflight_logical_index_is_not_physical_zero(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def properties(index):
        seen.append(index)
        return SimpleNamespace(name="NVIDIA GeForce RTX 5090", uuid=f"GPU-physical-{7-index}", major=12, minor=0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5")
    value = preflight.target_record("cuda:2")
    assert seen == [2] and value["gpu_uuid"] == "GPU-physical-5"
    assert value["resolved_device"] == "cuda:2" and value["cuda_visible_devices"] == "7,6,5"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_preflight_refuses_implicit_device(device):
    with pytest.raises(ValueError, match="explicit cuda:N"):
        preflight.target_record(device)


def test_preflight_requires_uuid(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(name="NVIDIA GeForce RTX 5090", major=12, minor=0))
    with pytest.raises(ValueError, match="UUID"):
        preflight.target_record("cuda:0")
