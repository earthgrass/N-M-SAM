from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cifar100_50k10k_final_v1.yaml"
TRAINING = ROOT / "scripts" / "run_cifar100_50k_formal_training.py"


def load_training():
    spec = importlib.util.spec_from_file_location("cifar100_50k_training", TRAINING)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IdDataset(Dataset):
    def __len__(self):
        return 50_000

    def __getitem__(self, index):
        return index


def test_frozen_config_is_full_train_fixed_final():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol_id"] == "cifar100_50k10k_final_v1"
    assert config["dataset"]["train_size"] == 50_000
    assert config["dataset"]["test_size"] == 10_000
    assert config["dataset"]["validation_size"] == 0
    assert config["dataset"]["drop_last"] is False
    assert config["training"]["dense_epochs"] == 200
    assert config["training"]["selection_mode"] == "fixed_final_epoch"
    assert config["training"]["primary_checkpoint"] == "final_epoch_200.pt"
    assert config["recovery"]["epochs"] == 20
    assert config["recovery"]["primary_checkpoint"] == "final_epoch_020.pt"
    assert not any(config["evaluation"][key] for key in (
        "test_used_for_training", "test_used_for_checkpoint_selection", "test_used_for_hyperparameter_tuning"))


def test_epoch_sampler_reaches_every_official_train_position():
    module = load_training()
    seen = []
    batches = 0
    for values in module.training_loader(IdDataset(), batch_size=128, seed=0, epoch=0, workers=0):
        seen.extend(values.tolist())
        batches += 1
    assert batches == 391
    assert len(seen) == 50_000
    assert len(set(seen)) == 50_000
    assert set(seen) == set(range(50_000))


def test_training_entry_has_no_test_dataset_path_or_selection():
    source = TRAINING.read_text(encoding="utf-8")
    assert '/ "test"' not in source
    assert "best_test" not in source
    assert "best_val" not in source
    assert "validation_accuracy" not in source
    assert "early_stop" not in source
    assert 'primary_checkpoint": "final_epoch_200.pt"' in source


def test_old_protocol_checkpoint_cannot_validate_or_resume():
    module = load_training()
    tensor = torch.tensor([1.0])
    state = {"weight": tensor}
    checkpoint = {"schema": 1, "status": "healthy", "protocol_id": "cifar100_45k5k",
                  "epoch": 14, "model": state, "model_sha256": module.tensor_sha(state),
                  "stepper": {}, "scheduler": {}, "history": [], "identity": {}, "rng": {}}
    with pytest.raises(ValueError, match="another data/selection protocol"):
        module.validate_checkpoint(checkpoint)


def test_final_checkpoint_budget_is_exactly_200_epochs_78200_steps():
    source = TRAINING.read_text(encoding="utf-8")
    assert 'final_payload["epoch"] != 200' in source
    assert 'final_payload["steps"] != 78_200' in source
    assert 'output / "final_epoch_200.pt"' in source
