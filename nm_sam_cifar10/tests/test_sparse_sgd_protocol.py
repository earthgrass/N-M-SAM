"""The corrected sparse-SGD protocol must remain entirely SAM-free."""
from dataclasses import fields
import inspect

import torch

from nm_sam_cifar10.strong_sparse_sgd import (
    StrongSparseSGDConfig,
    _setup,
    sparse_sgd_step,
)


def test_sparse_sgd_protocol_contains_no_sam_mechanism():
    config_fields = {field.name for field in fields(StrongSparseSGDConfig)}
    assert "rho" not in config_fields
    assert "adaptive" not in config_fields

    source = inspect.getsource(sparse_sgd_step)
    for forbidden in ("rho", "first_step", "second_step", "adversarial"):
        assert forbidden not in source

    setup_source = inspect.getsource(_setup)
    assert 'build_optimizer(model, "sgd"' in setup_source
    assert 'build_optimizer(model, "sam"' not in setup_source
    assert issubclass(torch.optim.SGD, torch.optim.Optimizer)
