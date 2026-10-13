"""The five methods must be declarations over shared components only."""
from __future__ import annotations

import inspect

from nm_sam_cifar10.methods import METHOD_NAMES, get_method_spec


def test_five_methods_are_component_compositions():
    expected = {
        "dense_sgd": ("sgd", "dense", None, "sgd"),
        "dense_sam": ("sam", "dense", None, "sam"),
        "nm_sgd": ("sgd", "nm", "magnitude", "sgd"),
        "nm_sam": ("sam", "nm", "magnitude", "sam"),
        "nm_sam_aware": ("sam", "nm", "sam_taylor", "sam"),
    }
    assert set(METHOD_NAMES) == set(expected)
    for name, components in expected.items():
        spec = get_method_spec(name)
        assert (spec.optimizer, spec.sparsity, spec.mask_score, spec.training_step) == components


def test_method_modules_do_not_implement_training_loops():
    for name in METHOD_NAMES:
        module = __import__(f"nm_sam_cifar10.methods.{name}", fromlist=["METHOD"])
        source = inspect.getsource(module)
        assert "optimizer.step(" not in source
        assert "loss.backward(" not in source
        assert "for epoch" not in source
