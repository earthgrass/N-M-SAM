import pytest
import torch

from joint_optimization.gradient_handoff import GradientHandoff, materialize_none_as_zeros


def test_handoff_preserves_none_and_cleans_without_accumulation():
    first = torch.nn.Parameter(torch.ones(3))
    unused = torch.nn.Parameter(torch.ones(2))
    adapter = GradientHandoff((first, unused))
    gradient = torch.tensor([1.0, 2.0, 3.0])
    with adapter.install((gradient, None)):
        assert torch.equal(first.grad, gradient)
        assert first.grad.data_ptr() != gradient.data_ptr()
        assert unused.grad is None
    assert first.grad is None and unused.grad is None
    with adapter.install((2 * gradient, None)):
        assert torch.equal(first.grad, 2 * gradient)
    assert first.grad is None


def test_handoff_rejects_dirty_destination_and_nested_use():
    parameter = torch.nn.Parameter(torch.ones(2))
    adapter = GradientHandoff((parameter,))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="not clean"):
        with adapter.install((torch.ones_like(parameter),)):
            pass
    parameter.grad = None
    with adapter.install((torch.ones_like(parameter),)):
        with pytest.raises(RuntimeError, match="Nested"):
            with adapter.install((torch.ones_like(parameter),)):
                pass


def test_handoff_cleans_after_exception_and_materializes_separately():
    parameter = torch.nn.Parameter(torch.ones(2))
    adapter = GradientHandoff((parameter,))
    with pytest.raises(ValueError):
        with adapter.install((torch.ones_like(parameter),)):
            raise ValueError("forced")
    assert parameter.grad is None and not adapter.active
    values = materialize_none_as_zeros((parameter,), (None,))
    assert torch.equal(values[0], torch.zeros_like(parameter))
