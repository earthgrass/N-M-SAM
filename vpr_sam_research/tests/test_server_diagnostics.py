import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from vpr_sam.models import build_model
from vpr_sam.optim import Stepper
from vpr_sam.repro import seed_all, rng_state

spec = importlib.util.spec_from_file_location("server_validation", Path(__file__).parents[1]/"scripts/server_validation.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_global_energy_includes_none_and_zero_coordinates():
    p = torch.nn.Parameter(torch.zeros(3)); q = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([1., 2., 0.])
    result = diagnostics.gradient_energy([("p", p), ("q", q)])
    assert result["d"] == 5 and result["c_max"] == 4
    assert result["max_abs_one_minus_c"] == 3 and result["nonzero_fraction"] == .4
    assert result["layer_energy"] == {"p":1., "q":0.}
    p.grad.zero_()
    assert diagnostics.gradient_energy([("p", p), ("q", q)])["zero"]
    p.grad[0] = float("nan")
    with pytest.raises(FloatingPointError): diagnostics.gradient_energy([("p", p)])


@pytest.mark.parametrize("method", ["sam", "vpr"])
def test_observer_does_not_change_update_or_rng(config, method):
    seed_all(17)
    config = copy.deepcopy(config); config["optimizer"]["method"] = method
    model = build_model("tiny"); other = copy.deepcopy(model)
    x = torch.randn(4, 3, 32, 32); y = torch.arange(4)
    normal, observed = Stepper(model, config), Stepper(other, config)
    before_rng = rng_state()
    expected = normal.step(x, y)
    sampler = diagnostics.Observer(observed, sample_energy=True); sampler.begin()
    result = observed.step(x, y); sampler.finish(); sampler.close()
    assert diagnostics.identical(normal.state_dict(), observed.state_dict())
    assert diagnostics.identical(model.state_dict(), other.state_dict())
    assert diagnostics.identical(expected, result)
    assert diagnostics.identical(before_rng, rng_state())
    assert sampler.energy["d"] == observed.geometry.d


def test_quantile_and_empty_statistics():
    result = diagnostics.distribution([1,2,3])
    assert result["p99"] == 2.98 and result["median"] == 2
    with pytest.raises(ValueError): diagnostics.distribution([])
