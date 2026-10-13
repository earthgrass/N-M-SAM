import pytest
from nmsam.config.loader import load_dense_config
from nmsam.config.schema import DenseConfig


def test_configs():
    sam, raw = load_dense_config("configs/dense_sam_official_toy.yaml")
    sgd, _ = load_dense_config("configs/dense_sgd_toy.yaml")
    assert sam.rho == 0.05 and sgd.rho == 0 and "seed:" in raw


@pytest.mark.parametrize("kw", [dict(rho=-1), dict(lr=0), dict(rho=float("nan")),
    dict(momentum=1), dict(momentum=0), dict(batch_size=1), dict(steps=True),
    dict(nesterov="false"), dict(seed=-1)])
def test_invalid(kw):
    with pytest.raises(ValueError):
        DenseConfig(**kw)

