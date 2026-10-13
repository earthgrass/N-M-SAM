from copy import deepcopy
import pytest
import torch
from torch import nn
from nmsam.sparsity.registry import MaskRegistry
from nmsam.experiment.evaluation import evaluate


def test_illegal_external_mask_rejected_before_step():
    model=nn.Linear(4,2);registry=MaskRegistry(model)
    registry.masks["weight"].zero_()
    with pytest.raises(ValueError,match="exactly N"):
        with registry.frozen():pass


def test_eval_restores_rng_and_registered_buffers():
    class Small(nn.Linear):
        def __init__(self):
            super().__init__(2,2);self.register_buffer("counter",torch.zeros(()))
        def forward(self,x):
            self.counter.add_(1)
            return super().forward(x)+torch.rand_like(x)
    model=Small();rng=torch.get_rng_state().clone()
    evaluate(model,[(torch.ones(2,2),torch.tensor([0,1]))])
    assert model.training and model.counter==0 and torch.equal(torch.get_rng_state(),rng)
