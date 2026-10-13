from copy import deepcopy
import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.core.functional_model import EffectiveLoss
from nmsam.sparsity.registry import MaskRegistry
from nmsam.diagnostics.hvp import hvp,tiny_hessian,finite_difference_hvp


@pytest.mark.parametrize("mode",["eval_fixed_buffers","train_fixed_batch"])
def test_functional_true_loss_new_activation_no_state_pollution(mode):
    m=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Tanh(),nn.Dropout(.2),nn.Linear(4,3)).double()
    r=MaskRegistry(m);b=StepBatch(torch.randn(5,4,dtype=torch.float64),torch.tensor([0,1,2,0,1]))
    before=deepcopy(m.state_dict());rng=torch.get_rng_state().clone();context=EffectiveLoss(m,b,r,state_mode=mode)
    z=context.z.clone();i=(~context.support).nonzero()[0].item();z[i]=1.
    assert context(z).item()!=context(context.z).item()
    torch.testing.assert_close(context(z),context(z),atol=0,rtol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    v=torch.randn_like(z);rng=torch.get_rng_state().clone()
    h=hvp(context,z,v);fd=finite_difference_hvp(context,z,v,h=1e-5)
    torch.testing.assert_close(h,fd,atol=2e-6,rtol=2e-5)
    for name,p in m.state_dict().items():assert torch.equal(p,before[name])
    assert torch.equal(torch.get_rng_state(),rng)
    with pytest.raises(ValueError):context(z[:-1])
    assert torch.equal(torch.get_rng_state(),rng)


def test_analytic_gradient_hessian_hvp():
    h=torch.tensor([[3.,1.,0.,1.],[1.,2.,1.,0.],[0.,1.,4.,0.],[1.,0.,0.,2.]],dtype=torch.float64)
    linear=torch.tensor([1.,-2.,.5,3.],dtype=torch.float64)
    loss=lambda z:linear@z+.5*z@h@z
    z=torch.tensor([2.,0.,0.,1.],dtype=torch.float64);v=torch.randn(4,dtype=torch.float64)
    from nmsam.diagnostics.hvp import gradient
    torch.testing.assert_close(gradient(loss,z)[1],linear+h@z)
    torch.testing.assert_close(tiny_hessian(loss,z),h)
    torch.testing.assert_close(hvp(loss,z,v),h@v)
    q=torch.tensor([-2.,3.,4.,-1.],dtype=torch.float64)
    torch.testing.assert_close(loss(z+q)-loss(z),(linear+h@z)@q+.5*q@h@q)
