from copy import deepcopy
import pytest
import torch
from torch import nn
from nmsam.core.batch import StepBatch
from nmsam.sparsity.registry import MaskRegistry
from nmsam.sparsity.training import sparse_step, sparse_loss
from nmsam.sam.reference_sgd import create_optimizer


def setup(ste=True):
    m=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.Tanh(),nn.Linear(4,3))
    r=MaskRegistry(m,ste=ste)
    b=StepBatch(torch.randn(6,4),torch.tensor([0,1,2,0,1,2]))
    return m,r,b


@pytest.mark.parametrize("ste",[True,False])
def test_sparse_forward_gradient_and_global_support(ste):
    m,r,b=setup(ste); before=deepcopy(m.state_dict()); effective=r.effective()
    for k,mask in r.masks.items():
        assert (effective[k][~mask]==0).all()
        assert torch.equal(dict(m.named_parameters())[k],before[k])
    report=sparse_step(m,create_optimizer(m.parameters()),b,r,l2_reg=0,capture_trace=True)
    for name,g,e,mask in zip([n for n,p in m.named_parameters()],report.trace.g1,
                              report.trace.epsilon,r.supports()):
        assert (e[~mask]==0).all()
        if name in r.masks:
            assert bool((g[~mask]!=0).any()) == ste
    assert report.epsilon_norm==pytest.approx(.05,abs=1e-7)
    assert m[1].num_batches_tracked==1 and r.version==0


def test_sparse_rho_zero_matches_direct_surrogate_sgd():
    m,r,b=setup(); ref=deepcopy(m); rr=MaskRegistry(ref)
    rr.load_state_dict(r.state_dict())
    opt=create_optimizer(m.parameters()); ropt=create_optimizer(ref.parameters())
    report=sparse_step(m,opt,b,r,rho=0,gradient_clipping=0)
    ropt.zero_grad(); sparse_loss(ref,b,rr).optimizer_loss.backward(); ropt.step()
    for p,q in zip(m.parameters(),ref.parameters()): torch.testing.assert_close(p,q)
    assert report.forward_passes==1


@pytest.mark.parametrize("phase",["forward","backward","optimizer","mask"])
def test_sparse_failure_restores_mask_master_bn_optimizer(phase):
    m,r,b=setup(); opt=create_optimizer(m.parameters()); saved=deepcopy(m.state_dict())
    masks=r.state_dict(); rng=torch.get_rng_state().clone()
    def hook(m,b,report):
        if phase=="mask":
            r.masks[next(iter(r.masks))].logical_not_()
        if phase=="forward": raise RuntimeError("forward")
        if phase=="backward":
            report.optimizer_loss.register_hook(lambda grad: (_ for _ in ()).throw(RuntimeError("backward")))
        return report
    if phase=="optimizer":
        real=opt.step
        def fail(): real(); raise RuntimeError("optimizer")
        opt.step=fail
    with pytest.raises(RuntimeError): sparse_step(m,opt,b,r,loss_hook=hook)
    for k,v in m.state_dict().items(): assert torch.equal(v,saved[k])
    assert all(torch.equal(r.masks[k],v) for k,v in masks["masks"].items())
    assert r.version==0 and not r.locked and torch.equal(torch.get_rng_state(),rng)
    assert not opt.state


def test_registry_update_forbidden_during_step():
    m,r,b=setup()
    def hook(m,b,report): r.update(r.masks); return report
    with pytest.raises(RuntimeError,match="during SAM"):
        sparse_step(m,create_optimizer(m.parameters()),b,r,loss_hook=hook)
