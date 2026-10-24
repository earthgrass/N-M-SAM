import torch
import pytest
from torch import nn
from core import sam_step


def test_exact_restore_and_global_perturbation(device):
    torch.manual_seed(2)
    model=nn.Linear(3,2).to(device)
    x=torch.tensor([[.3,1.,-.2],[-1.,.2,.7]],device=device)
    y=torch.tensor([0,1],device=device)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    torch.nn.functional.cross_entropy(model(x),y).backward()
    grads=[p.grad.clone() for p in model.parameters()]
    norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    perturbed=[p+g*(.1/(norm+1e-12)) for p,g in zip(before.values(),grads)]
    seen=[]
    hook=model.register_forward_pre_hook(lambda m,args:seen.append([p.detach().clone() for p in m.parameters()]))
    def restored(m):
        assert all(torch.equal(p,before[n]) for n,p in m.named_parameters())
    result=sam_step(model,torch.optim.SGD(model.parameters(),lr=.05),x,y,before_base_step=restored)
    hook.remove()
    assert all(torch.allclose(a,b) for a,b in zip(seen[1],perturbed))
    assert abs(result['perturbation_l2']-.1)<2e-7
    assert all(p.grad is None for p in model.parameters())


def test_second_forward_failure_restores_parameters(device):
    class Bad(nn.Linear):
        def forward(self,x):
            self.calls=getattr(self,'calls',0)+1
            return super().forward(x)*(float('nan') if self.calls==2 else 1.)
    model=Bad(3,2).to(device)
    saved=[p.detach().clone() for p in model.parameters()]
    with pytest.raises(FloatingPointError):
        sam_step(model,torch.optim.SGD(model.parameters(),lr=.05),torch.randn(4,3,device=device),torch.zeros(4,dtype=torch.long,device=device))
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),saved))
