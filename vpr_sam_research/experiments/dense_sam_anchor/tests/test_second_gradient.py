import copy
import torch
from torch import nn
from core import sam_step


def test_g2_and_weight_decay_applied_at_original_parameters(device):
    torch.manual_seed(6)
    model=nn.Linear(4,3).to(device); reference=copy.deepcopy(model)
    x=torch.randn(8,4,device=device); y=torch.arange(8,device=device)%3
    original=[p.detach().clone() for p in reference.parameters()]
    loss=torch.nn.functional.cross_entropy(reference(x),y); loss.backward()
    g1=[p.grad.clone() for p in reference.parameters()]
    norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in g1]))
    with torch.no_grad():
        for p,g in zip(reference.parameters(),g1): p.add_(.1*g/(norm+1e-12))
    reference.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(reference(x),y).backward()
    g2=[p.grad.clone() for p in reference.parameters()]
    optimizer=torch.optim.SGD(model.parameters(),lr=.05,weight_decay=.001,momentum=.9)
    sam_step(model,optimizer,x,y)
    for actual,saved,g in zip(model.parameters(),original,g2):
        torch.testing.assert_close(actual,saved-.05*(g+.001*saved),rtol=1e-5,atol=1e-7)
    assert any(not torch.allclose(a,b) for a,b in zip(g1,g2))
