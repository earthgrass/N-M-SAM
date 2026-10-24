import copy
import io
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from prototypes.blockwise import BlockwiseGeometry,CandidateStepper

C=dict(dtype='float64',gamma=5.482145661475941e-8,warmup_steps=0,volume_atol=1e-10,volume_rtol=1e-12,static_std=.1)
torch.set_num_threads(2)

def geometry(m=4,n=2):
    torch.manual_seed(4)
    p=nn.Parameter(torch.ones(2,11)); b=nn.Parameter(torch.ones(2))
    g=BlockwiseGeometry([('weight',p),('bn.bias',b)],C,['weight'],m=m,n=n,gamma_block=.01)
    return g,p,b

def test_block_volume_tail_excluded_and_low_energy():
    g,p,b=geometry(); e=torch.tensor([[1e-40,2e-40,3e-40,4e-40,0,0,0,0,9,9,9]]*2)
    info=g.prepare([e,torch.ones(2)],0); assert info['zero_energy_blocks']==2
    g.commit(); g.check_local(g.logs)
    assert torch.equal(g.logs[1],torch.zeros(2,dtype=torch.float64))
    assert bool((g.logs[0][:,8:]==0).all())
    assert float(g.logs[0][:,:4].abs().max())>0

@pytest.mark.parametrize('n,m',[(2,4),(1,3),(3,5)])
def test_block_layout_mapping(n,m):
    g,p,b=geometry(m,n); e=torch.arange(1,23,dtype=torch.float32).reshape(2,11)
    g.prepare([e,torch.zeros(2)],0); g.commit()
    head=11//m*m
    for row in range(2):
        for start in range(0,head,m):
            energy=e[row,start:start+m].double().square()
            assert torch.allclose(g.logs[0][row,start:start+m],.01*(1-m*energy/energy.sum()),atol=1e-15)

def test_global_l2_and_zero_gradients():
    g,p,b=geometry(); g.prepare([torch.randn_like(p),torch.zeros_like(b)],0); g.commit()
    p.grad=torch.randn_like(p); b.grad=torch.randn_like(b)
    eps,info=g.perturbations(.05)
    assert abs(sum(e.double().square().sum() for e in eps).sqrt().item()-.05)<1e-8
    g.prepare(eps,1); g.commit()
    p.grad=torch.zeros_like(p); b.grad=torch.zeros_like(b)
    eps,info=g.perturbations(.05); old=copy.deepcopy(g.logs)
    g.prepare(eps,2,zero=True); g.commit()
    assert all(torch.equal(a,b) for a,b in zip(old,g.logs))

def test_checkpoint_exact_and_corruption_rejected():
    g,p,b=geometry(); g.prepare([torch.randn_like(p),torch.zeros_like(b)],0); g.commit()
    memory=io.BytesIO();torch.save(g.state_dict(),memory);memory.seek(0)
    state=torch.load(memory,weights_only=False); h,_,_=geometry(); h.load_state_dict(state)
    assert all(torch.equal(a,b) for a,b in zip(g.logs,h.logs))
    bad=copy.deepcopy(state);bad['gamma_block']=.02
    with pytest.raises(ValueError): h.load_state_dict(bad)
    # Excluded affine corruption cannot enter a valid state.
    bad=copy.deepcopy(state);bad['global_state']['logs'][1][0]=1.
    with pytest.raises((ValueError,FloatingPointError)): h.load_state_dict(bad)

def test_nonfinite_and_failed_block_volume():
    g,p,b=geometry()
    with pytest.raises(FloatingPointError):g.prepare([torch.full_like(p,float('nan')),torch.zeros_like(b)],0)
    broken=torch.zeros_like(p,dtype=torch.float64);broken[:,:8]=1.
    with pytest.raises(FloatingPointError):g.check_local([broken,g.logs[1]])

def test_cpu_smoke_bn_rule_and_resume_reproducible():
    torch.set_num_threads(2);torch.manual_seed(11)
    model=nn.Sequential(nn.Linear(8,8),nn.BatchNorm1d(8),nn.ReLU(),nn.Linear(8,3))
    c=json.loads((Path(__file__).parents[1]/'configs/smoke/fixed_l2.json').read_text())
    c['device']='cpu';c['purpose']='smoke'
    step=CandidateStepper(model,c,['0.weight'],gamma_block=.01)
    x=torch.randn(12,8);y=torch.randint(0,3,(12,))
    info=step.step(x,y);assert model[1].num_batches_tracked.item()==1
    assert abs(info['eps_norm']-c['optimizer']['rho'])<1e-7
    saved=copy.deepcopy(step.state_dict()); weights=copy.deepcopy(model.state_dict())
    step.step(x,y); ref=copy.deepcopy(model.state_dict())
    model.load_state_dict(weights);step.load_state_dict(saved);step.step(x,y)
    assert all(torch.equal(a,model.state_dict()[k]) for k,a in ref.items())
    assert model[1].num_batches_tracked.item()==2
    step.geometry.check_local(step.geometry.logs)
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())

def test_no_formal_entry():
    c=json.loads((Path(__file__).parents[1]/'configs/smoke/fixed_l2.json').read_text()); c['purpose']='formal'
    with pytest.raises(ValueError):CandidateStepper(nn.Linear(4,2),c,['weight'],gamma_block=.01)
