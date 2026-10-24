import copy
import pytest
import torch
from torch import nn
from core import sam_step
from masks import build_masks,project,validate_masks
from state import checkpoint_payload,atomic_checkpoint,load_checkpoint
from vpr_core.optim.geometry import Geometry

CFG=dict(gamma=5.482145661475941e-8,dtype='float64',warmup_steps=0,
         volume_atol=1e-10,volume_rtol=1e-12,static_std=.1)

def factory(device):
    return nn.Sequential(nn.Linear(4,8),nn.BatchNorm1d(8),nn.ReLU(),nn.Linear(8,8),nn.Linear(8,2)).to(device)

def test_anisotropic_fixed_l2_feedback_direction(device):
    p=nn.Parameter(torch.tensor([1.,2.,3.,4.],device=device))
    a=Geometry([('w',p)],CFG,'fixed_l2');b=Geometry([('w',p)],CFG,'vpr')
    logs=torch.tensor([-.9,-.2,.4,.7],dtype=torch.float64,device=device)
    a.logs=[logs.clone()];b.logs=[logs.clone()];p.grad=torch.tensor([1.,-2.,0.,.3],device=device)
    ea,info=a.perturbations(.1);eb,_=b.perturbations(.1)
    expected=torch.exp(2*logs)*p.grad.double();expected=.1*expected/expected.norm()
    assert torch.allclose(ea[0].double(),expected,rtol=2e-6,atol=1e-10)
    assert abs(info['eps_norm']/.1-1)<2e-5
    pa=ea[0].double().square();pb=eb[0].double().square()
    assert torch.allclose(pa/pa.sum(),pb/pb.sum(),rtol=2e-6,atol=1e-12)
    a.prepare(ea,0);a.commit();a.check_volume(a.logs)
    p.grad.zero_();e,z=a.perturbations(.1);a.prepare(e,1,z['zero_perturbation']);a.commit()
    assert a.updates==1 and a.zero_skips==1
    p.grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):a.perturbations(.1)

def test_geometry_exact_resume_and_dual_bn(tmp_path,device):
    m=factory(device);g=Geometry(m.named_parameters(),CFG,'fixed_l2')
    opt=torch.optim.SGD(m.parameters(),lr=.05,momentum=.9);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,200)
    x=torch.randn(6,4,device=device);y=torch.arange(6,device=device)%2
    sam_step(m,opt,x,y,geometry=g,successful_steps=0,audit_bn=True)
    atomic_checkpoint(tmp_path/'state.pt',checkpoint_payload(m,opt,sched,1,[{'steps':1}],{},None,g))
    sam_step(m,opt,x,y,geometry=g,successful_steps=1)
    expected=copy.deepcopy(m.state_dict());expected_g=g.state_dict()
    n=factory(device);h=Geometry(n.named_parameters(),CFG,'fixed_l2')
    o=torch.optim.SGD(n.parameters(),lr=.05,momentum=.9);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,200)
    saved=load_checkpoint(tmp_path/'state.pt',n,o,s,{});h.load_state_dict(saved['geometry'])
    sam_step(n,o,x,y,geometry=h,successful_steps=1)
    assert all(torch.equal(v,n.state_dict()[k]) for k,v in expected.items())
    assert all(torch.equal(v,w) for v,w in zip(expected_g['logs'],h.state_dict()['logs']))
    assert n[1].num_batches_tracked.item()==4 and h.updates==2
    bad=copy.deepcopy(saved['geometry']);bad['mapping']=bad['mapping'][::-1]
    with pytest.raises(ValueError):h.load_state_dict(bad)

def test_mask_scope_ties_geometry_and_fixed_sgd(device):
    m=factory(device);g=Geometry(m.named_parameters(),CFG,'fixed_l2')
    with torch.no_grad():m[3].weight.fill_(1)
    mag=build_masks(m,'magnitude');guided=build_masks(m,'geometry_guided',g.state_dict())
    assert set(mag)=={'3.weight'} and torch.equal(mag['3.weight'],guided['3.weight'])
    assert mag['3.weight'][0].tolist()==[True,True,False,False]*2
    changed=copy.deepcopy(g.state_dict());index=[n for n,_ in changed['mapping']].index('3.weight')
    changed['logs'][index][:,0]=2
    assert not torch.equal(mag['3.weight'],build_masks(m,'geometry_guided',changed)['3.weight'])
    project(m,mag);opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,weight_decay=.001)
    for _ in range(3):
        opt.zero_grad();m(torch.randn(6,4,device=device)).sum().backward();project(m,mag,opt);opt.step();project(m,mag,opt)
        validate_masks(m,mag)
        assert torch.count_nonzero(opt.state[m[3].weight]['momentum_buffer'][~mag['3.weight'].to(device)])==0
    with pytest.raises(ValueError):validate_masks(m,{})

def test_failed_second_forward_cannot_checkpoint_or_continue(device):
    m=factory(device);g=Geometry(m.named_parameters(),CFG,'fixed_l2');opt=torch.optim.SGD(m.parameters(),lr=.1)
    count=[0]
    def fail(module,args,out):
        count[0]+=1
        return out*float('nan') if count[0]==2 else out
    hook=m.register_forward_hook(fail)
    with pytest.raises(FloatingPointError):sam_step(m,opt,torch.randn(6,4,device=device),torch.arange(6,device=device)%2,geometry=g)
    hook.remove();assert g.updates==0 and g.candidate is None
    with pytest.raises(RuntimeError):checkpoint_payload(m,opt,None,0,[],{},None,g)
    with pytest.raises(RuntimeError):sam_step(m,opt,torch.randn(6,4,device=device),torch.arange(6,device=device)%2,geometry=g)
