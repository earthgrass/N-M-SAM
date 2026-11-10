import copy
import pytest
import torch
from torch import nn
from vpr_sam.models import build_model
from vpr_sam.optim import Stepper
from vpr_sam.pruning import make_masks,apply_masks,validate_masks,nm_mask
from vpr_sam.evaluation import sphere_probe


@pytest.mark.parametrize("method",["sgd","sam","vpr","static","positive","fixed_l2"])
def test_step_bn_and_one_base_update(config,method):
    config["optimizer"]["method"]=method
    model=build_model("tiny")
    opt=Stepper(model,config)
    calls=[]
    original=opt.base.step
    def counting(*a,**k):calls.append(1);return original(*a,**k)
    opt.base.step=counting
    inputs=[]
    hook=model.register_forward_pre_hook(lambda _,args:inputs.append(args[0].data_ptr()))
    info=opt.step(torch.randn(4,3,32,32),torch.tensor([0,1,2,3]))
    hook.remove()
    assert len(calls)==1 and opt.steps==1 and int(model.bn.num_batches_tracked)==1
    assert len(inputs)==(1 if method=="sgd" else 2) and len(set(inputs))==1
    assert model.bn.track_running_stats
    if method in {"sam","fixed_l2"}:assert info["eps_norm"]==pytest.approx(config["optimizer"]["rho"],rel=1e-6)
    assert opt.geometry.updates==int(method in {"vpr","positive","fixed_l2"})


def test_exception_restores_all_and_prevents_checkpoint(config):
    model=build_model("tiny");opt=Stepper(model,config)
    before={k:v.clone() for k,v in model.state_dict().items()}
    calls=[]
    def fail(_,args):
        calls.append(1)
        if len(calls)==2:raise RuntimeError("injected second forward failure")
    hook=model.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError,match="injected"):opt.step(torch.randn(4,3,32,32),torch.arange(4))
    hook.remove()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert opt.geometry.updates==0 and opt.geometry.candidate is None and model.bn.track_running_stats
    with pytest.raises(RuntimeError):opt.state_dict()


def test_restore_before_base_and_external_probe(config):
    model=build_model("tiny");opt=Stepper(model,config)
    before={k:p.clone() for k,p in model.named_parameters()}
    original=opt.base.step
    def check(*args,**kwargs):
        assert all(torch.equal(before[k],p) for k,p in model.named_parameters())
        return original(*args,**kwargs)
    opt.base.step=check
    x,y=torch.randn(4,3,32,32),torch.arange(4)
    opt.step(x,y)
    state={k:v.clone() for k,v in model.state_dict().items()}
    model.train();sphere_probe(model,x,y,.05)
    assert model.training and all(torch.equal(state[k],v) for k,v in model.state_dict().items())


@pytest.mark.parametrize("n,m",[(1,3),(3,8),(1,2)])
def test_nm_legality_ties_and_tails(n,m):
    w=torch.ones(2,19)
    mask=nm_mask(w,n,m,"dense")
    head=19//m*m
    assert (mask[:,:head].reshape(2,-1,m).sum(-1)==n).all()
    assert mask[:,head:].all() and mask[0,0]
    if head<19:
        with pytest.raises(ValueError):nm_mask(w,n,m,"error")


@pytest.mark.parametrize("method",["sgd","sam"])
def test_masked_recovery_cannot_regrow(config,method):
    config["optimizer"]["method"]=method
    model=build_model("tiny")
    masks,_=make_masks(model,config["pruning"]);apply_masks(model,masks)
    opt=Stepper(model,config,masks)
    for _ in range(3):opt.step(torch.randn(4,3,32,32),torch.arange(4))
    validate_masks(model,masks,config["pruning"])


def test_resnet_architecture():
    model=build_model("resnet18")
    assert model.conv1.kernel_size==(3,3) and model.conv1.stride==(1,1)
    assert not any(isinstance(m,nn.MaxPool2d) for m in model.modules())
    assert sum(isinstance(m,nn.Conv2d) for m in model.modules())==20
    assert model(torch.randn(2,3,32,32)).shape==(2,10)


@pytest.mark.parametrize("method",["sgd","sam","vpr"])
def test_matches_manual_outer_update(config,method):
    config["optimizer"].update(method=method,momentum=0.,weight_decay=0.)
    model=nn.Linear(3,2)
    reference=copy.deepcopy(model)
    opt=Stepper(model,config)
    x,y=torch.randn(5,3),torch.tensor([0,1,0,1,0])
    loss=nn.functional.cross_entropy(reference(x),y)
    loss.backward()
    if method!="sgd":
        old=[p.detach().clone() for p in reference.parameters()]
        norm=sum(p.grad.double().square().sum() for p in reference.parameters()).sqrt()
        with torch.no_grad():
            for p in reference.parameters():p.add_((p.grad.double()*config["optimizer"]["rho"]/norm).float())
        reference.zero_grad()
        nn.functional.cross_entropy(reference(x),y).backward()
        with torch.no_grad():
            for p,value in zip(reference.parameters(),old):p.copy_(value)
    with torch.no_grad():
        for p in reference.parameters():p.add_(p.grad,alpha=-config["optimizer"]["lr"])
    opt.step(x,y)
    assert all(torch.allclose(a,b,atol=1e-7,rtol=1e-6) for a,b in zip(model.parameters(),reference.parameters()))


def test_nonfinite_second_gradient_before_outer_update(config):
    model=build_model("tiny");opt=Stepper(model,config)
    before={name:value.clone() for name,value in model.state_dict().items()}
    calls=[]
    def poison(grad):
        calls.append(1)
        return grad if len(calls)==1 else grad*float("nan")
    hook=model.fc.weight.register_hook(poison)
    with pytest.raises(FloatingPointError,match="g2"):
        opt.step(torch.randn(4,3,32,32),torch.arange(4))
    hook.remove()
    assert opt.steps==0 and not opt.base.state
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_nonunit_ellipsoid_full_step_matches_independent_reference(config):
    """Covers non-unit A through g1 -> epsilon -> g2 -> SGD -> feedback."""
    torch.manual_seed(53)
    model=nn.Linear(3,2)
    reference=copy.deepcopy(model)
    opt=Stepper(model,config)
    o=config["optimizer"]
    reference_optimizer=torch.optim.SGD(reference.parameters(),lr=o["lr"],momentum=o["momentum"],
                                         weight_decay=o["weight_decay"],nesterov=o["nesterov"])
    s=torch.tensor([.2,-.1,.3,-.2,.1,-.3,.15,-.15],dtype=torch.float64)
    offset=0
    for state,p in zip(opt.geometry.logs,model.parameters()):
        state.copy_(s[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        opt.base.state[p]["momentum_buffer"]=torch.full_like(p,.03)
    for p in reference.parameters():reference_optimizer.state[p]["momentum_buffer"]=torch.full_like(p,.03)
    x,y=torch.randn(5,3),torch.tensor([0,1,0,1,1])
    center=[p.detach().clone() for p in reference.parameters()]
    nn.functional.cross_entropy(reference(x),y).backward()
    g1=torch.cat([p.grad.flatten().double() for p in reference.parameters()])
    a=s.exp()
    epsilon=(o["rho"]*a.square()*g1/(a*g1).norm()).float()
    with torch.no_grad():
        offset=0
        for p in reference.parameters():
            p.add_(epsilon[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    reference_optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(reference(x),y).backward()
    with torch.no_grad():
        for p,before in zip(reference.parameters(),center):p.copy_(before)
    reference_optimizer.step()
    info=opt.step(x,y)
    for actual,expected in zip(model.parameters(),reference.parameters()):
        assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
        assert torch.allclose(opt.base.state[actual]["momentum_buffer"],
                              reference_optimizer.state[expected]["momentum_buffer"],atol=1e-7,rtol=1e-6)
    p=epsilon.double().square()/epsilon.double().square().sum()
    expected_s=s+config["geometry"]["gamma"]*(1-len(s)*p)
    actual_s=torch.cat([v.flatten() for v in opt.geometry.logs])
    assert torch.allclose(actual_s,expected_s,atol=1e-9,rtol=1e-7)
    assert info["eps_norm"]==pytest.approx(float(epsilon.double().norm()),rel=1e-6)


def test_scope_alias_and_sparsity_denominators(config):
    from vpr_sam.pruning import sparsity_summary
    model=build_model("resnet18")
    old=dict(config["pruning"],scope="residual_and_head")
    new=dict(config["pruning"],scope="exclude_first_and_pointwise")
    a,report=make_masks(model,old)
    b,_=make_masks(model,new)
    assert set(a)==set(b) and "linear.weight" in b and "conv1.weight" not in b
    assert not any("shortcut.0" in name for name in b)
    assert all(torch.equal(a[name],b[name]) for name in a)
    totals=sparsity_summary(model,report)
    assert totals["target_complete_block_sparsity"]==pytest.approx(2/3)
    assert totals["target_layer_sparsity"] < totals["target_complete_block_sparsity"]  # classifier tail stays dense
    assert totals["global_parameter_sparsity"] < totals["target_layer_sparsity"]
    assert totals["mask_deleted_parameters"]==sum(int((~mask).sum()) for mask in a.values())
