import copy
import pytest
import torch
from vpr_sam.optim import Geometry


def geometry(config, grad, method="vpr"):
    p = torch.nn.Parameter(torch.zeros_like(grad, dtype=torch.float32))
    p.grad = grad.float().clone()
    g = Geometry([("weight", p)], config["geometry"], method)
    return p, g


def test_fixed_point_and_jacobian(config):
    grad = torch.tensor([.3, -2., 5., .01], dtype=torch.float64)
    d, gamma = len(grad), .13
    star = -.5 * grad.abs().log()
    star -= star.mean()
    def mapping(s):
        p = torch.softmax(4*s + 2*grad.abs().log(), 0)
        return s + gamma * (1-d*p)
    assert torch.allclose(torch.softmax(4*star+2*grad.abs().log(), 0), torch.full((d,), 1/d, dtype=torch.float64))
    assert torch.allclose(mapping(star), star)
    jac = torch.autograd.functional.jacobian(mapping, star)
    expected = torch.eye(d, dtype=torch.float64)-4*gamma*(torch.eye(d, dtype=torch.float64)-torch.ones(d,d,dtype=torch.float64)/d)
    assert torch.allclose(jac, expected, atol=1e-12)
    delta = 1e-5
    finite_difference = torch.stack([(mapping(star+delta*torch.eye(d,dtype=torch.float64)[i])-
                                      mapping(star-delta*torch.eye(d,dtype=torch.float64)[i]))/(2*delta)
                                     for i in range(d)], 1)
    assert torch.allclose(jac, finite_difference, atol=1e-8)
    v = torch.tensor([1., -1., 2., -2.], dtype=torch.float64)
    assert torch.allclose(jac@v, (1-4*gamma)*v)
    assert torch.allclose(jac@torch.ones(d,dtype=torch.float64), torch.ones(d,dtype=torch.float64))


@pytest.mark.parametrize("gamma,contracts", [(.1,True),(.25,True),(.4,True),(.51,False)])
def test_local_contraction(gamma, contracts):
    s = torch.tensor([1e-5,-1e-5],dtype=torch.float64)
    new = s + gamma*(1-2*torch.softmax(4*s,0))
    assert bool(new.norm() < s.norm()) == contracts


def test_boundary_not_classified_by_linearization():
    x = torch.tensor(.01,dtype=torch.float64)
    new = x-.5*torch.tanh(4*x)
    new2 = new-.5*torch.tanh(4*new)
    assert abs(float(new2)) < abs(float(x))


def test_formula_volume_feedback_and_equivalence(config):
    p, g = geometry(config, torch.tensor([1.,-3.,2.,4.]))
    g.logs[0].copy_(torch.tensor([.1,-.2,.3,-.2],dtype=torch.float64))
    a = g.logs[0].exp()
    eps, info = g.perturbations(.05)
    reference = .05*a.square()*p.grad.double()/(a*p.grad.double()).norm()
    assert torch.allclose(eps[0].double(),reference,atol=1e-9)
    shares = eps[0].double().square()/eps[0].double().square().sum()
    old = g.logs[0].clone()
    g.prepare(eps,0); g.commit()
    eta = g.d * config["geometry"]["gamma"]
    assert torch.allclose(g.logs[0],old+eta*(1/g.d-shares),atol=1e-14)
    assert abs(g.diagnostics()["log_volume"]) < 1e-12
    assert abs(info["ellipsoid_norm"]-.05)<1e-8
    assert abs(info["eps_norm"]-.05)>1e-5


def test_identity_equals_standard_sam(config):
    grad = torch.tensor([1.,2.,-3.,.1])
    _, vpr = geometry(config, grad)
    _, sam = geometry(config, grad, "sam")
    a, _ = vpr.perturbations(.05)
    b, _ = sam.perturbations(.05)
    assert torch.equal(a[0],b[0])


def test_global_not_layerwise(config):
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    params[0].grad=torch.tensor([10.,10.]); params[1].grad=torch.tensor([1.,1.])
    g=Geometry(list(zip(["a","b"],params)),config["geometry"])
    eps, _=g.perturbations(.05)
    g.prepare(eps,0);g.commit()
    assert float(g.logs[0].sum())<0<float(g.logs[1].sum())
    assert abs(sum(float(s.sum()) for s in g.logs))<1e-12


def test_zero_and_missing_gradient(config):
    p,g=geometry(config,torch.zeros(3))
    eps,info=g.perturbations(.05)
    assert info["zero_perturbation"]
    g.prepare(eps,0,True);g.commit()
    assert g.zero_skips==1 and g.updates==0
    p.grad=torch.tensor([1.,0.,0.])
    eps,_=g.perturbations(.05);g.prepare(eps,1);g.commit()
    assert float(g.logs[0][1])==pytest.approx(config["geometry"]["gamma"])
    p.grad=None
    _,info=g.perturbations(.05)
    assert info["zero_perturbation"] and g.d==3


def test_nonfinite_volume_and_dtype_rejected(config):
    p,g=geometry(config,torch.ones(3))
    p.grad[0]=float("nan")
    with pytest.raises(FloatingPointError):g.perturbations(.05)
    p.grad.fill_(1)
    g.logs[0][0]=1
    with pytest.raises(FloatingPointError):g.diagnostics()
    _,good=geometry(config,torch.ones(3))
    state=good.state_dict();state["logs"][0]=state["logs"][0].float()
    with pytest.raises(ValueError,match="dtype"):good.load_state_dict(state)


def test_concentrated_high_d_is_not_protected_by_local_bound(config):
    config["geometry"]["gamma"]=.1
    grad=torch.zeros(1000);grad[0]=1
    _,g=geometry(config,grad)
    eps,_=g.perturbations(.05)
    stats=g.prepare(eps,0);g.commit()
    assert stats["max_abs_delta_s"]==pytest.approx(99.9)
    assert g.logs[0][0] < -90


@pytest.mark.parametrize("dtype",["float32","float64"])
def test_long_sequence_volume(config,dtype):
    config["geometry"].update(dtype=dtype,volume_atol=1e-6,volume_rtol=1e-6)
    _,g=geometry(config,torch.arange(1,33,dtype=torch.float32))
    for step in range(100):
        eps,_=g.perturbations(.05);g.prepare(eps,step);g.commit()
    assert abs(g.diagnostics()["log_volume"])<1e-5
