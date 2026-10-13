"""Budgeted supported ascent; finite-step network values are PROXIES, not max."""
import math
import torch
from .hvp import gradient


def project_ball(delta,support,rho):
    value=delta*support;norm=value.norm()
    return value*(rho/norm) if norm>rho else value


def supported_ascent(value_fn,gradient_fn,z,support,rho,*,steps=1,restarts=1,step_size=None,seed=0):
    if (not math.isfinite(rho) or rho<0 or type(steps) is not int or type(restarts) is not int
            or steps<1 or restarts<1 or support.shape!=z.shape or support.dtype!=torch.bool
            or support.device!=z.device):
        raise ValueError("invalid robust probe budget/domain")
    if step_size is not None and (not math.isfinite(step_size) or step_size<=0):
        raise ValueError("positive finite step size required")
    generator=torch.Generator(device=z.device).manual_seed(seed)
    best_delta=torch.zeros_like(z);best=float(value_fn(best_delta));initial=[]
    if not math.isfinite(best):raise FloatingPointError("nonfinite initial inner value")
    for restart in range(restarts):
        delta=torch.zeros_like(z) if restart==0 else project_ball(torch.randn(z.shape,device=z.device,
                 dtype=z.dtype,generator=generator),support,rho)
        initial.append(float(value_fn(delta)))
        if not math.isfinite(initial[-1]):raise FloatingPointError("nonfinite restart value")
        if initial[-1]>best:best,best_delta=initial[-1],delta.clone()
        for _ in range(steps):
            g=gradient_fn(delta).detach()*support
            if not torch.isfinite(g).all():raise FloatingPointError("nonfinite inner gradient")
            scale=step_size if step_size is not None else (rho/float(g.norm()) if g.norm()>0 else 0.)
            delta=project_ball(delta+scale*g,support,rho).detach()
            value=float(value_fn(delta))
            if not math.isfinite(value):raise FloatingPointError("nonfinite inner value")
            if value>best:best,best_delta=value,delta.clone()
    return {"value":best,"delta":best_delta,"initial_values":initial,"steps":steps,
            "restarts":restarts,"backward_evaluations":steps*restarts,
            "solver":"one_step_robust_proxy" if steps==restarts==1 else "projected_ascent_proxy",
            "termination":"budget_exhausted","global_max_proven":False}


def robust_proxy(loss,z,support,rho,**budget):
    return supported_ascent(lambda d:loss(z+d),lambda d:gradient(loss,z+d)[1],z,support,rho,**budget)


def quadratic_proxy(constant,linear,hessian_vector,z,support,rho,**budget):
    return supported_ascent(lambda d:constant+linear@d+.5*d@hessian_vector(d),
                            lambda d:linear+hessian_vector(d),z,support,rho,**budget)
