"""New-support compensation; small exact reference and matrix-free Steihaug CG."""
import math
import torch
from .hvp import gradient,hvp
from .quadratic_robust import quadratic_maximum
from .robust_probe import project_ball,robust_proxy


def quadratic_compensation_reference(h,g,q,support,budget,damping=0.):
    if damping<0:raise ValueError("nonnegative damping required")
    shifted=h+damping*torch.eye(len(g),dtype=h.dtype,device=h.device)
    result=quadratic_maximum(-shifted,-(g+h@q),support,budget)
    v=result["delta"]
    active=h[support][:,support]
    return {"v":v,"solver":"quadratic_compensation_global_reference",
            "proxy_change":float((g+h@q)@v+.5*v@h@v),"damping":damping,
            "curvature_status":"empty_support" if active.numel()==0 else "positive_definite" if torch.linalg.eigvalsh(active).min()>0 else "indefinite_or_semidefinite_trust_region",
            "certificate":result}


def steihaug_compensation(linear,hessian_vector,support,budget,*,max_hvps=10,damping=1e-3,tol=1e-8):
    if not math.isfinite(budget) or budget<0 or max_hvps<1 or damping<0:
        raise ValueError("invalid compensation budget")
    v=torch.zeros_like(linear);r=-linear*support;p=r.clone();calls=0
    def operator(d):
        nonlocal calls
        calls+=1
        return hessian_vector(d)*support+damping*d
    def boundary(direction):
        pp=direction@direction;vp=v@direction
        root=torch.sqrt(torch.clamp(vp*vp+pp*(budget*budget-v@v),min=0))
        return (-vp+root)/pp,(-vp-root)/pp
    status="budget_exhausted"
    if r.norm()<=tol or budget==0:status="stationary_or_zero_budget"
    else:
        for _ in range(max_hvps):
            ap=operator(p);pap=p@ap
            if pap<=0:
                plus,minus=boundary(p)
                slope=-r@p
                tau=plus if plus*slope+.5*plus*plus*pap<=minus*slope+.5*minus*minus*pap else minus
                v=project_ball(v+tau*p,support,budget)
                status="negative_curvature_boundary";break
            rr=r@r;alpha=rr/pap;proposed=v+alpha*p
            if proposed.norm()>=budget:
                tau,_=boundary(p);v=project_ball(v+tau*p,support,budget)
                status="trust_region_boundary";break
            v=proposed;new_r=r-alpha*ap
            if new_r.norm()<=tol:status="converged";break
            p=new_r+(new_r@new_r)/rr*p;r=new_r
    return {"v":v,"solver":"steihaug_quadratic_proxy","status":status,
            "hvp_calls":calls,"backward_calls":2*calls,"damping":damping,"global_min_proven":False}


def compensate_task(loss,z,q,new_support,budget,**options):
    g=gradient(loss,z)[1].detach();hq=hvp(loss,z,q)
    result=steihaug_compensation(g+hq,lambda v:hvp(loss,z,v),new_support,budget,**options)
    result["backward_calls"]+=3 # gradient + one HVP (two autograd.grad calls)
    result.update(task_before=float(loss(z+q)),task_after=float(loss(z+q+result["v"])))
    return result


def recovery_sgd(loss,center,support,budget,*,steps,lr=.01):
    """Task-only restricted recovery; matched autograd backward count, not time."""
    if steps<1 or lr<=0:raise ValueError("positive recovery steps/lr required")
    v=torch.zeros_like(center)
    for _ in range(steps):
        g=gradient(loss,center+v)[1].detach()
        v=project_ball(v-lr*g,support,budget).detach()
    return {"v":v,"backward_calls":steps,"task_after":float(loss(center+v)),
            "solver":"restricted_task_recovery_sgd"}


def robust_compensation(loss,center,support,budget,rho,*,outer_steps=1,inner_steps=1,restarts=1,lr=.01):
    if outer_steps<1 or lr<=0:raise ValueError("positive outer budget required")
    v=torch.zeros_like(center)
    for _ in range(outer_steps):
        inner=robust_proxy(loss,center+v,support,rho,steps=inner_steps,restarts=restarts)
        g=gradient(loss,center+v+inner["delta"])[1].detach()
        v=project_ball(v-lr*g,support,budget).detach()
    return {"v":v,"backward_calls":outer_steps*(inner_steps*restarts+1),
            "solver":"robust_compensation_proxy","global_min_proven":False,
            "derivative":"one_step_Danskin_surrogate","task_after":float(loss(center+v))}


def compare_compensation(loss,z,q,support,budget,rho,*,max_hvps=4,outer_steps=1,inner_steps=1):
    quadratic=compensate_task(loss,z,q,support,budget,max_hvps=max_hvps)
    robust=robust_compensation(loss,z+q,support,budget,rho,outer_steps=outer_steps,inner_steps=inner_steps)
    comparisons={"mask_only":{"v":torch.zeros_like(z),"backward_calls":0},
        "quadratic":quadratic,"robust":robust,
        "sgd_matched_quadratic":recovery_sgd(loss,z+q,support,budget,steps=quadratic["backward_calls"]),
        "sgd_matched_robust":recovery_sgd(loss,z+q,support,budget,steps=robust["backward_calls"])}
    for result in comparisons.values():
        v=result["v"]
        if v.norm()>budget+1e-6 or (v[~support]!=0).any():raise ArithmeticError("illegal compensation")
        measured=robust_proxy(loss,z+q+v,support,rho,steps=inner_steps)
        result.update(task_loss=float(loss(z+q+v)),robust_proxy=measured["value"],
                      measurement_backwards=measured["backward_evaluations"],v_norm=float(v.norm()))
    return comparisons
