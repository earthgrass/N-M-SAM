import torch
from .hvp import gradient,hvp


def gradient_path(loss,z,support,rho):
    if rho<0:raise ValueError("nonnegative rho required")
    g1=gradient(loss,z)[1].detach();d=g1*support;norm=d.norm()
    epsilon=torch.zeros_like(z) if norm==0 else rho*d/norm
    g2=gradient(loss,z+epsilon)[1].detach()
    if epsilon.norm()==0:
        return {"status":"undefined_direction","g1":g1,"g2":g2,"epsilon":epsilon}
    u=epsilon/epsilon.norm();hu=hvp(loss,z,u)
    observed=(g2-g1)/epsilon.norm()
    return {"status":"defined","g1":g1,"g2":g2,"epsilon":epsilon,"u":u,
            "Hu":hu,"h_obs":observed,"path_average_error":float((observed-hu).norm())}


def subtract_l2_path(optimizer_gradient_difference,epsilon,l2_reg,regularized_coordinates):
    """Only for same-space compatible explicit diagonal L2; not arbitrary STE repair."""
    return optimizer_gradient_difference-l2_reg*regularized_coordinates*epsilon
