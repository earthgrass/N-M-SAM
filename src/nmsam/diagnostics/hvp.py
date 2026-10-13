import torch


def gradient(loss,z,*,create_graph=False):
    x=z.detach().clone().requires_grad_(True)
    value=loss(x)
    if value.ndim!=0 or not torch.isfinite(value):
        raise FloatingPointError("probe loss must be finite scalar")
    g=torch.autograd.grad(value,x,create_graph=create_graph)[0]
    if not torch.isfinite(g).all():raise FloatingPointError("nonfinite effective gradient")
    return x,g


def hvp(loss,z,v):
    if v.shape!=z.shape or v.device!=z.device or v.dtype!=z.dtype:
        raise ValueError("HVP direction layout/device/dtype mismatch")
    x,g=gradient(loss,z,create_graph=True)
    if not g.requires_grad:return torch.zeros_like(z)
    result=torch.autograd.grad((g*v).sum(),x,allow_unused=True)[0]
    if result is not None and not torch.isfinite(result).all():raise FloatingPointError("nonfinite HVP")
    return torch.zeros_like(z) if result is None else result.detach()


def finite_difference_hvp(loss,z,v,h=1e-4):
    if h<=0:raise ValueError("positive finite difference step required")
    return (gradient(loss,z+h*v)[1]-gradient(loss,z-h*v)[1])/(2*h)


def tiny_hessian(loss,z,max_dimension=64):
    if z.numel()>max_dimension:raise ValueError("full Hessian limited to analytic toy size")
    return torch.autograd.functional.hessian(loss,z.detach().clone())
