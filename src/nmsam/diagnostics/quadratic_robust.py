"""Small symmetric quadratic trust-region GLOBAL reference with KKT certificate.

Maximize b^T d + .5 d^T H d on a supported Euclidean ball. Includes negative
definite interior solutions and the hard case (b orthogonal to top eigenspace).
Only a toy reference; never forms the Hessian of an actual network.
"""
import math
import torch


def quadratic_maximum(h,b,support,rho,*,tol=1e-12,max_dimension=64):
    if (b.ndim!=1 or h.shape!=(len(b),len(b)) or support.shape!=b.shape
            or b.dtype!=torch.float64 or h.dtype!=torch.float64 or len(b)>max_dimension
            or support.dtype!=torch.bool or support.device!=b.device or h.device!=b.device):
        raise ValueError("small FP64 matching quadratic required")
    if not math.isfinite(rho) or rho<0 or not torch.isfinite(h).all() or not torch.isfinite(b).all():
        raise ValueError("finite quadratic and nonnegative radius required")
    if not torch.allclose(h,h.T,atol=tol,rtol=tol):raise ValueError("symmetric Hessian required")
    index=support.bool().nonzero().flatten();delta=torch.zeros_like(b)
    if rho==0 or not len(index):
        return {"delta":delta,"value":0.,"solver":"quadratic_global_reference","stationarity_error":0.,"multiplier":None}
    reduced=h[index][:,index];linear=b[index]
    eigenvectors_values,eigenvectors=torch.linalg.eigh(reduced)
    eigen=eigenvectors_values;projected=eigenvectors.T@linear
    lower=max(0.,float(eigen[-1]));denominator=lower-eigen
    singular=denominator.abs()<=tol*(1+eigen.abs().max())
    base=torch.where(singular,torch.zeros_like(projected),projected/torch.where(singular,torch.ones_like(denominator),denominator))
    regular_singular=bool((projected[singular].abs()<=tol*(1+linear.norm())).all())
    if regular_singular and base.norm()<=rho:
        coordinates=base
        if lower>0:
            coordinates=coordinates.clone()
            coordinates[int(torch.argmax(eigen))]=torch.sqrt(torch.clamp(rho*rho-base.square().sum(),min=0))
        multiplier=lower
    else:
        lo=lower;hi=max(lower+1.,float(linear.norm())/rho+float(eigen.abs().max())+1.)
        def coordinates_at(value):return projected/(value-eigen)
        while coordinates_at(hi).norm()>rho:hi=2*hi+1
        for _ in range(200):
            middle=(lo+hi)/2
            if coordinates_at(middle).norm()>rho:lo=middle
            else:hi=middle
            if hi-lo<=tol*(1+abs(hi)):break
        multiplier=hi;coordinates=coordinates_at(multiplier)
    active=eigenvectors@coordinates;delta[index]=active
    residual=(multiplier*active-reduced@active-linear).norm()
    psd_margin=multiplier-float(eigen[-1])
    complementarity=abs(multiplier*(float(active.norm())-rho))
    if residual>1e-8*(1+linear.norm()) or psd_margin < -1e-9 or complementarity>1e-8*(1+rho):
        raise ArithmeticError("quadratic global certificate failed")
    return {"delta":delta,"value":float(b@delta+.5*delta@h@delta),
            "solver":"quadratic_global_reference","stationarity_error":float(residual),
            "multiplier":multiplier,"psd_margin":psd_margin,"complementarity_error":complementarity}
