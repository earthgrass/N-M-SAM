from .hvp import hvp


def decompose(loss,z,q,u,hu):
    if abs(float(u.norm())-1)>1e-5:raise ValueError("unit direction required")
    a=u@q;r=q-a*u;hr=hvp(loss,z,r)
    parallel=a*a*(u@hu);cross=2*a*(r@hu);residual=r@hr
    return {"a":a,"r":r,"parallel":parallel,"cross":cross,"residual":residual,
            "quadratic_sum":parallel+cross+residual}
