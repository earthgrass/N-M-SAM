"""Isolated controlled repair candidates. NOT an accepted full SNOWS optimizer.

The objective, Newton regularizer and normal CG recurrences are unchanged.
Unresolved full solver/model/line-search semantics keep the method BLOCKED.
"""
import torch

def conjugate_gradient_sparse(hvp_fn, b, tol=1e-3, max_iter=5000, lambda_reg=1e-4):
    if not torch.isfinite(b).all():raise FloatingPointError('CG nonfinite RHS')
    if tol<0 or max_iter<1 or lambda_reg<0:raise ValueError('Invalid CG arguments')
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rs_old=torch.sum(r*r)
    if rs_old==0 or torch.sqrt(rs_old)<tol:return x,0
    for i in range(max_iter):
        Ap=hvp_fn(p)+lambda_reg*p
        if not torch.isfinite(Ap).all():raise FloatingPointError('CG nonfinite HVP')
        denominator=torch.sum(p*Ap)
        if not torch.isfinite(denominator) or denominator<=0:
            raise FloatingPointError('CG nonpositive/nonfinite curvature; no unapproved damping/fallback')
        alpha=rs_old/denominator
        x=x+alpha*p;r=r-alpha*Ap;rs_new=torch.sum(r*r)
        if not torch.isfinite(x).all() or not torch.isfinite(rs_new):raise FloatingPointError('CG nonfinite iterate')
        if rs_new==0 or torch.sqrt(rs_new)<tol:break
        p=r+(rs_new/rs_old)*p;rs_old=rs_new
    return x,i+1

def hessian_vector_product_chunks(grad_W, W, vector, mask, max_chunk_size=1e5):
    if mask.dtype!=torch.bool or mask.shape!=W.shape:raise ValueError('HVP Mask mapping')
    if vector.numel()!=int(mask.sum()) or max_chunk_size<1:raise ValueError('HVP vector/chunk shape')
    if not torch.isfinite(vector).all() or not torch.isfinite(grad_W).all():raise FloatingPointError('HVP nonfinite input')
    vector=vector.to(W.device);full_vector=torch.zeros_like(W);full_vector[mask]=vector
    hvp=torch.zeros_like(W);num_elements=int(mask.sum())
    if num_elements==0:return vector.clone()
    num_chunks=max(1,(num_elements+int(max_chunk_size)-1)//int(max_chunk_size))
    chunk_size=(num_elements+num_chunks-1)//num_chunks
    coordinates=torch.arange(num_elements,device=W.device)
    for i in range(num_chunks):
        start=i*chunk_size;end=min(start+chunk_size,num_elements)
        chunk_mask=torch.zeros_like(mask);chunk_mask[mask]=(coordinates>=start)&(coordinates<end)
        chunk_hvp=torch.autograd.grad(torch.sum(grad_W*full_vector*chunk_mask),W,retain_graph=True)[0]
        if not torch.isfinite(chunk_hvp).all():raise FloatingPointError('HVP nonfinite output')
        # H(sum v_k)=sum H v_k: retain ALL output rows, including cross-chunk terms.
        hvp+=chunk_hvp
    return hvp[mask]
