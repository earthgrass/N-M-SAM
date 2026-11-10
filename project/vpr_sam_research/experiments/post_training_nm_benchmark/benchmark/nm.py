"""Canonical OIHW/output-row-contiguous N:M contract; no APBR score changes."""
import torch
from torch import nn

CONVENTION='oihw_output_row_contiguous_v1'

def eligible_layers(model):
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    return dict(layers[1:-1])

def groups(layer,m=4):
    columns=layer.weight[0].numel();return torch.arange(columns//m*m).reshape(-1,m)

def magnitude_mask(weight,n=2,m=4):
    if not 0<n<m:raise ValueError('Require0<N<M')
    flat=weight.detach().abs().reshape(weight.shape[0],-1);end=flat.shape[1]//m*m
    mask=torch.ones_like(flat,dtype=torch.bool)
    if end:
        blocks=flat[:,:end].reshape(flat.shape[0],-1,m)
        selected=blocks.argsort(dim=-1,descending=True,stable=True)[...,:n]
        h=torch.zeros_like(blocks,dtype=torch.bool);h.scatter_(-1,selected,True)
        mask[:,:end]=h.reshape(flat.shape[0],end)
    return mask.reshape_as(weight)

def validate_layer_mask(weight,mask,n=2,m=4,check_zeros=True):
    if mask.dtype!=torch.bool or mask.shape!=weight.shape or weight.ndim not in (2,4):raise ValueError('Mask shape/dtype')
    flat=mask.reshape(mask.shape[0],-1);end=flat.shape[1]//m*m
    if end and not bool((flat[:,:end].reshape(-1,m).sum(1)==n).all()):raise ValueError('Canonical N:M violation')
    if not bool(flat[:,end:].all()):raise ValueError('Tail must stay dense')
    if check_zeros and torch.count_nonzero(weight.detach()[~mask.to(weight.device)]):raise ValueError('Pruned coordinate regrowth')

def validate_masks(model,masks,n=2,m=4,complete=True):
    layers=eligible_layers(model)
    expected={name+'.weight' for name in layers}
    if (complete and set(masks)!=expected) or not set(masks)<=expected:raise ValueError('Scope excludes first/last,BN,bias')
    params=dict(model.named_parameters())
    for name,mask in masks.items():validate_layer_mask(params[name],mask,n,m)

@torch.no_grad()
def project(model,masks,optimizer=None):
    for name,p in model.named_parameters():
        if name not in masks:continue
        mask=masks[name].to(p.device);p.mul_(mask)
        if p.grad is not None:p.grad.mul_(mask)
        state={} if optimizer is None else optimizer.state.get(p,{})
        if 'momentum_buffer' in state:state['momentum_buffer'].mul_(mask)
