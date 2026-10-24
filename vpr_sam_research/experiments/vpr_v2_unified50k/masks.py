"""Frozen 2:4 magnitude / |w|exp(-s), output-row groups, stable lower-index ties."""
import torch
from torch import nn
from state import tensor_sha


def build_masks(model,score,geometry_state=None):
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    logs={}
    if score=='geometry_guided':
        if geometry_state is None or geometry_state['method']!='fixed_l2': raise ValueError('Matching VPR geometry required')
        logs=dict((n,s) for (n,_),s in zip(geometry_state['mapping'],geometry_state['logs']))
        if [(n,tuple(p.shape)) for n,p in model.named_parameters()]!=geometry_state['mapping']:
            raise ValueError('Geometry mapping/order mismatch')
    elif score!='magnitude': raise ValueError('Unapproved mask score')
    masks={}
    for name,layer in layers[1:-1]:
        key=name+'.weight'; w=layer.weight.detach().cpu(); value=w.abs()
        if score=='geometry_guided':
            s=logs[key]
            if s.dtype!=torch.float64 or s.shape!=w.shape or not torch.isfinite(s).all(): raise ValueError('Geometry dtype/coordinates')
            value=value.double()*torch.exp(-s)
        if not torch.isfinite(value).all(): raise FloatingPointError('Mask scores')
        flat=value.reshape(value.shape[0],-1);complete=flat.shape[1]//4*4
        mask=torch.ones_like(flat,dtype=torch.bool)
        if complete:
            groups=flat[:,:complete].reshape(flat.shape[0],-1,4)
            keep=groups.argsort(dim=-1,descending=True,stable=True)[...,:2]
            head=torch.zeros_like(groups,dtype=torch.bool);head.scatter_(-1,keep,True)
            mask[:,:complete]=head.reshape(flat.shape[0],complete)
        masks[key]=mask.reshape_as(w)
    validate_masks(model,masks,check_zeros=False)
    return masks


def validate_masks(model,masks,check_zeros=True):
    params=dict(model.named_parameters())
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    if set(masks)!={n+'.weight' for n,_ in layers[1:-1]}: raise ValueError('Mask scope mismatch')
    for name,mask in masks.items():
        p=params[name]
        if mask.dtype!=torch.bool or mask.shape!=p.shape or p.ndim not in (2,4): raise ValueError('Mask mapping')
        flat=mask.reshape(mask.shape[0],-1);complete=flat.shape[1]//4*4
        if complete and not (flat[:,:complete].reshape(-1,4).sum(-1)==2).all(): raise ValueError('Illegal 2:4')
        if not flat[:,complete:].all(): raise ValueError('Tail must remain dense')
        if check_zeros and torch.count_nonzero(p.detach()[~mask.to(p.device)]): raise ValueError('Mask regrowth')


@torch.no_grad()
def project(model,masks,optimizer=None):
    for name,p in model.named_parameters():
        if name in masks:
            mask=masks[name].to(p.device);p.mul_(mask)
            if p.grad is not None:p.grad.mul_(mask)
            if optimizer is not None and 'momentum_buffer' in optimizer.state[p]:optimizer.state[p]['momentum_buffer'].mul_(mask)
