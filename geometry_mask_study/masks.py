"""Frozen-score diagnostics only: no training or geometry update."""
import math
import torch


@torch.no_grad()
def score_mask(scores,n,m,tail='dense'):
    if scores.ndim not in (2,4) or not 0<n<m or tail not in ('dense','error'):
        raise ValueError('Invalid mask layout/protocol')
    if not bool(torch.isfinite(scores).all()):raise FloatingPointError('Nonfinite mask score')
    flat=scores.reshape(scores.shape[0],-1)
    head=flat.shape[1]//m*m
    if head!=flat.shape[1] and tail=='error':raise ValueError('Incomplete block')
    mask=torch.ones_like(flat,dtype=torch.bool)
    if head:
        blocks=flat[:,:head].reshape(flat.shape[0],-1,m)
        chosen=blocks.argsort(dim=-1,descending=True,stable=True)[...,:n]
        keep=torch.zeros_like(blocks,dtype=torch.bool).scatter_(-1,chosen,True)
        mask[:,:head]=keep.reshape(flat.shape[0],head)
    return mask.reshape_as(scores)


@torch.no_grad()
def three_masks(weight,s,n=2,m=4,tail='dense'):
    if s.dtype!=torch.float64 or s.shape!=weight.shape:
        raise ValueError('FP64 geometry coordinates must match weight shape')
    if not bool(torch.isfinite(s).all()):raise FloatingPointError('Nonfinite log-radius')
    inverse_radius=(-s).exp()
    if not bool(torch.isfinite(inverse_radius).all()) or not bool((inverse_radius>0).all()):
        raise FloatingPointError('Nonrepresentable inverse radius; no clipping allowed')
    return dict(magnitude=score_mask(weight.double().abs(),n,m,tail),
        radius_only=score_mask(-s,n,m,tail),
        geometry_guided=score_mask(weight.double().abs()*inverse_radius,n,m,tail))


@torch.no_grad()
def deletion_metrics(weight,s,mask):
    displacement=weight.double()[~mask]
    scaled=displacement*(-s[~mask]).exp()
    euclidean_sq=float(displacement.square().sum())
    geometric_sq=float(scaled.square().sum())
    if not math.isfinite(euclidean_sq+geometric_sq):raise FloatingPointError('Nonfinite deletion cost')
    return dict(deleted_weights=int((~mask).sum()),displacement_squared_l2=euclidean_sq,
        displacement_l2=math.sqrt(euclidean_sq),geometric_deletion_cost=geometric_sq,
        geometric_deletion_norm=math.sqrt(geometric_sq))


def mask_difference(left,right,m=4):
    if left.shape!=right.shape:raise ValueError('Mask shape mismatch')
    a,b=left.reshape(left.shape[0],-1),right.reshape(right.shape[0],-1)
    head=a.shape[1]//m*m
    diff=a!=b
    blocks=diff[:,:head].reshape(a.shape[0],-1,m)
    flipped=int(diff.sum());complete=a.shape[0]*head;total=left.numel();count=blocks.numel()//m
    differing=int(blocks.any(-1).sum())
    return dict(flipped_coordinates=flipped,total_target_weights=total,complete_block_weights=complete,
        full_blocks=count,different_blocks=differing,mask_flip_rate=flipped/complete if complete else 0.,
        mask_flip_rate_including_dense_tail=flipped/total,
        different_block_fraction=differing/count if count else 0.)
