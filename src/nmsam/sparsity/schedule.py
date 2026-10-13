"""Snapshot-aligned full-step-only dynamic mask update."""
from .mask import build_magnitude_mask


def masks_from_trace(registry, trace, scoring="magnitude"):
    if scoring not in ("magnitude","taylor_g1","taylor_g2"):
        raise ValueError("unknown mask scoring baseline")
    names=[n for n,p in registry.model.named_parameters() if p.requires_grad]
    weights=dict(zip(names,trace.base))
    grads=dict(zip(names,trace.g1 if scoring=="taylor_g1" else trace.g2))
    return {name:build_magnitude_mask(weights[name] if scoring=="magnitude"
                else (weights[name]*grads[name]).abs()) for name in registry.masks}


def update_after_step(registry, trace, step, interval, scoring):
    if interval < 1 or step < 1:
        raise ValueError("positive step/update interval required")
    if registry.locked:
        raise RuntimeError("schedule must run after full optimizer step")
    if step % interval:
        return 0.
    masks=masks_from_trace(registry,trace,scoring)
    changed=sum((registry.masks[n]!=m).sum().item() for n,m in masks.items())
    count=sum(m.numel() for m in masks.values())
    registry.update(masks)
    return changed/count
