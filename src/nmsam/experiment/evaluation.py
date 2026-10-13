from torch.nn import functional as F
import torch
from nmsam.utils.seed import capture_rng,restore_rng


@torch.no_grad()
def evaluate(model,loader,registry=None,*,max_batches=None):
    """Task-only sample-weighted eval, restoring every module's original mode."""
    modes=[(m,m.training) for m in model.modules()];device=next(model.parameters()).device
    buffers=[(b,b.clone()) for b in model.buffers()];rng=capture_rng()
    dtype=next(model.parameters()).dtype;total=correct=0;loss=0.
    try:
        model.eval()
        for index,(x,y) in enumerate(loader):
            if max_batches is not None and index>=max_batches:break
            x,y=x.to(device=device,dtype=dtype),y.to(device)
            logits=registry.forward(x) if registry else model(x)
            total+=len(y);correct+=(logits.argmax(-1)==y).sum().item()
            loss+=float(F.cross_entropy(logits,y,reduction="sum"))
    finally:
        for buffer,value in buffers:buffer.copy_(value)
        restore_rng(rng)
        for module,training in modes:module.training=training
    if total==0:raise ValueError("no evaluation samples")
    return {"task_CE":loss/total,"accuracy":correct/total,"samples":total,"objective":"task_CE",
            "derivative":"none","state_mode":"eval_fixed_buffers"}
