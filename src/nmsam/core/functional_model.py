"""Pure effective-space scalar CE. No STE and no reapplication of old support."""
from copy import deepcopy
import hashlib
import torch
from torch.func import functional_call
from torch.nn import functional as F
from nmsam.utils.seed import capture_rng,restore_rng


class EffectiveLoss:
    def __init__(self,model,batch,registry=None,*,state_mode="eval_fixed_buffers"):
        if state_mode not in ("eval_fixed_buffers","train_fixed_batch"):
            raise ValueError("explicit fixed state mode required")
        self.model=deepcopy(model)
        self.model.train(state_mode=="train_fixed_batch")
        self.inputs,self.targets=batch.inputs.detach().clone(),batch.targets.detach().clone()
        self.names=tuple(n for n,p in model.named_parameters())
        self.shapes=tuple(p.shape for p in model.parameters())
        self.counts=tuple(p.numel() for p in model.parameters())
        self.master=torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        masks=registry.masks if registry else {}
        self.support=torch.cat([masks[n].flatten() if n in masks else torch.ones_like(p,dtype=torch.bool).flatten()
                                for n,p in model.named_parameters()])
        self.z=self.master*self.support
        self.buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        self.rng=capture_rng()
        self.labels={"objective":"task_CE","derivative":"effective_true","state_mode":state_mode,
            "mask_id":registry.version if registry else "dense",
            "snapshot_id":hashlib.sha256(self.z.cpu().numpy().tobytes()).hexdigest()[:16]}

    def unpack(self,vector):
        if vector.shape!=self.z.shape or vector.device!=self.z.device or vector.dtype!=self.z.dtype:
            raise ValueError("effective vector layout/device/dtype mismatch")
        return {n:v.reshape(s) for n,v,s in zip(self.names,vector.split(self.counts),self.shapes)}

    def parameter_slice(self,name):
        i=self.names.index(name);start=sum(self.counts[:i])
        return slice(start,start+self.counts[i])

    def logits(self,vector):
        current=capture_rng()
        try:
            restore_rng(self.rng)
            buffers={n:b.clone() for n,b in self.buffers.items()}
            return functional_call(self.model,(self.unpack(vector),buffers),(self.inputs,),strict=True)
        finally:
            restore_rng(current)

    def __call__(self,vector):
        return F.cross_entropy(self.logits(vector),self.targets)
