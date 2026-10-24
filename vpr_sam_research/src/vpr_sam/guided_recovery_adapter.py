"""Passive safety and logging around the unchanged ordinary-SGD Stepper."""
import torch
from types import MethodType
from torch import nn
from .optim import Stepper


class AuditedRecoveryStepper(Stepper):
    def __init__(self,model,config,masks=None):
        super().__init__(model,config,masks)
        if self.method!='sgd' or masks is None or self.base.state or self.steps:
            raise RuntimeError('Require fresh ordinary SGD and explicit frozen masks')
        self._latest={}
        self._gradient_violations=0
        original=self.base.step
        def checked_step(_optimizer,*args,**kwargs):
            violations=sum(int(torch.count_nonzero(self.geometry.params[i].grad[~mask]))
                for i,mask in self.masks.items() if self.geometry.params[i].grad is not None)
            if violations:raise RuntimeError('Masked gradients reached SGD')
            self._gradient_violations=violations
            return original(*args,**kwargs)
        self.base.step=MethodType(checked_step,self.base)

    def step(self,x,y):
        bns=[m for m in self.model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        before=[int(m.num_batches_tracked) for m in bns]
        info=super().step(x,y)
        violations=momentum=0
        for i,mask in self.masks.items():
            p=self.geometry.params[i]
            violations+=int(torch.count_nonzero(p.detach()[~mask]))
            buffer=self.base.state[p].get('momentum_buffer')
            if buffer is not None:momentum+=int(torch.count_nonzero(buffer[~mask]))
        if violations or momentum:raise RuntimeError('Masked parameter or momentum regrew')
        if any(int(m.num_batches_tracked)!=v+1 or not m.track_running_stats for m,v in zip(bns,before)):
            raise RuntimeError('BN SGD training statistics rule changed')
        if self.geometry.logs or self.geometry.updates or info['eps_norm']!=0:
            raise RuntimeError('Recovery must not use geometry or perturbations')
        info.update(mask_violations=violations,regrown_parameters=violations,
            masked_gradient_violations=self._gradient_violations,masked_momentum_violations=momentum,
            bn_batches_min=min(int(m.num_batches_tracked) for m in bns) if bns else None,
            bn_batches_max=max(int(m.num_batches_tracked) for m in bns) if bns else None)
        return info


def arithmetic_aulc(raw_accuracy,epochs):
    if [row['epoch'] for row in epochs]!=list(range(1,21)):
        raise ValueError('AULC requires the actual epochs 1..20 plus actual Raw Hard epoch0')
    values=[raw_accuracy]+[row['validation']['accuracy'] for row in epochs]
    if not all(__import__('math').isfinite(v) for v in values):raise ValueError('Nonfinite AULC input')
    return sum(values)/21
