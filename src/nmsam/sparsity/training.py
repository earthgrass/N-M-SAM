from copy import deepcopy
import torch
from torch.nn import functional as F
from nmsam.core.loss import LossReport
from nmsam.sam.train_step import train_step
from nmsam.utils.seed import capture_rng, restore_rng


def sparse_loss(model, batch, registry, *, l2_reg=.001, regularization_space="master"):
    if regularization_space not in ("master","effective") or l2_reg < 0:
        raise ValueError("explicit master/effective nonnegative L2 required")
    logits=registry.forward(batch.inputs)
    task=F.cross_entropy(logits,batch.targets)
    parameters=dict(model.named_parameters()) if regularization_space=="master" else registry.effective()
    penalty=sum(l2_reg*.5*p.square().sum() for p in parameters.values())
    return LossReport(task,penalty,task+penalty,logits)


def sparse_step(model, optimizer, batch, registry, *, rho=.05, gradient_clipping=5.,
                l2_reg=.001, regularization_space="master", capture_trace=False, loss_hook=None):
    # Outer transaction also covers illegal registry mutation after optimizer.step.
    state=deepcopy(model.state_dict()); opt=deepcopy(optimizer.state_dict())
    grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    rng=capture_rng()
    try:
        with registry.frozen() as masks:
            def loss(m,b):
                registry.assert_unchanged(masks)
                report=sparse_loss(m,b,registry,l2_reg=l2_reg,regularization_space=regularization_space)
                if loss_hook:
                    report=loss_hook(m,b,report)
                registry.assert_unchanged(masks)
                return report
            return train_step(model,optimizer,batch,loss,rho=rho,gradient_clipping=gradient_clipping,
                              policy=registry,capture_trace=capture_trace)
    except BaseException:
        model.load_state_dict(state); optimizer.load_state_dict(opt); restore_rng(rng)
        for p,g in zip(model.parameters(),grads):
            p.grad=g
        raise
