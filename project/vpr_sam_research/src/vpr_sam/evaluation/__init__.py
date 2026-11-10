"""Common external sphere probe; never adapted to a method's learned geometry."""
from __future__ import annotations
import torch
from torch import nn
from ..repro import finite


@torch.no_grad()
def evaluate(model, batches, device):
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    loss, correct, count = 0., 0, 0
    try:
        for x, y, _ in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            finite(logits, "evaluation logits")
            loss += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
    finally:
        for mod, flag in modes:
            mod.training = flag
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    return {"loss": loss/count, "accuracy": correct/count, "count": count}


def sphere_probe(model, x, y, rho, masks=None):
    """One first-order external SAM probe; not an exact robust-loss bound."""
    modes = [(m, m.training) for m in model.modules()]
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    gradients = {p: None if p.grad is None else p.grad.clone() for p in model.parameters()}
    try:
        model.eval()
        params = dict(model.named_parameters())
        loss = nn.functional.cross_entropy(model(x), y)
        finite(loss, "probe center loss")
        grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
        directions = []
        for (name, p), g in zip(params.items(), grads):
            direction = torch.zeros_like(p) if g is None else g
            if masks is not None and name in masks:
                direction = direction * masks[name].to(p.device)
            finite(direction, "probe gradient")
            directions.append(direction)
        norm = sum(g.double().square().sum() for g in directions).sqrt()
        with torch.no_grad():
            if float(norm) > 0:
                for p, g in zip(params.values(), directions):
                    p.add_((g.double() * (rho/norm)).to(p.dtype))
            adv = nn.functional.cross_entropy(model(x), y)
            finite(adv, "probe perturbed loss")
        return {"center_loss": float(loss.detach()), "adversarial_loss": float(adv),
                "increase": float(adv-loss.detach()), "zero_gradient": float(norm) == 0}
    finally:
        model.load_state_dict(saved)
        for p, grad in gradients.items():
            p.grad = grad
        for m, flag in modes:
            m.training = flag


@torch.no_grad()
def output_change(before, after, batches, device):
    before.eval(); after.eval()
    squared, kl, count = 0., 0., 0
    for x, _, _ in batches:
        x = x.to(device)
        a, b = before(x), after(x)
        finite(a, "pre-pruning logits"); finite(b, "post-pruning logits")
        squared += float((a-b).double().square().sum())
        kl += float(nn.functional.kl_div(b.log_softmax(-1), a.softmax(-1), reduction="sum"))
        count += len(x)
    return {"logit_squared_l2_per_sample": squared/count, "kl_dense_to_pruned": kl/count}
