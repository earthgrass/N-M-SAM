"""Standard global SAM; intentionally independent of frozen VPR modules."""
import torch
from torch import nn
from torch.nn import functional as F


def require_finite(values, label):
    values = list(values)
    if values and not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
        raise FloatingPointError(label)


def global_norm(values):
    values = list(values)
    if not values:
        raise RuntimeError('No gradients')
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v) for v in values]))
    require_finite([norm], 'Non-finite gradient norm')
    return norm


def bn_snapshot(model):
    return {name: {key: value.detach().clone() for key in
                  ('running_mean', 'running_var', 'num_batches_tracked')
                  if (value := getattr(module, key)) is not None}
            for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}


def sam_step(model, optimizer, images, labels, rho=0.1, numerical_epsilon=1e-7,
             before_base_step=None, audit_bn=False):
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    bn_before = bn_snapshot(model) if audit_bn else None
    center_output = model(images)
    center_loss = F.cross_entropy(center_output, labels)
    require_finite([center_loss], 'Center loss')
    center_loss.backward()
    bn_center = bn_snapshot(model) if audit_bn else None
    grads = [p.grad for p in parameters if p.grad is not None]
    norm1 = global_norm(grads)
    original = [p.detach().clone() for p in parameters]
    perturbations = [torch.zeros_like(p) if p.grad is None else
                     p.grad.detach() * (rho/(norm1+numerical_epsilon)) for p in parameters]
    require_finite(perturbations, 'Perturbation')
    try:
        with torch.no_grad():
            for p, epsilon in zip(parameters, perturbations):
                p.add_(epsilon)
        optimizer.zero_grad(set_to_none=True)
        adversarial_loss = F.cross_entropy(model(images), labels)
        require_finite([adversarial_loss], 'Adversarial loss')
        adversarial_loss.backward()
        bn_second = bn_snapshot(model) if audit_bn else None
        norm2 = global_norm(p.grad for p in parameters if p.grad is not None)
    finally:
        # Exact copies avoid cancellation error from subtracting the perturbation.
        with torch.no_grad():
            for p, saved in zip(parameters, original):
                p.copy_(saved)
    if before_base_step is not None:
        before_base_step(model)
    optimizer.step()  # g2 survives until here; weight decay is applied only here.
    require_finite(model.state_dict().values(), 'Model or BN state after SGD')
    optimizer.zero_grad(set_to_none=True)
    audit = {}
    if audit_bn:
        for name, before in bn_before.items():
            first, second = bn_center[name], bn_second[name]
            if (first['num_batches_tracked'].item()!=before['num_batches_tracked'].item()+1
                or second['num_batches_tracked'].item()!=first['num_batches_tracked'].item()+1):
                raise RuntimeError('BN did not update exactly twice: '+name)
            audit[name]={'first_counter':first['num_batches_tracked'].item(),
                         'second_counter':second['num_batches_tracked'].item(),
                         'first_mean_change':(first['running_mean']-before['running_mean']).abs().max().item(),
                         'second_mean_change':(second['running_mean']-first['running_mean']).abs().max().item(),
                         'first_var_change':(first['running_var']-before['running_var']).abs().max().item(),
                         'second_var_change':(second['running_var']-first['running_var']).abs().max().item()}
    return {'train_loss': center_loss.item(), 'adversarial_train_loss': adversarial_loss.item(),
            'correct': int((center_output.argmax(1)==labels).sum()), 'count': len(labels),
            'g1_norm': norm1.item(), 'g2_norm': norm2.item(),
            'perturbation_l2': global_norm(perturbations).item(), 'bn_audit':audit}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    before = bn_snapshot(model)
    total_loss, correct, count = 0., 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        output = model(x)
        loss = F.cross_entropy(output, y, reduction='sum')
        require_finite([loss], 'Evaluation loss')
        total_loss += loss.item()
        correct += int((output.argmax(1)==y).sum())
        count += len(y)
    after = bn_snapshot(model)
    assert all(torch.equal(t, after[n][k]) for n,b in before.items() for k,t in b.items())
    if count == 0:
        raise RuntimeError('Empty evaluation')
    return {'accuracy':correct/count, 'loss':total_loss/count, 'count':count}
