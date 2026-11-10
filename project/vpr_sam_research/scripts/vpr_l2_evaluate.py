"""Final-only validation, raw hard, train-only BN calibration and fixed probes.

No recovery, test evaluation, or optimizer step exists in this evaluator.
Historical V1 data are audited JSON/CSV copies; V1 checkpoints are never loaded.
"""
import copy
import csv
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vpr_sam.data import build_data, Images, loader
from vpr_sam.models import build_model
from vpr_sam.optim.geometry import Geometry
from vpr_sam.evaluation import evaluate
from vpr_sam.repro import atomic_json, file_hash, seed_all, seed_for, tensor_hash, source_hashes
from vpr_sam.l2_protocol import OUTPUT, verify_final, validate_protocol
from vpr_sam.training.engine import read_checkpoint, prune, evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vpr_l2'


def endpoint(model, geometry, dataset):
    """Eval-mode same-point spherical counterfactual, no training-state change."""
    model.eval()
    initial = tensor_hash(model.state_dict())
    rows = []
    pars = list(model.parameters())
    device = pars[0].device
    for batch,(x,y,ids) in enumerate(loader(dataset,128,0,workers=0)):
        x,y = x.to(device),y.to(device)
        model.zero_grad(set_to_none=True)
        center = nn.functional.cross_entropy(model(x),y)
        center.backward()
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in pars]
        eps, stats = geometry.perturbations(.05)
        norm = sum(g.double().square().sum() for g in gradients).sqrt()
        dot = sum((g.double()*e.double()).sum() for g,e in zip(gradients,eps))
        saved = [p.detach().clone() for p in pars]
        @torch.no_grad()
        def adversarial(directions):
            try:
                for p,b,e in zip(pars,saved,directions): p.copy_(b+e)
                return float(nn.functional.cross_entropy(model(x),y))
            finally:
                for p,b in zip(pars,saved): p.copy_(b)
        own = adversarial(eps)
        spherical = adversarial([(g.double()*(.05/norm)).float() for g in gradients])
        rows.append(dict(batch=batch,count=len(y),center_loss=float(center.detach()),
            fixed_l2_adversarial_loss=own,same_point_sam_adversarial_loss=spherical,
            R_epsilon=stats['eps_norm']/.05,direction_cosine=float(dot/(norm*stats['eps_norm'])),
            linear_strength=float(dot),linear_actual_ratio=float(dot/(.05*norm)),
            actual_loss_increase=own-float(center.detach()),
            spherical_loss_increase=spherical-float(center.detach())))
    model.zero_grad(set_to_none=True)
    if tensor_hash(model.state_dict()) != initial:
        raise RuntimeError('Endpoint diagnostic mutated model/BN state')
    return rows


def bn_calibrate(model, train, validation):
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    initial = tensor_hash(dict(model.named_parameters()))
    modules = [m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    for m in modules:
        m.reset_running_stats();m.momentum=None;m.train()
    with torch.no_grad():
        for x,_,_ in loader(train,128,0,workers=0): model(x.to(device))
    if tensor_hash(dict(model.named_parameters())) != initial or any(int(m.num_batches_tracked)!=32 for m in modules):
        raise RuntimeError('BN-only calibration violated frozen protocol')
    model.eval()
    result = evaluate(model,loader(validation,256,0,workers=0),device)
    return dict(metrics=result,train_parameters_unchanged=True,parameter_sha256=initial,
        bn_modules=len(modules),batches_per_bn=32,samples=4096,augmentation=False,
        algorithm='reset_running_stats; momentum=None; BN train, all others eval',
        test_used=False,optimizer_steps=0)


def main():
    c,_ = validate_protocol(ROOT)
    gates = json.loads((OUT/'control/gates.json').read_text())
    if source_hashes(ROOT) != gates['source_hashes']:
        raise RuntimeError('Frozen source mismatch')
    torch.set_num_threads(c['threads']);seed_all(seed_for(0,'evaluation'),True)
    final = ROOT/OUTPUT/'final.pt'
    state,_ = read_checkpoint(final,ROOT)
    verify_final(state,gates)
    if state['config'] != c or state['environment'] != gates['environment']:
        raise RuntimeError('Final config/environment identity mismatch')
    bn_original = {k:v.clone() for k,v in state['model'].items()
        if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    # Validate checkpoint state loads strictly, including SGD and FP64 geometry.
    from vpr_sam.l2_observer import ObservedStepper
    model = build_model(c['model']).cuda();model.load_state_dict(state['model'])
    stepper = ObservedStepper(model,c);stepper.load_state_dict(state['stepper'])
    stepper.geometry.check_volume(stepper.geometry.logs)
    if tensor_hash(model.state_dict()) != tensor_hash(state['model']):
        raise RuntimeError('Final model load mismatch')
    datasets,manifest = build_data(c,ROOT)
    fixed = json.loads((ROOT/'docs/vpr_l2_reference/fixed_sample_ids.json').read_text())['0']
    if fixed['bn_train'] != manifest['train_indices'][:4096]:
        raise RuntimeError('BN sample mismatch')
    base = datasets['validation']
    def subset(indices):
        return Images(base.images,base.labels,indices,0,augment=False,cutout=0)
    endpoint_rows = endpoint(model,stepper.geometry,subset(fixed['validation']))
    atomic_json(OUT/'endpoint_probes.json',endpoint_rows)
    geometry = stepper.geometry.diagnostics()
    del model,stepper,state
    gc.collect();torch.cuda.empty_cache()
    dense_probe = evaluate_checkpoint(final,ROOT,'validation',True)
    pruned = prune(final,ROOT)
    hard = ROOT/pruned['checkpoint']
    hard_probe = evaluate_checkpoint(hard,ROOT,'validation',True)
    hard_state,_ = read_checkpoint(hard,ROOT)
    if any(not torch.equal(hard_state['model'][k],v) for k,v in bn_original.items()):
        raise RuntimeError('Raw hard changed original BN running buffers')
    if hard_state['parent']['sha256'] != file_hash(final):
        raise RuntimeError('Hard pruning final source mismatch')
    model = build_model(c['model']).cuda();model.load_state_dict(hard_state['model'])
    # Independent ephemeral copy loaded from raw hard; its buffers are not saved
    # back into raw hard artifacts and trainable parameters remain unchanged.
    bn = bn_calibrate(model,subset(fixed['bn_train']),datasets['validation'])
    bn['source_checkpoint_sha256'] = file_hash(hard)
    bn['fixed_sample_reference_sha256'] = file_hash(ROOT/'docs/vpr_l2_reference/fixed_sample_ids.json')
    atomic_json(OUT/'bn_calibrated_hard.json',bn)
    if file_hash(hard) != bn['source_checkpoint_sha256']:
        raise RuntimeError('Raw hard artifact was changed')
    result = dict(status='PASS',commit=gates['commit'],config_sha256=gates['config_sha256'],
        checkpoint_sha256={'final':file_hash(final),'hard':file_hash(hard)},
        dense=dense_probe,raw_hard=pruned['report'],hard_probe=hard_probe,bn_calibrated_hard=bn,
        geometry=geometry,endpoint=endpoint_rows,gates=gates,
        completed_epochs=200,completed_steps=70400,test_used=False,recovery_executed=False,
        extra_seeds_executed=False,historical_comparison='audited V1 seed0 JSON/CSV only; no V1 checkpoint loaded')
    atomic_json(OUT/'evaluation.json',result)
    print('VPR-L2 approved non-training evaluations PASS',flush=True)


if __name__ == '__main__':main()
