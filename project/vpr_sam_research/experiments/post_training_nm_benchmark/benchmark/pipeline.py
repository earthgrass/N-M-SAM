"""Resumable OBC layer boundaries; reference activations always use dense teacher.

Caller supplies explicitly approved train-only calibration batches. No dataset,
budget, augmentation, or GPU permission is inferred by this module.
"""
import copy
from pathlib import Path
import torch
from . import artifacts
from .common import finite,tensor_sha
from .nm import eligible_layers,validate_masks
from .obc import load_official,prune_with_hessian

@torch.no_grad()
def prune_obc(dense,calibration_factory,out,identity,parent,parallel=32,resume=False,upstream_root=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    teacher=copy.deepcopy(dense).eval();student=copy.deepcopy(dense).eval()
    teacher_sha=tensor_sha(teacher.state_dict());layers=eligible_layers(student)
    masks={};completed=[];receipts=[];boundary=out/'native_boundary.pt'
    if resume:
        saved=artifacts.load(boundary,identity)
        if saved['parent']!=parent or saved['dense_model_sha256']!=teacher_sha:raise ValueError('OBC parent changed')
        if saved['stage'] not in ('native_partial','native_complete'):raise ValueError('OBC resume stage')
        completed=saved['completed_layers'];masks=saved['masks'];receipts=saved['layer_receipts']
        if completed!=list(layers)[:len(completed)]:raise ValueError('OBC layer order changed')
        student.load_state_dict(saved['model']);validate_masks(student,masks,complete=len(completed)==len(layers))
    elif boundary.exists():raise FileExistsError('Use explicit OBC boundary resume')
    for name,layer in layers.items():
        if name in completed:continue
        reference=teacher.get_submodule(name);cls=load_official(reference.weight.device,upstream_root=upstream_root)
        handle=cls(reference,rel_damp=0.)
        def observe(module,inputs,output):handle.add_batch(inputs[0].detach(),output.detach())
        hook=reference.register_forward_hook(observe);samples=0
        try:
            for images,*_ in calibration_factory():
                teacher(images.to(reference.weight.device));samples+=len(images)
        finally:hook.remove()
        finite([handle.H],'OBC native H')
        if samples==0 or handle.nsamples!=samples:raise ValueError('OBC calibration coverage')
        result,mask,receipt=prune_with_hessian(layer,handle.H,parallel=parallel,upstream_root=upstream_root)
        layer.weight.copy_(result);masks[name+'.weight']=mask.cpu();completed.append(name)
        receipts.append(dict(receipt,layer=name,samples_including_rounds=samples))
        del handle
        if tensor_sha(teacher.state_dict())!=teacher_sha:raise ValueError('Dense reference/BN modified')
        validate_masks(student,masks,complete=len(completed)==len(layers))
        value={'stage':'native_complete' if len(completed)==len(layers) else 'native_partial',
               'identity':identity,'parent':parent,'dense_model_sha256':teacher_sha,'model':student.state_dict(),
               'masks':masks,'completed_layers':completed,'layer_receipts':receipts,
               'output_semantics':'native post-pruning; surviving-weight compensation; not Raw Hard'}
        artifacts.save(boundary,value)
    if not completed:raise ValueError('No eligible OBC layers')
    if not (out/'native_complete.pt').exists():artifacts.save(out/'native_complete.pt',artifacts.load(boundary,identity))
    else:
        final=artifacts.load(out/'native_complete.pt',identity)
        if final['model_sha256']!=tensor_sha(student.state_dict()) or final['parent']!=parent:raise ValueError('Existing native output mismatch')
    return student,masks,receipts
