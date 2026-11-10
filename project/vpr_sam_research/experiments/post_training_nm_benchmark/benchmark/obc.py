"""Official TrueOBS kernel behind a proven canonical coordinate interface.

No greedy score, compensation, inverse update or numerical fallback is rewritten.
The observer records pruning decisions; it never changes their values.
"""
import inspect,sys,types
from pathlib import Path
import torch
from torch import nn
from .common import finite
from .nm import validate_layer_mask
from .upstream import source

class _CudaInterface:
    def __init__(self,device):self.device=device
    def synchronize(self,*args,**kwargs):
        if self.device.type=='cuda':return torch.cuda.synchronize(self.device)
        # CPU-only adaptation of a timing fence, not a numerical operation.
    def __getattr__(self,name):return getattr(torch.cuda,name)

class _TorchInterface:
    def __init__(self,device,selection):self.cuda=_CudaInterface(device);self.selection=selection
    def __getattr__(self,name):return getattr(torch,name)
    def argmin(self,values,*args,**kwargs):
        result=torch.argmin(values,*args,**kwargs)
        frame=inspect.currentframe().f_back
        if frame.f_code.co_name!='nmprune':raise RuntimeError('Unexpected official selection call')
        info=frame.f_locals
        self.selection[info['i1']+info['rangecount'],result]=False
        return result

def load_official(device='cpu',selection=None,upstream_root=None):
    quant=types.ModuleType('_pinned_obc_quant')
    exec(compile(source('OBC','quant.py',upstream_root),'official/OBC/quant.py','exec'),quant.__dict__)
    saved=sys.modules.get('quant');sys.modules['quant']=quant
    module=types.ModuleType('_pinned_obc_trueobs')
    try:exec(compile(source('OBC','trueobs.py',upstream_root),'official/OBC/trueobs.py','exec'),module.__dict__)
    finally:
        if saved is None:sys.modules.pop('quant',None)
        else:sys.modules['quant']=saved
    device=torch.device(device)
    if selection is not None:module.torch=_TorchInterface(device,selection)
    elif device.type=='cpu':module.torch=_TorchInterface(device,None)
    return module.TrueOBS

@torch.no_grad()
def accumulate(layer,batches,relative_damping=0.,upstream_root=None):
    if isinstance(layer,nn.Conv2d) and layer.groups!=1:raise ValueError('Grouped convolution is not adapted')
    cls=load_official(layer.weight.device,upstream_root=upstream_root)
    handle=cls(layer,rel_damp=relative_damping)
    for images,outputs in batches:handle.add_batch(images,outputs)
    finite([handle.H],'Official Hessian accumulation')
    if handle.nsamples==0:raise ValueError('No training calibration images')
    return handle.H,handle.nsamples

@torch.no_grad()
def prune_with_hessian(layer,hessian,n=2,m=4,parallel=32,relative_damping=0.,upstream_root=None):
    if not isinstance(layer,(nn.Conv2d,nn.Linear)):raise TypeError('Expected Conv2d/Linear')
    if isinstance(layer,nn.Conv2d) and layer.groups!=1:raise ValueError('Unsupported grouped convolution')
    weight=layer.weight.detach().flatten(1)
    if weight.shape[1]%m:raise ValueError('Official kernel has no partial bucket; tails require a separately verified adapter')
    if not 0<n<m or parallel<1:raise ValueError('Invalid N:M/parallel')
    finite([weight,hessian],'OBC input');selection=torch.ones_like(weight,dtype=torch.bool)
    cls=load_official(weight.device,selection,upstream_root)
    # This is exactly the Linear interface already supported by official code.
    matrix=nn.Linear(weight.shape[1],weight.shape[0],bias=False,device=weight.device,dtype=weight.dtype)
    matrix.weight.copy_(weight)
    handle=cls(matrix,rel_damp=relative_damping);handle.H=hessian.detach().clone()
    handle.nmprune(n=m-n,m=m,parallel=parallel) # official n is number removed, not retained
    result=matrix.weight.detach().clone().reshape_as(layer.weight)
    mask=selection.reshape_as(layer.weight)
    finite([result],'Official compensated output');validate_layer_mask(result,mask,n,m)
    return result,mask,{'algorithm':'official_TrueOBS_nmprune','layout_adapter':'original Conv Unfold Hessian -> Linear identity permutation',
        'compensation':'native surviving-weight update preserved','cpu_timing_fence_only':weight.device.type=='cpu',
        'retained_zero_values':int(torch.count_nonzero(result[mask]==0))}
