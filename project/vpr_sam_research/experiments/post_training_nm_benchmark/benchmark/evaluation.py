"""Dataset evaluation and separate native/common BN-only model copies."""
import copy
import torch
from torch import nn
from .common import finite,tensor_sha

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();before=tensor_sha(model.state_dict());correct=count=0;loss_sum=0.
    for batch in loader:
        x,y=batch[:2];output=model(x.to(device));loss=nn.functional.cross_entropy(output,y.to(device),reduction='sum')
        finite([loss,output],'Evaluation');loss_sum+=float(loss);correct+=int((output.argmax(1)==y.to(device)).sum());count+=len(y)
    if count==0 or tensor_sha(model.state_dict())!=before:raise ValueError('Empty evaluation or modified BN')
    return {'accuracy':correct/count,'loss':loss_sum/count,'count':count}

@torch.no_grad()
def recalibrate_copy(model,loader,device,mode='common',native_batches=100):
    clone=copy.deepcopy(model).eval();before=tensor_sha(dict(clone.named_parameters()))
    batchnorms=[m for m in clone.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    if mode not in ('common','obc_native'):raise ValueError('Unknown BN policy')
    for m in batchnorms:m.reset_running_stats();m.momentum=None if mode=='common' else .1;m.train()
    batches=0
    if mode=='common':
        for x,*_ in loader:clone(x.to(device));batches+=1
    else:
        while batches<native_batches:
            observed=0
            for x,*_ in loader:
                if batches==native_batches:break
                clone(x.to(device));batches+=1;observed+=1
            if observed==0:raise ValueError('Empty native BN loader')
    if batches==0 or any(int(m.num_batches_tracked)!=batches for m in batchnorms):raise ValueError('BN batch counters')
    if tensor_sha(dict(clone.named_parameters()))!=before:raise ValueError('BN-only modified weights')
    finite(clone.state_dict().values(),'BN recalibration');clone.eval()
    return clone,{'mode':mode,'batches':batches,'momentum':None if mode=='common' else .1,'weights_unchanged':True}
