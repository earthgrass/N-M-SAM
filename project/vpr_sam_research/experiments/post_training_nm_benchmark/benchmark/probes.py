"""Stage2 convention proof and independently checked upstream numerical blockers."""
import json
from pathlib import Path
import torch
from .common import atomic_json
from .upstream import ROOT,selected_definitions,source

def grouping_probe():
    # Actual prune_NM -> Layer_pruner_new imports this definition from SNOWS.py.
    # helpers.py contains a similar duplicate, but is not this executed import.
    snows=selected_definitions('SNOWS','CNN_pruning/prune/SNOWS.py',['generate_M_sized_groups'],diagnostic_allow_invalid_module=True)
    native=snows['generate_M_sized_groups'](4,3,3,4)
    perm=torch.arange(36).reshape(4,3,3).permute(1,2,0).flatten().reshape(-1,4).tolist()
    canonical=torch.arange(36).reshape(-1,4).tolist()
    assert native==perm and canonical!=native
    return {'status':'DIFFERENT_GROUPS_REQUIRES_ADAPTER','weight_shape':[1,4,3,3],
            'canonical_first_group':canonical[0],'obc_and_snows_first_group':native[0],
            'canonical':'OIHW [out, in*kH*kW], consecutive groups per output row',
            'upstream':'input-channel strided groups at each kernel spatial position',
            'safe_adaptation_requirement':'Change grouping interface only; preserve compensation/optimization; never reinterpret a completed Mask'}

def snows_probe():
    ns=selected_definitions('SNOWS','CNN_pruning/prune/SNOWS.py',
          ['hessian_vector_product_chunks','conjugate_gradient_sparse'],diagnostic_allow_invalid_module=True)
    sizes=[8,200008];results=[]
    for size in sizes:
        weight=torch.nn.Parameter(torch.linspace(.1,1,size).reshape(1,-1))
        mask=torch.tensor([True,True,False,False]).repeat(size//4).reshape_as(weight)
        loss=.5*weight.square().sum()
        gradient=torch.autograd.grad(loss,weight,create_graph=True)[0]*mask
        vector=torch.ones(int(mask.sum()))
        actual=ns['hessian_vector_product_chunks'](gradient,weight,vector,mask)
        expected=vector  # Exact Hessian of .5||w||² is identity.
        results.append({'weight_coordinates':size,'kept_coordinates':int(mask.sum()),
            'max_chunk_size':100000,'max_abs_hvp_error':float((actual-expected).abs().max()),
            'observed_hvp_values':actual.unique().tolist(),'expected_hvp_value':1.0})
    zero,iterations=ns['conjugate_gradient_sparse'](lambda v:v,torch.zeros(4),max_iter=2)
    blocked=results[-1]['max_abs_hvp_error']>0
    return {'status':'BLOCKED_NATIVE_HVP' if blocked else 'PASS','pinned_default_hvp':results,
        'whole_module_syntax_error':ns['_whole_module_syntax_error'],
        'execution_scope':'Untouched function bodies extracted for diagnostics only; full module is not importable',
        'zero_rhs_cg_finite':bool(torch.isfinite(zero).all()),'zero_rhs_result':str(zero.tolist()),
        'core_code_modified':False,'full_cifar_resnet18_integration':'pending; native named-stage and shortcut adapters are not verified',
        'decision':'Do not run SNOWS pilot/queue until an official fix or explicitly approved controlled correction is audited'}

def cap_probe():
    # The official scorer's N:M branch calls CAPHandle(..., n=..., m=...).
    ns=selected_definitions('CAP','src/sparseml/pytorch/sparsification/pruning/pruning_handle.py',
        ['CAPHandle','CAPNMHandle'],extra={'Tensor':torch.Tensor,'Optional':__import__('typing').Optional})
    errors={}
    for name in ['CAPHandle','CAPNMHandle']:
        try:ns[name](torch.ones(2,4),n=2,m=4)
        except Exception as exc:errors[name]={'type':type(exc).__name__,'message':str(exc)}
    return {'status':'BLOCKED_OFFICIAL_NM_ENTRY' if errors else 'PASS','errors':errors,
        'source_scorer':'modifier_pruning_cap.py N:M branch constructs CAPHandle with n/m',
        'core_code_modified':False,'decision':'Conditional only; do not silently repair the official algorithm'}

def main():
    results={'stage2_grouping':grouping_probe(),'SNOWS':snows_probe(),'CAP':cap_probe()}
    atomic_json(ROOT/'artifacts/compatibility_probes.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
