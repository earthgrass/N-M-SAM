"""New controlled-patch diagnostics only; old full suites are not repeated."""
import numpy as np
import pytest,torch
from benchmark.snows_patch import load_candidate
from benchmark.snows_repair_kernels import conjugate_gradient_sparse as cg,hessian_vector_product_chunks as hvp
from benchmark.nm import validate_layer_mask,eligible_layers
from benchmark.model import build_model
from benchmark.upstream import source

def test_full_patched_module_import_and_unused_placeholder_fails_explicitly():
    module=load_candidate()
    assert callable(module.solve_for_W_given_Z)
    with pytest.raises(NotImplementedError):module.prune_and_update(None,None,None,None)
    # Upstream remains invalid and untouched; this is a labelled controlled candidate.
    with pytest.raises(IndentationError):compile(source('SNOWS','CNN_pruning/prune/SNOWS.py'),'untouched','exec')

def analytic_hvp(chunk):
    w=torch.nn.Parameter(torch.linspace(.2,1.2,12,dtype=torch.float64).reshape(2,6))
    matrix=torch.arange(1,145,dtype=torch.float64).reshape(12,12)/100
    h=matrix.T@matrix+torch.eye(12,dtype=torch.float64)
    mask=torch.tensor([[True,False,True,True,False,False],[False,True,True,False,True,False]])
    loss=.5*w.flatten()@h@w.flatten()
    grad=torch.autograd.grad(loss,w,create_graph=True)[0]*mask
    vector=torch.arange(1,7,dtype=torch.float64)
    actual=hvp(grad,w,vector,mask,max_chunk_size=chunk)
    expected=h[mask.flatten()][:,mask.flatten()]@vector
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    return actual

@pytest.mark.parametrize('chunk',[1,2,3,100000])
def test_hvp_dense_offdiagonal_analytic_and_chunk_vs_unchunked(chunk):
    torch.testing.assert_close(analytic_hvp(chunk),analytic_hvp(100000),rtol=1e-12,atol=1e-12)

def test_hvp_permutation_gradient_and_step_coordinates():
    torch.manual_seed(6);x=torch.randn(8,8,dtype=torch.float64);h=x.T@x+torch.eye(8,dtype=torch.float64)
    p=torch.tensor([0,2,4,6,1,3,5,7]);inverse=p.argsort();w=torch.randn(8,dtype=torch.float64)
    wp=torch.nn.Parameter(w[p].clone());hp=h[p][:,p]
    grad=torch.autograd.grad(.5*wp@hp@wp,wp,create_graph=True)[0]
    torch.testing.assert_close(grad[inverse],h@w)
    mask=torch.ones(8,dtype=torch.bool);v=torch.randn(8,dtype=torch.float64)
    result=hvp(grad,wp,v[p],mask,3)
    torch.testing.assert_close(result[inverse],h@v,rtol=1e-12,atol=1e-12)
    direction,_=cg(lambda z:hp@z,-grad.detach(),tol=1e-12,max_iter=20,lambda_reg=1e-4)
    reference=torch.linalg.solve(h+1e-4*torch.eye(8,dtype=torch.float64),-(h@w))
    torch.testing.assert_close(direction[inverse],reference,rtol=1e-9,atol=1e-10)

@pytest.mark.parametrize('scale',[0.,1e-12,1.])
def test_cg_zero_nearzero_and_regularized_spd(scale):
    h=torch.tensor([[3.,1.],[1.,2.]],dtype=torch.float64);b=torch.tensor([1.,-2.],dtype=torch.float64)*scale
    result,iterations=cg(lambda v:h@v,b,tol=1e-10,max_iter=10)
    if scale<1e-10:assert iterations==0 and torch.count_nonzero(result)==0
    else:torch.testing.assert_close(result,torch.linalg.solve(h+1e-4*torch.eye(2,dtype=torch.float64),b),rtol=1e-12,atol=1e-12)

def test_cg_unsafe_inputs_fail_without_choosing_new_damping_or_step():
    with pytest.raises(FloatingPointError):cg(lambda v:v,torch.tensor([float('nan')]))
    with pytest.raises(FloatingPointError):cg(lambda v:v*float('inf'),torch.ones(2))
    with pytest.raises(FloatingPointError,match='curvature'):cg(lambda v:-v,torch.ones(2))
    with pytest.raises(FloatingPointError,match='curvature'):cg(lambda v:v*0,torch.ones(2),lambda_reg=0)

def test_native_mask_routine_accepts_canonical_groups_and_preserves_transpose_mapping():
    module=load_candidate();weight=torch.arange(1,73,dtype=torch.float64).reshape(2,4,3,3)
    matrix=weight.flatten(1).T.clone();groups=torch.arange(36).reshape(-1,4).tolist()
    result=module.MP_MN(matrix,groups,2)
    restored=torch.from_numpy(result).T.reshape_as(weight);mask=restored!=0
    validate_layer_mask(restored,mask)
    assert torch.equal(restored,weight*mask)
    native_groups=module.generate_M_sized_groups(4,3,3,4)
    assert native_groups[0]==[0,9,18,27]
    # This local routine proof is not a proof of the blocked full solver/zero-value Mask bridge.

def test_real_cifar_model_bridge_requires_functional_relu_and_projection_scope():
    model=build_model().eval();graph=torch.fx.symbolic_trace(model,concrete_args={'y':None})
    assert any(n.op=='call_function' and n.target is torch.nn.functional.relu for n in graph.graph.nodes)
    names=eligible_layers(model)
    assert len(names)==19 and len([n for n in names if '.shortcut.' in n])==3
    assert hasattr(model,'layer4') and not hasattr(model,'name')
