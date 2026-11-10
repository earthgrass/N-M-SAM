"""Execute actual patched Armijo statements with deterministic loss oracles."""
import ast
from types import SimpleNamespace
import pytest,torch
from benchmark.snows_patch import candidate_source
from benchmark.snows_step_patch import sources,load_step_candidate
from benchmark.snows_step_result import SNOWSSolverResult,SNOWSStepFailure,ensure_step_finite,record_step

def solver(text):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='solve_for_W_given_Z')

def execute_search(losses,gradient=None):
    text,_,_=sources();fn=solver(text)
    batch=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch_start')
    start=next(i for i,n in enumerate(batch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='alpha' for t in n.targets))
    end=next(i for i,n in enumerate(batch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model_update' for t in n.targets))
    original=torch.tensor([1.,2.],dtype=torch.float64);holder=SimpleNamespace(weight=torch.nn.Parameter(original.clone()))
    outputs=[];losses=iter(losses);attempted=[]
    class Sparse:
        def __getitem__(self,index):return SimpleNamespace(conv1=holder)
        def __call__(self,x):
            loss=next(losses);attempted.append(holder.weight.detach().clone())
            outputs.append(torch.tensor([loss],dtype=torch.float64).sqrt())
    sparse=Sparse();gradient=torch.ones(2,dtype=torch.float64) if gradient is None else gradient.double()
    ns={'torch':torch,'device':'cpu','sub_model_sparse':sparse,'get_submodel':lambda *args:sparse,'model_update':None,
        'num_blocks':1,'block_number':0,'mod_num':0,'conv_layer':'conv1','full_newton_step':torch.tensor([-.1,-.2],dtype=torch.float64),
        'outputs_to_reconstruct':outputs,'last_batch_xdata':None,'last_batch_ydata_dense':[torch.zeros(1,dtype=torch.float64)],
        'last_batch_loss_pruned':torch.tensor(1.,dtype=torch.float64),'min_length':1,'mse_loss':torch.nn.MSELoss(),
        'grad_W':gradient,'grad_W_val':torch.ones(2,dtype=torch.float64),'line_search_trace':[],
        'ensure_step_finite':ensure_step_finite,'record_step':record_step}
    exec(compile(ast.Module(body=batch.body[start:end],type_ignores=[]),'actual_patched_armijo','exec'),ns)
    result=SNOWSSolverResult(holder.weight.data,1.,ns['line_search_trace'])
    return holder,original,result,attempted

@pytest.mark.parametrize('rejections',[0,2,49])
def test_normal_accepted_paths_preserve_native_candidate_sequence(rejections):
    holder,original,result,attempted=execute_search([2.]*rejections+[.5])
    expected=original+(.9**rejections)*torch.tensor([-.1,-.2],dtype=torch.float64)
    torch.testing.assert_close(holder.weight,expected,rtol=1e-14,atol=1e-14)
    assert result.status=='accepted' and result.line_search[-1]['trials']==rejections+1
    for i,weight in enumerate(attempted):
        torch.testing.assert_close(weight,original+(.9**i)*torch.tensor([-.1,-.2],dtype=torch.float64),rtol=1e-14,atol=1e-14)

def test_all_rejected_weights_bitwise_original_and_subsequent_result_cannot_reapply_update():
    holder,original,result,attempted=execute_search([2.]*50)
    assert len(attempted)==50 and torch.equal(holder.weight,original)
    assert result.status=='rejected' and result.line_search[-1]['accepted_alpha'] is None
    weight,loss=result
    holder.weight.data=weight # unchanged caller behavior after result unpack
    assert torch.equal(holder.weight,original) and loss==1.
    assert len(result)==2 and result[0] is weight

@pytest.mark.parametrize('kind',['gradient','loss'])
def test_nonfinite_step_fails_with_evidence_and_restores_weight(kind):
    if kind=='gradient':
        with pytest.raises(SNOWSStepFailure) as exc:execute_search([.5],torch.tensor([float('inf'),1.]))
    else:
        with pytest.raises(SNOWSStepFailure) as exc:execute_search([float('nan')])
    assert exc.value.diagnostics['status']=='failed_nonfinite' and exc.value.diagnostics['original_weights_restored']
    parameter=torch.nn.Parameter(torch.tensor([7.]));original=torch.tensor([1.])
    with pytest.raises(SNOWSStepFailure):ensure_step_finite(parameter,original,loss=float('inf'))
    assert torch.equal(parameter,original)

def test_original_objective_condition_batches_gradient_and_math_kernels_unchanged():
    old,_=candidate_source();new,caller,_=sources();before=solver(old);after=solver(new)
    def conditions(fn):return [ast.dump(n.test,include_attributes=False) for n in ast.walk(fn)
        if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and 'last_batch_loss_pruned_new' in ast.unparse(n.test)]
    assert conditions(before)==conditions(after)
    for name in ('hessian_vector_product_chunks','conjugate_gradient_sparse'):
        find=lambda text:next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(find(old),include_attributes=False)==ast.dump(find(new),include_attributes=False)
    def selected(fn):
        keys={'last_batch_xdata','grad_W_val','grad_W'}
        return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id in keys for t in n.targets)]
    assert selected(before)==selected(after)
    assert sum(isinstance(n,ast.Return) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
               and n.value.func.id=='SNOWSSolverResult' for n in ast.walk(after))==2

def test_both_actual_call_sites_preserve_pair_unpack_and_record_status():
    _,caller,_=sources();tree=ast.parse(caller)
    for call_assignment in [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='solve_for_W_given_Z']:
        # Execute the native argument expression and the3new caller statements unchanged.
        lines=caller.splitlines();segment='\n'.join(lines[call_assignment.lineno-1:call_assignment.end_lineno+3])
        import textwrap
        for state in ('accepted','rejected'):
            returned=SNOWSSolverResult(torch.tensor([1.]),.2,[{'status':state}])
            ns={n.id:None for n in ast.walk(call_assignment.value) if isinstance(n,ast.Name)}
            attrs={n.attr:None for n in ast.walk(call_assignment.value) if isinstance(n,ast.Attribute)
                   and isinstance(n.value,ast.Name) and n.value.id=='self'}
            owner=SimpleNamespace(**attrs)
            for n in ast.walk(call_assignment.value):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id!='solve_for_W_given_Z':
                    ns[n.func.id]=lambda *args,**kwargs:0
            ns.update(self=owner,solve_for_W_given_Z=lambda *a,**kw:returned)
            exec(textwrap.dedent(segment),ns)
            assert ns['w_sol'] is returned[0] and ns['_']==.2
            assert owner.solver_diagnostics[-1]['status']==state

def test_full_controlled_module_import_keeps_old_math_and_new_result_type():
    module=load_step_candidate();assert callable(module.solve_for_W_given_Z)
    assert module.SNOWSSolverResult is SNOWSSolverResult
