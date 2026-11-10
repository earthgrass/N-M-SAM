"""Reproduce exact solver fragments without a model training or full solver run."""
import ast,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.snows_patch import candidate_source
from benchmark.common import atomic_json

def execute(node,namespace):
    exec(compile(ast.Module(body=[node],type_ignores=[]),'controlled_exact_SN0WS_solver_fragment','exec'),namespace)

def main():
    text,_=candidate_source();tree=ast.parse(text)
    solver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve_for_W_given_Z')
    deletion=next(n for n in ast.walk(solver) if isinstance(n,ast.Delete) and any(isinstance(t,ast.Name) and t.id=='last_batch_ydata_dense' for t in n.targets))
    evidence={'scope':'exact original solver AST statements; no solver patch or GPU execution'}
    # Actual failure branch reset followed by its unconditional final write.
    reset=next(n for n in ast.walk(solver) if isinstance(n,ast.If) and 'bt == max_backtracking_steps - 1' in ast.unparse(n.test))
    parents=[n for n in ast.walk(solver) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch_start']
    batch_loop=parents[0]
    bt_loop=next(n for n in batch_loop.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='bt')
    assert batch_loop.body.index(deletion)>batch_loop.body.index(bt_loop)
    evidence['lifetime_hypothesis_corrected']={'dense_target_deletion_is_after_backtracking':True,
        'status':'NOT_A_BLOCKER','note':'An initial out-of-context harness selected post-loop cleanup; that false hypothesis was withdrawn.'}
    subsequent=batch_loop.body[batch_loop.body.index(bt_loop)+1]
    from types import SimpleNamespace
    holder=SimpleNamespace(weight=torch.nn.Parameter(torch.tensor([1.])))
    scope={'torch':torch,'bt':49,'max_backtracking_steps':50,'W_original':torch.tensor([1.]),'W_new':torch.tensor([7.]),
           'sub_model_sparse':[SimpleNamespace(conv1=holder)],'mod_num':0,'conv_layer':'conv1'}
    execute(reset,scope);after_reset=holder.weight.detach().tolist()
    execute(subsequent,scope);after_final=holder.weight.detach().tolist()
    assert after_reset==[1.] and after_final==[7.]
    evidence.update(failure_reset_statement=ast.unparse(reset),unconditional_statement=ast.unparse(subsequent),
        after_rejection_reset=after_reset,after_unconditional_write=after_final,
        decision='BLOCKED_SOURCE: do not invent corrected Armijo batch/gradient/termination semantics or run full method')
    destination=ROOT/'artifacts/smoke_readiness_20261025/solver_blockers.json'
    if destination.exists():raise FileExistsError('Preserve previous evidence')
    atomic_json(destination,evidence);print('Exact solver fragments confirm additional blockers; full optimizer remains stopped.')

if __name__=='__main__':main()
