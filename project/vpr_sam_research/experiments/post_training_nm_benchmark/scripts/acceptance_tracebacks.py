"""Missing exception evidence only; no full model import/optimization experiment."""
import ast,json,subprocess,sys,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.common import atomic_json,environment,sha_file
from benchmark.upstream import source,selected_definitions

def main():
    destination=ROOT/'artifacts/preexecution_acceptance'
    destination.mkdir(exist_ok=False) # preserve earlier attempts rather than overwrite
    lock=json.loads((ROOT/'configs/upstream_lock.json').read_text(encoding='utf-8'))
    records={'actual_diagnostic_environment':environment(),'scope':'pinned source and isolated native definitions; not full optional dependency stack'}
    try:compile(source('SNOWS','CNN_pruning/prune/SNOWS.py'),'official/SNOWS/CNN_pruning/prune/SNOWS.py','exec')
    except SyntaxError:
        records['SNOWS']={'classification':'source parsing before dependency/model loading',
            'traceback':traceback.format_exc(),'repository':lock['SNOWS']['origin'],'commit':lock['SNOWS']['commit'],
            'model_adapter_can_solve':False,'status':'BLOCKED_AT_PIN_NOT_GENERAL_ALGORITHM_CLAIM'}
    ns=selected_definitions('CAP','src/sparseml/pytorch/sparsification/pruning/pruning_handle.py',
        ['CAPHandle','CAPNMHandle'],extra={'Tensor':torch.Tensor,'Optional':__import__('typing').Optional})
    errors={}
    for name in ('CAPHandle','CAPNMHandle'):
        try:ns[name](torch.ones(2,4),n=2,m=4)
        except Exception:
            errors[name]={'traceback':traceback.format_exc(),'input':'torch.ones(2,4), n=2, m=4',
                'classification':'native constructor/interface before second-order computation'}
    # Also execute the exact AST call from the N:M scorer branch, not a retyped surrogate.
    scorer=source('CAP','src/sparseml/pytorch/sparsification/pruning/modifier_pruning_cap.py')
    calls=[n for n in ast.walk(ast.parse(scorer)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='CAPHandle' and any(k.arg=='n' for k in n.keywords)]
    assert len(calls)==1
    class Standin:pass
    standin=Standin();standin._devices=['cpu']
    scope=dict(ns,param=torch.ones(2,4),n=2,m=4,blocks_in_parallel=32,self=standin,i=0)
    try:eval(compile(ast.Expression(calls[0]),'official/CAP/modifier_pruning_cap.py','eval'),scope)
    except Exception:
        errors['exact_nm_scorer_call']={'source_line':calls[0].lineno,'expression':ast.unparse(calls[0]),
            'traceback':traceback.format_exc(),'full_sparseml_entry_point':'NOT_EXECUTED_OPTIONAL_DEPENDENCIES_NOT_VALIDATED'}
    records['CAP']={'repository':lock['CAP']['origin'],'commit':lock['CAP']['commit'],'errors':errors,
        'classification':'conditional candidate; native N:M constructors blocked at pin',
        'model_adapter_can_solve':False,'core_patch_applied':False}
    records['dependency_boundary']={'validated':['Python','PyTorch','NumPy','isolated official tensor code'],
        'not_validated':['full SparseML/CAP installation and trainer','full SNOWS import and model-specific pipeline'],
        'numerical_witness':'existing compatibility_probes.json retained; no need to rerun HVP/CG',
        'requires_decision':'official corrected revision or separately authorized documented patch; no patch applied'}
    atomic_json(destination/'tracebacks.json',records)
    print(json.dumps({'output':str(destination/'tracebacks.json'),'SNOWS':'BLOCKED','CAP':'CONDITIONAL_BLOCKED'}))

if __name__=='__main__':main()
