"""Approved acceptance-only extension after the separately preserved math patch."""
import ast,difflib,sys,types
from .snows_patch import candidate_source
from .upstream import ROOT,source

def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unexpected native acceptance anchor: '+old[:60])
    return text.replace(old,new,1)

def sources():
    base,_=candidate_source();text=base
    text='from benchmark.snows_step_result import SNOWSSolverResult, ensure_step_finite, record_step\n'+text
    text=once(text,'    loss_history = []','    line_search_trace = []\n    loss_history = []')
    text=once(text,'            for bt in range(max_backtracking_steps):',
        '            ensure_step_finite(getattr(sub_model_sparse[mod_num], conv_layer).weight, W_original, input_weight=W_original, gradient=grad_W, validation_gradient=grad_W_val, direction=full_newton_step, loss=last_batch_loss_pruned, gradient_norm=grad_W.norm(), validation_gradient_norm=grad_W_val.norm(), direction_norm=full_newton_step.norm(), gradient_dot_direction=torch.dot(grad_W.flatten(), full_newton_step.flatten()))\n'
        '            step_accepted = False\n            for bt in range(max_backtracking_steps):')
    text=once(text,'                W_new = W_original + alpha * full_newton_step',
        '                attempted_alpha = alpha\n                W_new = W_original + alpha * full_newton_step\n'
        '                ensure_step_finite(getattr(sub_model_sparse[mod_num], conv_layer).weight, W_original, candidate=W_new)')
    condition='                if last_batch_loss_pruned_new <= last_batch_loss_pruned + c * alpha * torch.dot(grad_W.flatten(), full_newton_step.flatten()):'
    text=once(text,condition+'\n                    break',
        '                ensure_step_finite(getattr(sub_model_sparse[mod_num], conv_layer).weight, W_original, candidate_loss=last_batch_loss_pruned_new)\n'+condition+'\n                    step_accepted = True\n                    break')
    anchor='            with torch.no_grad():\n                getattr(sub_model_sparse[mod_num], conv_layer).weight.data = W_new\n\n            model_update'
    replacement='            with torch.no_grad():\n                getattr(sub_model_sparse[mod_num], conv_layer).weight.data = W_new if step_accepted else W_original\n'
    replacement+='            line_search_trace.append(record_step(step_accepted, bt + 1, attempted_alpha, last_batch_loss_pruned, last_batch_loss_pruned_new, grad_W, full_newton_step, grad_W_val))\n\n            model_update'
    text=once(text,anchor,replacement)
    original_return='return getattr(sub_model_sparse[mod_num], conv_layer).weight.data, current_loss'
    if text.count(original_return)!=2:raise ValueError('Solver return sites changed')
    text=text.replace(original_return,'return SNOWSSolverResult(getattr(sub_model_sparse[mod_num], conv_layer).weight.data, current_loss, line_search_trace)')
    caller=source('SNOWS','CNN_pruning/prune/Layer_pruner_new.py');updated=caller;lines=caller.splitlines(True)
    assignments=[n for n in ast.walk(ast.parse(caller)) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='solve_for_W_given_Z']
    if len(assignments)!=2:raise ValueError('Native solver caller sites changed')
    for node in sorted(assignments,key=lambda n:n.lineno,reverse=True):
        indent=lines[node.lineno-1][:len(lines[node.lineno-1])-len(lines[node.lineno-1].lstrip())]
        segment=lines[node.lineno-1:node.end_lineno]
        segment[0]=segment[0].replace('w_sol, _ =','solver_result =',1)
        segment.extend([indent+'w_sol, _ = solver_result\n',indent+"self.solver_diagnostics = getattr(self, 'solver_diagnostics', [])\n",
                        indent+"self.solver_diagnostics.append({'status': solver_result.status, 'line_search': solver_result.line_search})\n"])
        lines[node.lineno-1:node.end_lineno]=segment
    updated=''.join(lines)
    diff=''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='a/CNN_pruning/prune/SNOWS.py',tofile='b/CNN_pruning/prune/SNOWS.py'))
    diff+=''.join(difflib.unified_diff(caller.splitlines(True),updated.splitlines(True),fromfile='a/CNN_pruning/prune/Layer_pruner_new.py',tofile='b/CNN_pruning/prune/Layer_pruner_new.py'))
    return text,updated,diff

def load_step_candidate():
    text,caller,diff=sources()
    if (ROOT/'patches/snows_step_acceptance.patch').read_text(encoding='utf-8')!=diff:raise ValueError('Step patch mismatch')
    helpers=types.ModuleType('_snows_step_helpers')
    exec(compile(source('SNOWS','CNN_pruning/prune/helpers.py'),'controlled/helpers.py','exec'),helpers.__dict__)
    previous=sys.modules.get('helpers');sys.modules['helpers']=helpers
    module=types.ModuleType('_snows_step_candidate')
    try:exec(compile(text,'controlled/SNOWS_step_acceptance.py','exec'),module.__dict__)
    finally:
        if previous is None:sys.modules.pop('helpers',None)
        else:sys.modules['helpers']=previous
    return module
