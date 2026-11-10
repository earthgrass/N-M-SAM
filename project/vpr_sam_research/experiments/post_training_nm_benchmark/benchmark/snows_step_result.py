"""Compatible two-item solver result plus explicit per-step acceptance evidence."""
import torch

class SNOWSSolverResult(tuple):
    def __new__(cls,weight,loss,line_search):
        result=super().__new__(cls,(weight,loss))
        result.line_search=tuple(line_search)
        result.status=line_search[-1]['status'] if line_search else 'not_run'
        return result

class SNOWSStepFailure(FloatingPointError):
    def __init__(self,field):
        self.diagnostics={'status':'failed_nonfinite','field':field,'original_weights_restored':True}
        super().__init__('SNOWS step nonfinite: '+field)

def ensure_step_finite(parameter,original,**values):
    for name,value in values.items():
        if not bool(torch.isfinite(torch.as_tensor(value)).all()):
            with torch.no_grad():parameter.data.copy_(original)
            raise SNOWSStepFailure(name)

def record_step(accepted,trials,alpha,before,trial,gradient,direction,validation_gradient):
    def scalar(value):return float(value.detach()) if isinstance(value,torch.Tensor) else float(value)
    return {'status':'accepted' if accepted else 'rejected','trials':int(trials),
        'candidate_alpha':scalar(alpha),'accepted_alpha':scalar(alpha) if accepted else None,
        'loss_before':scalar(before),'last_candidate_loss':scalar(trial),
        'gradient_dot_direction':scalar(torch.dot(gradient.flatten(),direction.flatten())),
        'grad_W_val_norm':scalar(validation_gradient.norm()),
        'data_semantics':'native last-batch objective/current-Newton-batch gradient retained'}
