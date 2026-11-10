"""Real ResNet-18/CIFAR-100 CPU parity gate for the APBR gradient handoff."""
from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, random, sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(PROJECT/'src')]
from baseline_adapters.cifar100_model import build_cifar100_model
from joint_optimization.joint_stepper import JointStepper
from vpr_sam.optim.geometry import Geometry
from vpr_sam.optim.step import Stepper


def digest(values):
    h=hashlib.sha256()
    for value in values:
        if value is None:h.update(b'<NONE>');continue
        item=value.detach().cpu().contiguous();h.update(str(item.dtype).encode());h.update(str(tuple(item.shape)).encode());h.update(item.numpy().tobytes())
    return h.hexdigest()


def compare(left,right):
    if len(left)!=len(right):raise RuntimeError('Trace length mismatch')
    maximum=0.;num=den=0.;none_equal=True
    for a,b in zip(left,right):
        if a is None or b is None:
            none_equal &= a is None and b is None;continue
        delta=(a.double()-b.double());maximum=max(maximum,float(delta.abs().max()));num+=float(delta.square().sum());den+=float(a.double().square().sum())
    return {'none_pattern_equal':none_equal,'max_abs':maximum,'relative_l2':math.sqrt(num/max(den,1e-300)),'bitwise_equal':digest(left)==digest(right)}


def instrument(model,geometry,optimizer):
    trace={'forward_parameter_hashes':[]}
    hook=model.register_forward_pre_hook(lambda _m,_x:trace['forward_parameter_hashes'].append(digest([p for p in geometry.params])))
    original_perturb=geometry.perturbations
    def perturb(*args,**kwargs):
        trace['first_grads']=[None if p.grad is None else p.grad.detach().clone() for p in geometry.params]
        result=original_perturb(*args,**kwargs);trace['epsilon']=[e.detach().clone() for e in result[0]];trace['perturbation_stats']=dict(result[1]);return result
    geometry.perturbations=perturb
    original_prepare=geometry.prepare
    def prepare(*args,**kwargs):
        result=original_prepare(*args,**kwargs);candidate,_zero=geometry.candidate;trace['energy_candidate']=([s.detach().clone() for s in geometry.logs] if candidate is None else [s.detach().clone() for s in candidate]);return result
    geometry.prepare=prepare
    original_step=optimizer.step
    def step(*args,**kwargs):
        trace['restored_pre_optimizer']=[p.detach().clone() for p in geometry.params]
        trace['final_optimizer_grads']=[None if p.grad is None else p.grad.detach().clone() for p in geometry.params]
        return original_step(*args,**kwargs)
    optimizer.step=step
    return trace,hook


def rng_state():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
def restore_rng(state):random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial-asset',type=Path,required=True);p.add_argument('--train-pickle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if torch.cuda.is_available():raise RuntimeError('Parity gate is deliberately CPU-only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(20261101);np.random.seed(20261101);torch.manual_seed(20261101)
    initial=torch.load(a.initial_asset,map_location='cpu',weights_only=False)
    with a.train_pickle.open('rb') as f:raw=pickle.load(f,encoding='bytes')
    data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64);ids=np.asarray([39328,14404,39830,46380,29132,30158,8556,45709])
    x=torch.from_numpy(data[ids].copy()).reshape(-1,3,32,32).float().div_(255);mean=torch.tensor((.507075159237132,.486548873314951,.440917843367034)).view(1,3,1,1);std=torch.tensor((.267334285879241,.256438462917088,.276150471325684)).view(1,3,1,1);x=(x-mean)/std;y=torch.from_numpy(labels[ids].copy())
    config={'seed':0,'optimizer':{'method':'vpr','lr':.1,'momentum':.9,'weight_decay':5e-4,'nesterov':False,'rho':.05},'geometry':initial['apbr_geometry']['config']}
    original_model=build_cifar100_model();original_model.load_state_dict(initial['model']);original=Stepper(original_model,config);original.geometry.load_state_dict(initial['apbr_geometry'])
    joint_model=build_cifar100_model();joint_model.load_state_dict(initial['model']);joint_geometry=Geometry(joint_model.named_parameters(),config['geometry'],'vpr',0);joint_geometry.load_state_dict(initial['apbr_geometry']);joint_optimizer=torch.optim.SGD(joint_geometry.params,lr=.1,momentum=.9,weight_decay=5e-4)
    joint=JointStepper(joint_model,joint_geometry,joint_optimizer,eligible_parameter_names=(),lambda_p=0.,eta_s=0.,rho=.05)
    initial_parameters=[p.detach().clone() for p in original.geometry.params]
    original_trace,original_hook=instrument(original_model,original.geometry,original.base);joint_trace,joint_hook=instrument(joint_model,joint_geometry,joint_optimizer)
    state=rng_state();restore_rng(state);original_result=original.step(x,y);restore_rng(state);joint_result=joint.step(x,y,pruning_active=True);original_hook.remove();joint_hook.remove()
    checks={
        'first_pass_gradients':compare(original_trace['first_grads'],joint_trace['first_grads']),
        'global_perturbation':compare(original_trace['epsilon'],joint_trace['epsilon']),
        'energy_feedback_candidate':compare(original_trace['energy_candidate'],joint_trace['energy_candidate']),
        'restored_pre_optimizer':compare(original_trace['restored_pre_optimizer'],joint_trace['restored_pre_optimizer']),
        'final_optimizer_gradients':compare(original_trace['final_optimizer_grads'],joint_trace['final_optimizer_grads']),
        'final_parameters':compare([p.detach() for p in original.geometry.params],[p.detach() for p in joint_geometry.params]),
        'final_geometry':compare(original.geometry.logs,joint_geometry.logs),
    }
    restored_to_base=compare(initial_parameters,original_trace['restored_pre_optimizer'])
    forward_equal=original_trace['forward_parameter_hashes']==joint_trace['forward_parameter_hashes'] and len(original_trace['forward_parameter_hashes'])==2
    scalar={'eps_norm_abs':abs(original_result['eps_norm']-joint_result['eps_norm']),'ellipsoid_norm_abs':abs(original_result['ellipsoid_norm']-joint_result['ellipsoid_norm']),'center_loss_abs':abs(original_result['center_loss']-joint_result['center_loss']),'adversarial_loss_abs':abs(original_result['adversarial_loss']-joint_result['adversarial_loss'])}
    tolerance=1e-7;passed=all(v['none_pattern_equal'] and v['max_abs']<=tolerance and v['relative_l2']<=1e-7 for v in checks.values()) and restored_to_base['bitwise_equal'] and forward_equal and max(scalar.values())<=tolerance and joint_result['eps_norm']>0 and abs(joint_result['ellipsoid_norm']-.05)<=1e-6
    result={'status':'APBR_PARITY_PASS' if passed else 'APBR_PARITY_FAILED','real_model':True,'real_cifar100_train_batch_ids':ids.tolist(),'checks':checks,'restored_to_base':restored_to_base,'adversarial_forward_parameter_states_equal':forward_equal,'scalar_differences':scalar,'original_result':original_result,'joint_result':joint_result,'final_test_accessed':False}
    temp=a.output.with_suffix(a.output.suffix+'.tmp');temp.parent.mkdir(parents=True,exist_ok=True);temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,a.output);print(json.dumps({'status':result['status'],'eps_norm':joint_result['eps_norm'],'max_check_error':max(v['max_abs'] for v in checks.values())}));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
