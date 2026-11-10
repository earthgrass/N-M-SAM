"""Real train-batch CPU gate for repaired APBR + pruning-scale update."""
from __future__ import annotations
import argparse,json,os,pickle,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1];sys.path[:0]=[str(ROOT),str(PROJECT/'src')]
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.hessian_cache import build_training_factors,capture_layer_inputs
from joint_optimization.joint_stepper import JointStepper
from vpr_sam.optim.geometry import Geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('--initial-asset',type=Path,required=True);p.add_argument('--train-pickle',type=Path,required=True);p.add_argument('--calibration-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if torch.cuda.is_available():raise RuntimeError('CPU gate forbids CUDA')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261101)
 initial=torch.load(a.initial_asset,map_location='cpu',weights_only=False);lam=json.loads((a.calibration_dir/'lambda_calibration.json').read_text())['lambda_p'];eta=json.loads((a.calibration_dir/'eta_s_calibration.json').read_text())['eta_s'];identity=json.loads((a.calibration_dir/'calibration_identity.json').read_text())
 with a.train_pickle.open('rb') as f:raw=pickle.load(f,encoding='bytes')
 data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64);ids=np.asarray(identity['batch_ids'][:8]);x=torch.from_numpy(data[ids].copy()).reshape(-1,3,32,32).float().div_(255);mean=torch.tensor((.507075159237132,.486548873314951,.440917843367034)).view(1,3,1,1);std=torch.tensor((.267334285879241,.256438462917088,.276150471325684)).view(1,3,1,1);x=(x-mean)/std;y=torch.from_numpy(labels[ids].copy())
 model=build_cifar100_model();model.load_state_dict(initial['model']);layers=eligible_layers(model);name='layer1.0.conv1';captured=capture_layer_inputs(model,{name:layers[name]},x,storage_device='cpu');factor=build_training_factors(layers[name],captured[name],sample_seed=20261101,max_patches=256)
 geometry=Geometry(model.named_parameters(),initial['apbr_geometry']['config'],'vpr',0);geometry.load_state_dict(initial['apbr_geometry']);optimizer=torch.optim.SGD(geometry.params,lr=.1,momentum=.9,weight_decay=5e-4);stepper=JointStepper(model,geometry,optimizer,eligible_parameter_names=(name+'.weight',),lambda_p=lam,eta_s=eta);stepper.refresh({name+'.weight':factor.factors},{name+'.weight':identity['references'][name]})
 result=stepper.step(x,y,pruning_active=True);volume=geometry.diagnostics()['log_volume'];grads_cleared=all(p.grad is None for p in geometry.params)
 checks={'eps_nonzero':result['eps_norm']>0,'ellipsoid_rho':abs(result['ellipsoid_norm']-.05)<=1e-6,'two_forward_losses_differ':result['center_loss']!=result['adversarial_loss'],'energy_feedback_executed':result['energy_update_norm']>0,'pruning_scale_update_executed':result['pruning_scale_update_norm']>0,'combined_update_finite':result['combined_scale_update_norm']>0 and np.isfinite(result['combined_scale_update_norm']),'restoration_exact':result['restoration_max_abs']==0,'volume_constraint':abs(volume)<=1e-10,'optimizer_grads_cleared':grads_cleared,'geometry_committed_once':geometry.updates==1,'zero_skip_absent':geometry.zero_skips==0}
 checks={key:bool(value) for key,value in checks.items()};passed=all(checks.values());payload={'status':'JOINT_CPU_GATE_PASS' if passed else 'JOINT_CPU_GATE_FAILED','real_model':True,'real_cifar100_train_batch_ids':ids.tolist(),'lambda_p':lam,'eta_s':eta,'result':result,'geometry_log_volume':volume,'checks':checks,'final_test_accessed':False};a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');os.replace(tmp,a.output);print(json.dumps({'status':payload['status'],'eps_norm':result['eps_norm'],'ellipsoid_norm':result['ellipsoid_norm'],'energy_update_norm':result['energy_update_norm'],'pruning_scale_update_norm':result['pruning_scale_update_norm']}));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
