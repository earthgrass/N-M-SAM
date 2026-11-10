"""Phase-4 one-shot GPU smoke and 3x performance gate.

This is an isolated smoke driver.  It reads CIFAR-100 train only, never builds
a Test loader, never publishes a formal checkpoint, and performs no retry.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, pickle, platform, statistics
from pathlib import Path
import subprocess, sys, threading, time, traceback

import numpy as np
import torch
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_cifar100_50k_formal_training as base
from baseline_adapters.apbr_gpt_joint import CandidateFactors, dynamic_joint_obs_matrix
from baseline_adapters.cifar100_model import build_cifar100_model
from benchmark.nm import eligible_layers
from joint_optimization.hessian_cache import build_training_factors, capture_layer_inputs
from joint_optimization.joint_stepper import JointStepper
from joint_optimization.orchestration import audit_nm_masks
from vpr_sam.optim.geometry import Geometry
from vpr_sam.optim.step import Stepper


class GPUSampler:
    def __init__(self): self.rows=[];self.stop=threading.Event();self.thread=None
    def __enter__(self):
        def run():
            while not self.stop.is_set():
                try:
                    text=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
                    ts,util,mem,power=[x.strip() for x in text.split(',')]
                    self.rows.append({'timestamp':ts,'utilization_percent':float(util),'memory_used_mib':float(mem),'power_w':float(power)})
                except Exception as e:self.rows.append({'sampler_error':repr(e)})
                self.stop.wait(0.25)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join()


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)


def tensor_norm_delta(before,after):
    return math.sqrt(sum(float((after[k].double()-before[k].double()).square().sum()) for k in before if torch.is_floating_point(before[k])))


def state_cpu(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def timing_summary(values):
    ordered=sorted(values);n=len(ordered)
    return {'batches':n,'mean_seconds':statistics.fmean(values),'median_seconds':statistics.median(values),
            'p95_seconds':ordered[min(n-1,math.ceil(.95*n)-1)],'total_seconds':sum(values)}


def gpu_peak():
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def normalized_batch(data,labels,ids,mean,std,device):
    x=torch.from_numpy(data[np.asarray(ids)].copy()).reshape(-1,3,32,32).float().div_(255)
    return ((x-mean)/std).to(device),torch.from_numpy(labels[np.asarray(ids)].copy()).to(device)


def make_geometry(model,config): return Geometry(model.named_parameters(),config,'vpr',0)


def one_epoch(method,initial,data,labels,transform,geometry_config,device,factors,references,lambda_p,eta_s):
    base.seed_all(0);dataset=base.OfficialTrain(data,labels,transform)
    model=build_cifar100_model().to(device);model.load_state_dict(initial['model'])
    before=state_cpu(model);before_geometry=None
    if method=='joint':
        geometry=make_geometry(model,geometry_config);geometry.load_state_dict(initial['apbr_geometry']);before_geometry=[s.clone() for s in geometry.logs]
        optimizer=torch.optim.SGD(geometry.params,lr=.1,momentum=.9,weight_decay=5e-4)
        names=tuple(name+'.weight' for name in eligible_layers(model))
        stepper=JointStepper(model,geometry,optimizer,eligible_parameter_names=names,lambda_p=lambda_p,eta_s=eta_s)
        stepper.refresh({name+'.weight':factor for name,factor in factors.items()},
                        {name+'.weight':references[name] for name in factors})
    else:
        config={'seed':0,'optimizer':{'method':'sam' if method=='dense_sam' else 'vpr','lr':.1,'momentum':.9,'weight_decay':5e-4,'nesterov':False,'rho':.05},'geometry':geometry_config}
        stepper=Stepper(model,config);geometry=stepper.geometry
        if method=='apbr_sam':before_geometry=[s.clone() for s in geometry.logs]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[];count=batches=0
    with GPUSampler() as sampler:
        for inputs,targets,_ids in base.training_loader(dataset,128,0,0):
            inputs=inputs.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            torch.cuda.synchronize();started=time.perf_counter()
            if method=='joint':result=stepper.step(inputs,targets,pruning_active=True)
            else:result=stepper.step(inputs,targets)
            torch.cuda.synchronize();times.append(time.perf_counter()-started);count+=result['count'];batches+=1
    if count!=50000 or batches!=391:raise RuntimeError('Single-epoch coverage failure')
    after=state_cpu(model);geometry_delta=0.0
    if before_geometry is not None:geometry_delta=math.sqrt(sum(float((a.double()-b.double()).square().sum()) for a,b in zip(geometry.logs,before_geometry)))
    valid=[r for r in sampler.rows if 'utilization_percent' in r]
    result={'method':method,**timing_summary(times),**gpu_peak(),'parameter_update_l2':tensor_norm_delta(before,after),
            'geometry_update_l2':geometry_delta,'train_samples':count,'test_accessed':False,
            'gpu_utilization_mean_percent':statistics.fmean(r['utilization_percent'] for r in valid) if valid else None,
            'gpu_utilization_max_percent':max((r['utilization_percent'] for r in valid),default=None)}
    del model,stepper;torch.cuda.empty_cache();return result,sampler.rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--train-pickle',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    overall_sampler=None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one CUDA GPU required')
        device=torch.device('cuda:0');base.seed_all(0)
        with a.train_pickle.open('rb') as f:raw=pickle.load(f,encoding='bytes')
        data=np.asarray(raw[b'data'],dtype=np.uint8);labels=np.asarray(raw[b'fine_labels'],dtype=np.int64)
        if data.shape!=(50000,3072) or labels.shape!=(50000,):raise RuntimeError('Train identity failure')
        initial=torch.load(a.bundle/'model_assets/initial_common_seed0.pt',map_location='cpu',weights_only=False)
        calibration=json.loads((a.bundle/'calibration/calibration_identity.json').read_text())
        lambda_p=json.loads((a.bundle/'calibration/lambda_calibration.json').read_text())['lambda_p']
        eta_s=json.loads((a.bundle/'calibration/eta_s_calibration.json').read_text())['eta_s']
        import yaml
        protocol=yaml.safe_load((a.bundle/'config/cifar100_50k10k_final_v1.yaml').read_text())
        geometry_config=protocol['apbr_sam']['geometry'];ids=calibration['batch_ids']
        mean=torch.tensor(protocol['dataset']['normalize_mean']).view(1,3,1,1);std=torch.tensor(protocol['dataset']['normalize_std']).view(1,3,1,1)
        fixed_x,fixed_y=normalized_batch(data,labels,ids,mean,std,device)
        environment={'gpu':torch.cuda.get_device_name(0),'gpu_properties':str(torch.cuda.get_device_properties(0)),
            'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
            'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'tf32':torch.backends.cuda.matmul.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision(),'deterministic':torch.are_deterministic_algorithms_enabled()}
        atomic_json(out/'environment.json',environment)
        overall_sampler=GPUSampler();overall_sampler.__enter__()

        # A/B: real one-batch Joint path using one representative layer cache.
        one=build_cifar100_model().to(device);one.load_state_dict(initial['model']);one_layers=eligible_layers(one)
        captured=capture_layer_inputs(one,{'layer1.0.conv1':one_layers['layer1.0.conv1']},fixed_x[:16],storage_device=device)
        small=build_training_factors(one_layers['layer1.0.conv1'],captured['layer1.0.conv1'],sample_seed=20261101,max_patches=256)
        geometry=make_geometry(one,geometry_config);geometry.load_state_dict(initial['apbr_geometry']);opt=torch.optim.SGD(geometry.params,lr=.1,momentum=.9,weight_decay=5e-4)
        js=JointStepper(one,geometry,opt,eligible_parameter_names=('layer1.0.conv1.weight',),lambda_p=lambda_p,eta_s=eta_s)
        js.refresh({'layer1.0.conv1.weight':small.factors},{'layer1.0.conv1.weight':calibration['references']['layer1.0.conv1']})
        w0={k:v.detach().clone() for k,v in one.named_parameters()};s0=[s.clone() for s in geometry.logs]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();batch_result=js.step(fixed_x[:16],fixed_y[:16],pruning_active=True);torch.cuda.synchronize()
        single={'seconds':time.perf_counter()-t,**gpu_peak(),'result':batch_result,
            'parameter_update_l2':math.sqrt(sum(float((p.detach()-w0[n]).double().square().sum()) for n,p in one.named_parameters())),
            'geometry_update_l2':math.sqrt(sum(float((x-y).double().square().sum()) for x,y in zip(geometry.logs,s0))),
            'lambda_p':lambda_p,'eta_s':eta_s,'test_accessed':False}
        atomic_json(out/'single_batch_joint.json',single);del one,js,opt,geometry,small;torch.cuda.empty_cache()

        # C: full real 19-layer Hessian refresh.
        hmodel=build_cifar100_model().to(device);hmodel.load_state_dict(initial['model']);layers=eligible_layers(hmodel)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();refresh_started=time.perf_counter()
        captured=capture_layer_inputs(hmodel,layers,fixed_x,storage_device=device);factors={};hrows=[]
        for index,(name,layer) in enumerate(layers.items()):
            item=build_training_factors(layer,captured.pop(name),sample_seed=20261101+index,max_patches=8192,percdamp=.01,compute_dtype=torch.float32)
            factors[name]=item.factors;hrows.append({'layer':name,'k':layer.weight[0].numel(),'sampled_patches':item.sampled_patches,
                'gram_seconds':item.gram_seconds,'cholesky_factor_seconds':item.factor_seconds,'condition_max':item.factors.condition_max,
                'damp':item.factors.damp,'cholesky_pass':True,'finite':bool(torch.isfinite(item.factors.inverse_factor).all())})
            print('HESSIAN_DONE',name,flush=True)
        torch.cuda.synchronize();refresh={'seconds':time.perf_counter()-refresh_started,**gpu_peak(),'layers':19,'all_cholesky_pass':all(r['cholesky_pass'] for r in hrows)}
        atomic_json(out/'hessian_refresh.json',refresh)
        with (out/'hessian_layers.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=hrows[0]);w.writeheader();w.writerows(hrows)

        # D/E: eight real APBR probes, full 19-layer dynamic Groupwise OBS.
        pmodel=build_cifar100_model().to(device);pmodel.load_state_dict(initial['model']);pgeometry=make_geometry(pmodel,geometry_config);pgeometry.load_state_dict(initial['apbr_geometry'])
        names=[n for n,_ in pgeometry.mapping];probes={n:[] for n in names};pmodel.eval()
        for q in range(8):
            pmodel.zero_grad(set_to_none=True);lo=q*16;hi=(q+1)*16;loss=torch.nn.functional.cross_entropy(pmodel(fixed_x[lo:hi]),fixed_y[lo:hi]);loss.backward();eps,_=pgeometry.perturbations(.05)
            for n,e in zip(names,eps):probes[n].append(e.detach().clone())
        pmodel.zero_grad(set_to_none=True);pmasks={};prows=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();obs_started=time.perf_counter()
        for name,layer in eligible_layers(pmodel).items():
            probe=torch.stack(probes[name+'.weight']).reshape(8,layer.weight.shape[0],-1);weight=layer.weight.detach().reshape(layer.weight.shape[0],-1)
            torch.cuda.synchronize();started=time.perf_counter();result=dynamic_joint_obs_matrix(weight,factors[name].damped_hessian,probe,blocksize=128,percdamp=0.,compute_dtype=torch.float32);torch.cuda.synchronize()
            layer.weight.data.copy_(result.weight.reshape_as(layer.weight));pmasks[name]=result.keep_mask.reshape_as(layer.weight)
            prows.append({'layer':name,'total_seconds':time.perf_counter()-started,'decomposition_seconds':result.decomposition_seconds,
                'candidate_compensation_seconds':result.selection_compensation_seconds,'deleted':int((~result.keep_mask).sum()),
                'gpt_cost':float(result.selected_gpt_cost.sum()),'apbr_cost':float(result.selected_apbr_cost.sum()),'finite':bool(torch.isfinite(result.weight).all())})
            print('OBS_DONE',name,prows[-1]['total_seconds'],flush=True)
        torch.cuda.synchronize();mask_audit=audit_nm_masks(pmodel,eligible_layers(pmodel),pmasks)
        obs={'seconds':time.perf_counter()-obs_started,**gpu_peak(),'mask_audit':mask_audit,'all_finite':all(r['finite'] for r in prows),'test_accessed':False}
        atomic_json(out/'groupwise_obs.json',obs)
        with (out/'groupwise_obs_layers.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=prows[0]);w.writeheader();w.writerows(prows)
        del pmodel,pgeometry,probes,pmasks;torch.cuda.empty_cache()

        # F: full official-train single epoch for all three branches.
        transform=transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(.5),transforms.ToTensor(),transforms.Normalize(protocol['dataset']['normalize_mean'],protocol['dataset']['normalize_std'])])
        epoch_rows=[];sampler_rows={}
        for method in ('dense_sam','apbr_sam','joint'):
            row,samples=one_epoch(method,initial,data,labels,transform,geometry_config,device,factors,calibration['references'],lambda_p,eta_s)
            epoch_rows.append(row);sampler_rows[method]=samples;print('EPOCH_DONE',method,row['total_seconds'],flush=True)
        ratio=next(r['total_seconds'] for r in epoch_rows if r['method']=='joint')/next(r['total_seconds'] for r in epoch_rows if r['method']=='apbr_sam')
        status='GPU_SMOKE_PASS' if ratio<=3 else 'GPU_SMOKE_BLOCKED'
        performance={'status':status,'joint_over_apbr_ratio':ratio,'threshold':3.0,'rows':epoch_rows,'test_accessed':False}
        atomic_json(out/'performance_gate.json',performance);atomic_json(out/'gpu_utilization_samples.json',sampler_rows)
        with (out/'single_epoch_performance.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=epoch_rows[0]);w.writeheader();w.writerows(epoch_rows)
        overall_sampler.__exit__(None,None,None);atomic_json(out/'overall_gpu_utilization_samples.json',overall_sampler.rows);overall_sampler=None
        atomic_json(out/'status.json',{'status':status,'formal_training_started':False,'final_test_accessed':False,'automatic_retry':False})
        return 0 if status=='GPU_SMOKE_PASS' else 3
    except BaseException as e:
        if overall_sampler is not None:
            overall_sampler.__exit__(None,None,None);atomic_json(out/'overall_gpu_utilization_samples.json',overall_sampler.rows)
        atomic_json(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'automatic_retry':False,'final_test_accessed':False})
        atomic_json(out/'status.json',{'status':'GPU_SMOKE_BLOCKED','reason':'exception','formal_training_started':False,'final_test_accessed':False});raise

if __name__=='__main__':raise SystemExit(main())
