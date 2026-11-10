"""Authorized GPU gate -> three temporary Epochs -> Seed0 200 -> frozen Test.

Any exception stops the chain. --resume is explicit, never an automatic retry.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, gc, hashlib, json, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from hq_apbr_v1 import math_core
from hq_apbr_v1.calibration import eligible_layers, refresh
from hq_apbr_v1.data import load_train, make_train_dataset, training_loader, calibration_loader, calibration_ids
from hq_apbr_v1.vendor.model import build_model
from hq_apbr_v1.training import HQStepper, seed_all, train_epoch, build_checkpoint, atomic_checkpoint, load_checkpoint, tensor_sha, file_sha, cpu_copy, capture_rng, restore_rng

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def status(out,stage,**kw):
    row={'stage':stage,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'automatic_retry':False,**kw}
    write_json(out/'status.json',row);print(json.dumps(row),flush=True)

def verify_source():
    identity=json.loads((ROOT/'SOURCE_IDENTITY.json').read_text())
    for name,digest in identity['files'].items():
        if file_sha(ROOT/name)!=digest:raise RuntimeError('Source identity changed: '+name)
    return identity

def environment():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    props=torch.cuda.get_device_properties(0)
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda_build':torch.version.cuda,
      'cudnn':torch.backends.cudnn.version(),'gpu':props.name,'total_memory':props.total_memory,
      'compute_capability':[props.major,props.minor],'free_memory':torch.cuda.mem_get_info()[0],
      'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
      'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
      'gpu_uuid':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
      'compute_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True),
      'disk_free_bytes':shutil.disk_usage(ROOT).free,'deterministic':torch.are_deterministic_algorithms_enabled(),
      'tf32':torch.backends.cuda.matmul.allow_tf32}

def resource_check(config):
    actual={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    for key,value in actual.items():
        if value>config['gpu_limits'][key]:raise RuntimeError(f'Resource limit {key}: {value}')
    return actual

def math_gate():
    torch.manual_seed(713)
    x=torch.randn(32,64,dtype=torch.float64);h=x@x.T/64+.1*torch.eye(32,dtype=torch.float64)
    ref,_,_=math_core.inverse_diagonal_cholesky(h)
    rows=[]
    for dtype,tolerance in ((torch.float32,5e-5),(torch.float64,1e-10)):
        q,hd,_=math_core.inverse_diagonal_cholesky(h.to('cuda',dtype))
        error=float((q.cpu().double()-ref).norm()/ref.norm())
        if error>tolerance:raise ArithmeticError(f'CPU/GPU inverse diagonal mismatch {dtype}: {error}')
        cpu_target,_,_=math_core.target_from_inverse_diagonals([math_core.GeometrySpec('conv.weight',(2,32,1,1),1,ref.reshape(1,-1))])
        gpu_target,_,_=math_core.target_from_inverse_diagonals([math_core.GeometrySpec('conv.weight',(2,32,1,1),1,q.reshape(1,-1))])
        gradients={'conv.weight':torch.linspace(-1,1,64).reshape(2,32,1,1),'bias':torch.tensor([.2,-.1])}
        params={k:torch.empty_like(v) for k,v in gradients.items()}
        cpu_eps,_,_=math_core.hq_global_l2_perturbation(params,gradients,cpu_target,.05)
        gpu_eps,_,_=math_core.hq_global_l2_perturbation({k:v.cuda() for k,v in params.items()},{k:v.cuda() for k,v in gradients.items()},gpu_target,.05)
        perturb_error=max(float((gpu_eps[k].cpu()-cpu_eps[k]).abs().max()) for k in params)
        target_error=float((gpu_target['conv.weight'].cpu()-cpu_target['conv.weight']).abs().max())
        if perturb_error>1e-6 or target_error>5e-5:raise ArithmeticError('CPU/GPU geometry or perturbation mismatch')
        rows.append({'dtype':str(dtype),'relative_l2':error,'tolerance':tolerance,'cholesky':'PASS','geometry_max_abs':target_error,'epsilon_max_abs':perturb_error})
    return {'status':'CPU_GPU_MATH_PASS','cases':rows,'test_accessed':False}

def fresh(config):
    seed_all(config['seed']);model=build_model().cuda()
    geometry={name+'.weight':torch.zeros_like(layer.weight,dtype=torch.float64) for name,layer in eligible_layers(model).items()}
    stepper=HQStepper(model,geometry,rho=config['rho'],lr=config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(stepper.optimizer,T_max=config['epochs'],eta_min=0.)
    return model,stepper,scheduler

def nested_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    # numpy RNG arrays
    if hasattr(a,'shape') and hasattr(a,'dtype'):return bool((a==b).all())
    return a==b

def real_step_resume_gate(out,config,calib,identity):
    model,stepper,scheduler=fresh(config)
    torch.cuda.reset_peak_memory_stats()
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    x,y,_=next(iter(calib));x=x.cuda();y=y.cuda()
    info=stepper.step(x,y,audit=True)
    g=info['audit']['first_gradients'];eps=info['audit']['epsilon']
    norm=sum(v.double().square().sum() for v in g.values() if v is not None).sqrt()
    delta=max(float((eps[k].double()-(torch.zeros_like(p) if g[k] is None else config['rho']*g[k].double()/norm)).abs().max()) for k,p in model.named_parameters())
    if delta>1e-8:raise ArithmeticError('Epoch0 is not SAM direction')
    if not all(torch.equal(original[k],v) for k,v in info['audit']['restored_parameters'].items()):raise ArithmeticError('Weight restore failed')
    if not all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in info['audit']['bn_after_center'].items()):raise ArithmeticError('Second-pass BN pollution')
    if abs(info['physical_addition_norm']/config['rho']-1)>1e-3:raise ArithmeticError('Physical radius after FP32 addition')
    if tensor_sha(dict(model.named_parameters()))==tensor_sha(original):raise ArithmeticError('No model update')
    del original,g,eps
    summary={k:v for k,v in info.items() if k!='audit'};del info
    geometry,q,calreport=refresh(model,calib,stepper.geometry,{**config,'reference_checks':True},'cuda:0')
    if calreport['log_geometry_drift']['rms']<=0:raise ArithmeticError('Zero geometry update')
    stepper.set_geometry(geometry)
    if stepper.geometry_sha256!=calreport['current_geometry_hash']:raise ArithmeticError('Geometry report/hash mismatch')
    payload=build_checkpoint(model,stepper,scheduler,epoch=0,history=[],identity=identity,calibration_ids=calibration_ids(),q_diags=q,calibration_report=calreport)
    path=out/'gate/temporary_resume.pt';atomic_checkpoint(path,payload)
    info_a=stepper.step(x,y);state_a=cpu_copy({'model':model.state_dict(),'stepper':stepper.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng()})
    load_checkpoint(path,model,stepper,scheduler,identity=identity)
    info_b=stepper.step(x,y);state_b=cpu_copy({'model':model.state_dict(),'stepper':stepper.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng()})
    if not nested_equal(state_a,state_b):raise ArithmeticError('Real GPU resume not bitwise equal')
    result={'status':'REAL_STEP_RESUME_PASS','sam_direction_max_abs':delta,'first_step':summary,
      'resume_bitwise_equal':True,'resume_step_info_equal':info_a==info_b,'checkpoint_sha256':file_sha(path),
      'test_accessed':False,**resource_check(config)}
    write_json(out/'gate/resume.json',result);write_json(out/'gate/calibration.json',calreport)
    del model,stepper,scheduler,geometry,q,payload,state_a,state_b
    gc.collect();torch.cuda.empty_cache()
    return result,calreport

def run_epochs(out,config,identity,trainset,calib,*,count,resume=None,phase,expected_initial_sha=None):
    model,stepper,scheduler=fresh(config);history=[];q=None;start_epoch=0
    if expected_initial_sha is not None and tensor_sha(model.state_dict())!=expected_initial_sha:
        raise ArithmeticError('Fresh Seed0 initialization hash mismatch')
    if resume:
        payload=load_checkpoint(resume,model,stepper,scheduler,identity=identity)
        history=payload['history'];q={k:v.cuda() for k,v in payload['q_diags'].items()};start_epoch=payload['next_epoch']
        if start_epoch>=count:raise ValueError('Training already complete; refusing final overwrite')
        if stepper.steps!=start_epoch*391:raise ArithmeticError('Resume step contract')
    phase_dir=out/phase;phase_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch,count):
        verify_source()
        if epoch==0 and any(bool(s.count_nonzero()) for s in stepper.geometry.values()):raise ArithmeticError('Epoch0 geometry must be zero')
        if epoch>0 and not any(bool(s.count_nonzero()) for s in stepper.geometry.values()):raise ArithmeticError('Historical geometry missing')
        def progress(batches,samples,info):
            if batches==1 or batches%50==0:
                status(out,phase,epoch_index=epoch,completed_epochs=epoch,batches=batches,steps=stepper.steps,test_accessed=False,eps_norm=info['eps_norm'])
        row=train_epoch(stepper,training_loader(trainset,config['batch_size'],config['seed'],epoch,config['workers']), 'cuda:0',epoch,progress_callback=progress)
        if row['zero_steps']:raise ArithmeticError('Unexpected zero perturbation steps')
        resource_check(config)
        geometry,q,calreport=refresh(model,calib,stepper.geometry,config,'cuda:0',previous_q=q)
        calreport['calibration_epoch']=epoch
        resource_check(config)
        stepper.set_geometry(geometry);scheduler.step()
        if stepper.geometry_sha256!=calreport['current_geometry_hash']:raise ArithmeticError('Geometry report/hash mismatch')
        row.update(calibration_epoch=epoch,calibration_duration=calreport['calibration_duration'],
          q_diagonal_statistics={k:{'min':float(v.min()),'max':float(v.max()),'mean':float(v.double().mean())} for k,v in q.items()},
          log_geometry_drift=calreport['log_geometry_drift'],spearman_stability=calreport['spearman_stability'],
          next_geometry_hash=stepper.geometry_sha256,skipped_calibration_epochs=[],**resource_check(config))
        history.append(row)
        write_json(phase_dir/f'calibration_epoch_{epoch:03d}.json',calreport)
        write_json(phase_dir/'history.json',history)
        payload=build_checkpoint(model,stepper,scheduler,epoch=epoch,history=history,identity=identity,calibration_ids=calibration_ids(),q_diags=q,calibration_report=calreport)
        sha=atomic_checkpoint(phase_dir/'last.pt',payload)
        if epoch in (0,9,49,99,149) and phase=='training':atomic_checkpoint(phase_dir/f'epoch_{epoch:03d}.pt',payload)
        status(out,phase,epoch_index=epoch,completed_epochs=epoch+1,steps=stepper.steps,checkpoint_sha256=sha,test_accessed=False,train_seconds=row['train_seconds'],calibration_seconds=row['calibration_duration'])
        del payload
    final=phase_dir/('final_epoch_199.pt' if phase=='training' else 'smoke_final.pt')
    # Preserve the complete final state under its fixed final filename.
    shutil.copyfile(phase_dir/'last.pt',final)
    if phase=='training' and stepper.steps!=78200:raise ArithmeticError('Expected exactly78200 optimizer steps')
    del model,stepper,scheduler,q
    gc.collect();torch.cuda.empty_cache()
    return history,final

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);args=p.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Refusing existing run directory')
    out.mkdir(parents=True,exist_ok=True)
    try:
        config=json.loads(args.config.read_text());source=verify_source();seed_all(config['seed'])
        env=environment();write_json(out/(f'environment_resume_{time.time_ns()}.json' if args.resume else 'environment.json'),env)
        if env['disk_free_bytes']<8*1024**3:raise RuntimeError('Less than8GiB disk free')
        data,labels,data_id=load_train(args.archive);write_json(out/'data_identity.json',data_id)
        identity={'experiment_id':config['experiment_id'],'seed':0,'config_sha256':file_sha(args.config),'source_commit':source['commit'],'source_manifest_sha256':file_sha(ROOT/'SOURCE_IDENTITY.json'),'dataset':data_id,'environment':env}
        # Environment free-memory values naturally change on resume; validate against saved run identity.
        if args.resume:
            old=json.loads((out/'identity.json').read_text())
            for k in ('config_sha256','source_commit','source_manifest_sha256','dataset'):
                if old[k]!=identity[k]:raise RuntimeError('Resume identity mismatch '+k)
            for k in ('gpu','gpu_uuid','compute_capability','total_memory','torch','cuda_build','cudnn','driver','deterministic','tf32'):
                if old['environment'][k]!=env[k]:raise RuntimeError('Resume environment mismatch '+k)
            identity=old
            if json.loads((out/'GPU_GATE_REPORT.json').read_text())['status']!='GPU_SMOKE_PASS':raise RuntimeError('Missing prior gates')
        else:write_json(out/'identity.json',identity);shutil.copyfile(args.config,out/'FINAL_TRAINING_CONFIG.json')
        trainset=make_train_dataset(data,labels);calib=calibration_loader(data,labels,config['calibration_batch_size'],config['workers'])
        write_json(out/'calibration_ids.json',calibration_ids())
        if not args.resume:
            status(out,'gpu_gate',test_accessed=False)
            write_json(out/'gate/math.json',math_gate())
            resume_report,calreport=real_step_resume_gate(out,config,calib,identity)
            smoke,_=run_epochs(out,config,identity,trainset,calib,count=config['smoke_epochs'],phase='smoke')
            ratio=sum(r['calibration_duration'] for r in smoke)/sum(r['train_seconds'] for r in smoke)
            gate={'status':'GPU_SMOKE_PASS' if ratio<=config['gpu_limits']['maximum_calibration_to_train_ratio'] else 'GPU_SMOKE_BLOCKED',
              'gpu':env['gpu'],'epochs':len(smoke),'calibration_to_train_ratio':ratio,
              'estimated_200_epoch_compute_seconds':200*sum(r['train_seconds']+r['calibration_duration'] for r in smoke)/len(smoke),
              'every_epoch_refresh_feasible':ratio<=config['gpu_limits']['maximum_calibration_to_train_ratio'],
              'resource_limits':config['gpu_limits'],'test_accessed':False,'resume_bitwise_equal':resume_report['resume_bitwise_equal']}
            write_json(out/'GPU_GATE_REPORT.json',gate)
            (out/'GPU_GATE_REPORT.md').write_text('# HQ-APBR V1 GPU gate\n\n'+json.dumps(gate,indent=2)+'\n\nReal data: 1024 fixed Train images, all19 Conv2d, sequential layer matrices. No Test access. Temporary3Epoch smoke is separate from formal Seed0.\n',encoding='utf8')
            if gate['status']!='GPU_SMOKE_PASS':raise RuntimeError('Calibration overhead exceeds frozen limit; no adaptive fallback')
        history,final=run_epochs(out,config,identity,trainset,calib,count=200,resume=args.resume,phase='training')
        status(out,'final_pruning',checkpoint_sha256=file_sha(final),completed_epochs=200,test_accessed=False)
        from hq_apbr_v1.evaluation import evaluate_final
        evaluate_final(final,args.archive,args.config,out/'evaluation',torch.device('cuda:0'))
        status(out,'complete',completed_epochs=200,steps=78200,test_accessed=True,final_checkpoint_sha256=file_sha(final))
        sums={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256_EVIDENCE.json'}
        write_json(out/'SHA256_EVIDENCE.json',sums)
    except BaseException as exc:
        record={'status':'FAILED','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False}
        if torch.cuda.is_available():record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(out/f'failure_{time.time_ns()}.json',record);status(out,'failed',error=str(exc),automatic_retry=False)
        raise

if __name__=='__main__':main()
