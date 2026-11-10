"""Explicit performance exception; original failing gate and limit stay intact."""
from __future__ import annotations
import argparse,gc,json,shutil,time,traceback
from pathlib import Path
import run_gpu_exploration as runtime
from run_gpu_exploration import torch,ROOT,write_json,status,verify_source,environment,fresh,run_epochs
from hq_apbr_v1.training import tensor_sha,file_sha,cpu_copy,capture_rng,seed_all
from hq_apbr_v1.data import load_train,make_train_dataset,calibration_loader,calibration_ids

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--prior-run',type=Path,required=True);p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Refusing existing output')
    out.mkdir(parents=True,exist_ok=True)
    try:
        approval=json.loads(args.approval.read_text());config=json.loads(args.config.read_text());source=verify_source()
        if approval['decision']!='EXPLICIT RESEARCH OVERRIDE APPROVED':raise PermissionError('Explicit exception missing')
        if file_sha(args.config)!=approval['config_sha256']:raise ValueError('Frozen training config changed')
        gate_path=args.prior_run/'GPU_GATE_REPORT.json';gate=json.loads(gate_path.read_text())
        if file_sha(gate_path)!=approval['prior_gpu_gate_report_sha256']:raise ValueError('Original failed gate changed')
        if gate['status']!='GPU_SMOKE_BLOCKED' or gate['calibration_to_train_ratio']!=approval['measured_calibration_to_train_ratio']:raise ValueError('Wrong exception evidence')
        if config['gpu_limits']['maximum_calibration_to_train_ratio']!=approval['unchanged_original_limit']:raise ValueError('Gate limit changed')
        if config['seed']!=0 or config['epochs']!=200 or config['refresh_every_epochs']!=1:raise ValueError('Unauthorized run scope')
        smoke=json.loads((args.prior_run/'smoke/history.json').read_text())
        if len(smoke)!=3 or smoke[-1]['steps']!=1173 or any(r['test_accessed'] for r in smoke):raise ValueError('Incomplete smoke')
        if not json.loads((args.prior_run/'gate/resume.json').read_text())['resume_bitwise_equal']:raise ValueError('Resume gate missing')
        seed_all(0);env=environment();old_env=json.loads((args.prior_run/'environment.json').read_text())
        for k in ('gpu','gpu_uuid','compute_capability','total_memory','torch','cuda_build','cudnn','driver','deterministic','tf32'):
            if env[k]!=old_env[k]:raise ValueError('Environment changed since gates: '+k)
        if env['disk_free_bytes']<8*1024**3:raise RuntimeError('Insufficient disk')
        write_json(out/(f'environment_resume_{time.time_ns()}.json' if args.resume else 'environment.json'),env)
        data,labels,data_id=load_train(args.archive)
        identity={'experiment_id':config['experiment_id'],'seed':0,'config_sha256':file_sha(args.config),
          'source_commit':source['commit'],'source_manifest_sha256':file_sha(ROOT/'SOURCE_IDENTITY.json'),
          'dataset':data_id,'environment':env,'performance_gate':'FAIL',
          'execution_authorization':approval['decision'],'approval_sha256':file_sha(args.approval),
          'prior_gate_sha256':file_sha(gate_path),'gate_source_identity':json.loads((args.prior_run/'identity.json').read_text())}
        if args.resume:
            old=json.loads((out/'identity.json').read_text())
            for k in ('config_sha256','source_commit','source_manifest_sha256','dataset','approval_sha256','prior_gate_sha256'):
                if old[k]!=identity[k]:raise ValueError('Resume identity changed: '+k)
            identity=old;initial_sha=json.loads((out/'INITIALIZATION_PROOF.json').read_text())['model_sha256']
        else:
            write_json(out/'identity.json',identity);shutil.copyfile(args.config,out/'FINAL_TRAINING_CONFIG.json')
            shutil.copyfile(args.approval,out/'RESEARCH_OVERRIDE_APPROVAL.json')
            shutil.copyfile(gate_path,out/'ORIGINAL_GPU_GATE_REPORT_UNCHANGED.json')
            a,step,scheduler=fresh(config);initial_sha=tensor_sha(a.state_dict())
            initial={'schema':'hq_seed0_initial_state_v1','seed':0,'steps':0,'model':cpu_copy(a.state_dict()),
              'model_sha256':initial_sha,'stepper':step.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng(),
              'identity':identity,'test_accessed':False,'smoke_state_reused':False}
            temporary=out/'initial_seed0.pt.tmp';torch.save(initial,temporary)
            loaded=torch.load(temporary,map_location='cpu',weights_only=False)
            if tensor_sha(loaded['model'])!=initial_sha:raise ValueError('Initial checkpoint save failed')
            temporary.replace(out/'initial_seed0.pt')
            del a,step,scheduler,initial,loaded;gc.collect();torch.cuda.empty_cache()
            b,step,scheduler=fresh(config)
            if tensor_sha(b.state_dict())!=initial_sha or step.steps or any(bool(v.count_nonzero()) for v in step.geometry.values()):raise ValueError('Fresh initialization replay failed')
            write_json(out/'INITIALIZATION_PROOF.json',{'status':'PASS','seed':0,'two_independent_constructions_bitwise_equal':True,
              'model_sha256':initial_sha,'checkpoint_sha256':file_sha(out/'initial_seed0.pt'),'initial_geometry_all_zero':True,
              'source_function':'run_gpu_exploration.fresh (same function used by smoke)','smoke_state_reused':False})
            del b,step,scheduler;gc.collect();torch.cuda.empty_cache()
        trainset=make_train_dataset(data,labels);calib=calibration_loader(data,labels,config['calibration_batch_size'],config['workers'])
        write_json(out/'calibration_ids.json',calibration_ids())
        status(out,'authorized_training_start',performance_gate='FAIL',execution_authorization=approval['decision'],test_accessed=False)
        history,final=run_epochs(out,config,identity,trainset,calib,count=200,resume=args.resume,phase='training',expected_initial_sha=initial_sha)
        status(out,'final_pruning',completed_epochs=200,test_accessed=False,performance_gate='FAIL')
        from hq_apbr_v1.evaluation import evaluate_final
        result=evaluate_final(final,args.archive,args.config,out/'evaluation',torch.device('cuda:0'))
        dense=result['results']['dense']['accuracy_percent'];native=result['results']['sparsegpt_native']['accuracy_percent']
        summary={'status':'SEED0_EXPLORATORY_COMPLETE','performance_gate':'FAIL','research_override':approval['decision'],
          'identity':identity,'completed_epochs':200,'steps':78200,'seed':0,'recovery_epochs':0,'test_accessed':True,
          'final_checkpoint_sha256':file_sha(final),'results':result['results'],'dense_to_native_sparse_drop_pp':dense-native,
          'train_seconds':sum(r['train_seconds'] for r in history),'calibration_seconds':sum(r['calibration_duration'] for r in history),
          'historical_results_status':'Reference comparison pending protocol audit; no multi-seed claims'}
        write_json(out/'SEED0_EVIDENCE_REPORT.json',summary)
        lines=['# HQ-APBR / HC-SAM Seed0 exploratory result','',
          'Original GPU performance gate FAIL is retained. Execution used explicit research override. No recovery.','',
          '| Endpoint | Accuracy % | CE |','|---|---:|---:|']
        lines += [f"| {name} | {value['accuracy_percent']:.2f} | {value['ce']:.6f} |" for name,value in result['results'].items()]
        lines += ['',f'Dense-to-native drop: {dense-native:.2f} percentage points.',f"Train seconds: {summary['train_seconds']:.3f}; calibration seconds: {summary['calibration_seconds']:.3f}.",
          'Historical SAM/APBR results require protocol audit before a formal ranking.']
        (out/'SEED0_EVIDENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        status(out,'complete',completed_epochs=200,steps=78200,test_accessed=True,performance_gate='FAIL',execution_authorization=approval['decision'])
        write_json(out/'SHA256_EVIDENCE.json',{str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256_EVIDENCE.json'})
    except BaseException as exc:
        write_json(out/f'failure_{time.time_ns()}.json',{'status':'FAILED','message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        status(out,'failed',error=str(exc),automatic_retry=False);raise

if __name__=='__main__':main()
