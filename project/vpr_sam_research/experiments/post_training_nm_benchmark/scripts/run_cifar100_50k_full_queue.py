"""Serial three-seed full CIFAR-100 formal matrix; fail closed, no retries."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_cifar100_50k_dense_queue as base

PROTOCOL='cifar100_50k10k_final_v1';METHODS=('magnitude','wanda','sparsegpt','rose','apbr_sam')

def write(path,value):base.atomic_json(path,value)

def wait_run(pid,output,kind,seed,method=None):
    while True:
        status=base.read_status(output)
        if status and status.get('status')=='failed':raise RuntimeError(f'{kind} seed{seed} {method or ""} failed: {status}')
        if status and status.get('status')=='completed':
            if kind=='training':
                return status
            if (status.get('protocol_id')!=PROTOCOL or status.get('recovery_epochs')!=20
                or status.get('recovery_steps')!=7820 or status.get('test_accessed') is not False):
                raise RuntimeError(f'Postprocess final contract failed: seed{seed} {method}')
            return status
        if not base.process_alive(pid):raise RuntimeError(f'{kind} process {pid} exited without completion')
        time.sleep(60)

def wait_exit(pid):
    for _ in range(12):
        if not base.process_alive(pid):return
        stat=Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().split()[2]=='Z':return
        time.sleep(5)
    raise RuntimeError(f'Process {pid} did not exit after publishing completion')

def existing_training(output,config_sha,training_sha):
    """Accept only a fully verified completed stage when resuming the queue."""
    if not output.exists():return None
    return training_verify(output,config_sha,training_sha)

def launch(command,log,receipt):
    if base.gpu_pids():raise RuntimeError(f'GPU occupied before launch: {sorted(base.gpu_pids())}')
    log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('ab',buffering=0)
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);stream.close()
    write(receipt,{'pid':child.pid,'command':command,'started_at_unix':time.time()});return child.pid

def training_verify(output,config_sha,training_sha):return base.verify_complete(output,config_sha,training_sha)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--vpr-src',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--training-script',type=Path,required=True)
    p.add_argument('--postprocess-script',type=Path,required=True);p.add_argument('--evaluation-script',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seed0-dense-pid',type=int,required=True);a=p.parse_args()
    root=a.output_root.resolve();status_path=root/'full_queue_status.json';config_sha=base.sha_file(a.config);training_sha=base.sha_file(a.training_script);completed=[]
    write(status_path,{'status':'RUNNING','phase':'seed0_dense','completed':completed,'started_at_unix':time.time()})
    for seed in (0,1,2):
        seedroot=root/f'seed{seed}';dense=seedroot/'dense_sam';apbr=seedroot/'apbr_sam'
        dense_record=existing_training(dense,config_sha,training_sha)
        if dense_record is not None:
            completed.append({'seed':seed,'stage':'dense',**dense_record})
        else:
            if seed==0:
                pid=a.seed0_dense_pid
            else:
                cmd=[sys.executable,str(a.training_script),'--dataset-root',str(a.dataset_root),'--vpr-src',str(a.vpr_src),'--config',str(a.config),'--output-dir',str(dense),'--method','sam','--seed',str(seed)]
                pid=launch(cmd,seedroot/'dense_sam_console.log',seedroot/'dense_sam_launcher.json')
            write(status_path,{'status':'RUNNING','phase':f'seed{seed}_dense','pid':pid,'completed':completed})
            wait_run(pid,dense,'training',seed);wait_exit(pid);dense_record=training_verify(dense,config_sha,training_sha);completed.append({'seed':seed,'stage':'dense',**dense_record})
        apbr_record=existing_training(apbr,config_sha,training_sha)
        if apbr_record is not None:
            completed.append({'seed':seed,'stage':'apbr',**apbr_record})
        else:
            cmd=[sys.executable,str(a.training_script),'--dataset-root',str(a.dataset_root),'--vpr-src',str(a.vpr_src),'--config',str(a.config),'--output-dir',str(apbr),'--method','vpr','--seed',str(seed)]
            pid=launch(cmd,seedroot/'apbr_sam_console.log',seedroot/'apbr_sam_launcher.json');write(status_path,{'status':'RUNNING','phase':f'seed{seed}_apbr','pid':pid,'completed':completed})
            wait_run(pid,apbr,'training',seed);wait_exit(pid);apbr_record=training_verify(apbr,config_sha,training_sha);completed.append({'seed':seed,'stage':'apbr',**apbr_record})
        for method in METHODS:
            output=seedroot/'methods'/method
            if output.exists():raise FileExistsError(f'Existing method output: {output}')
            parent=apbr/'final_epoch_200.pt' if method=='apbr_sam' else dense/'final_epoch_200.pt'
            cmd=[sys.executable,str(a.postprocess_script),'--dataset-root',str(a.dataset_root),'--config',str(a.config),'--parent',str(parent),'--output-dir',str(output),'--method',method,'--seed',str(seed)]
            pid=launch(cmd,seedroot/'methods'/f'{method}_console.log',seedroot/'methods'/f'{method}_launcher.json');write(status_path,{'status':'RUNNING','phase':f'seed{seed}_{method}','pid':pid,'completed':completed})
            method_status=wait_run(pid,output,'postprocess',seed,method);wait_exit(pid);completed.append({'seed':seed,'stage':method,**method_status})
        evaluations=seedroot/'test_final';evaluations.mkdir(parents=True,exist_ok=False)
        targets=[('dense_final',dense/'final_epoch_200.pt','dense_final'),('apbr_final',apbr/'final_epoch_200.pt','dense_final')]
        for method in METHODS:
            targets += [(f'{method}_raw',seedroot/'methods'/method/'raw_hard.pt','raw_hard'),(f'{method}_recovery_final',seedroot/'methods'/method/'recovery/final_epoch_020.pt','recovery_final')]
        for name,checkpoint,role in targets:
            cmd=[sys.executable,str(a.evaluation_script),'--dataset-root',str(a.dataset_root),'--config',str(a.config),'--checkpoint',str(checkpoint),'--checkpoint-role',role,'--output',str(evaluations/f'{name}.json')]
            log=evaluations/f'{name}.log'
            with log.open('wb') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Final Test evaluation failed: seed{seed} {name}')
        completed.append({'seed':seed,'stage':'final_test','targets':len(targets)})
        write(status_path,{'status':'RUNNING','phase':f'seed{seed}_complete','completed':completed})
    write(status_path,{'status':'COMPLETE','completed':completed,'finished_at_unix':time.time(),'seeds':[0,1,2],'methods':list(METHODS)})

if __name__=='__main__':
    try:main()
    except Exception as e:
        if '--output-root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--output-root')+1]);write(root/'full_queue_failure.json',{'status':'FAILED','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'automatic_retry':False,'failed_at_unix':time.time()})
        raise
