"""Detached serial queue, waiting for immutable seed0's full completion."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HOME=ROOT.parent
PYTHON='/root/autodl-tmp/vpr_l2_seed0_20261024/workspace/vpr_sam_research/.venv/bin/python'
SEED0=Path('/root/autodl-tmp/vpr_v2_unified50k_20261024/project')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def source_files(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()
        and p.suffix in ('.py','.yaml','.yml','.md','.json','.sh','.txt','.toml')
        and p.relative_to(root).parts[0]!='data'
        and 'artifacts' not in p.relative_to(root).parts
        and not any(k in p.relative_to(root).parts for k in ('.git','__pycache__','.pytest_cache'))
        and p.name!='SOURCE_FREEZE.json'}

def verify_bundle():
    freeze=json.loads((ROOT/'SOURCE_FREEZE.json').read_text())
    if source_files(ROOT)!=freeze['source_sha256']:raise ValueError('Frozen overnight bundle changed')
    return freeze

def deploy(freeze):
    paths=[]
    for seed in (1,2):
        for method in ('sam','vpr'):
            project=HOME/'runs'/f'{method}_seed{seed}'/'project'
            if project.exists():raise FileExistsError('Preserve existing worker: '+str(project))
            template=ROOT/'workers'/method;shutil.copytree(template,project)
            config=ROOT/'configs'/f'{method}_seed{seed}.yaml';shutil.copyfile(config,project/'config.yaml')
            files=source_files(project);mapping={}
            for name,digest in files.items():
                git_path=f'configs/{method}_seed{seed}.yaml' if name=='config.yaml' else f'workers/{method}/{name}'
                if freeze['source_sha256'][git_path]!=digest:raise ValueError('Deployment bytes differ from frozen Git source')
                mapping[name]=git_path
            save(project/'SOURCE_FREEZE.json',{'commit':freeze['commit'],'source_sha256':files,
                 'bundle_commit':freeze['commit'],'deployment_git_paths':mapping,'branch':freeze['branch']})
            paths.append(project)
    return paths

def main():
    import fcntl
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True);status=artifacts/'queue_status.json'
    with open('/tmp/vpr_v2_seeds12_overnight.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            freeze=verify_bundle();free=shutil.disk_usage(HOME).free
            required=24*1024**3  # Both seeds' full retained epoch checkpoints, logs, temporary files and reserves.
            save(artifacts/'initial_disk_budget.json',{'free_bytes':free,'required_bytes':required,
                 'seed0_partition_separate':True,'historical_cleanup_needed':False})
            if free<required:raise RuntimeError('Insufficient full two-seed artifact budget')
            paths=deploy(freeze)
            save(status,{'status':'waiting_seed0_full_pipeline','pid':os.getpid(),'commit':freeze['commit'],
                 'new_dense_trajectories':4,'seeds':[1,2],'gpu_tasks_launched':0})
            while True:
                verify_bundle();s=json.loads((SEED0/'artifacts/controller_status.json').read_text())
                if s['status']=='failed':raise RuntimeError('Seed0 failed; queue will not start')
                if s['status']=='completed':
                    audit=json.loads((SEED0/'artifacts/comparison/FINAL_AUDIT.json').read_text())
                    if audit['status']!='PASS':raise ValueError('Seed0 comparison audit failed')
                    break
                time.sleep(30)
            for index,project in enumerate(paths):
                verify_bundle()
                save(status,{'status':'running_worker','worker':project.parent.name,'pid':os.getpid(),
                     'worker_index':index+1,'workers_total':4,'commit':freeze['commit'],'unix_time':time.time()})
                (project/'artifacts').mkdir(exist_ok=True)
                with (project/'artifacts/controller.log').open('x') as log:
                    child=subprocess.Popen([PYTHON,'-u','launch.py'],cwd=project,stdout=log,stderr=subprocess.STDOUT)
                    save(artifacts/(project.parent.name+'_process.json'),{'pid':child.pid,'project':str(project)})
                    if child.wait()!=0:raise RuntimeError('Worker failed: '+str(project))
                done=json.loads((project/'artifacts/controller_status.json').read_text())
                if done['status']!='completed':raise ValueError('Worker completion missing')
            subprocess.run([PYTHON,'-u','aggregate.py'],cwd=ROOT,check=True)
            save(status,{'status':'completed','seeds':[0,1,2],'commit':freeze['commit'],
                 'additional_gpu_tasks':False,'unix_time':time.time()})
        except BaseException as exc:
            save(artifacts/'queue_failure.json',{'error':str(exc),'type':type(exc).__name__,
                 'traceback':traceback.format_exc(),'automatic_restart':False,'unix_time':time.time()})
            save(status,{'status':'failed','error':str(exc),'additional_gpu_tasks':False})
            raise

if __name__=='__main__':main()
