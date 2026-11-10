"""Read-only gates. Pending decisions never become implicit defaults."""
import argparse,json,shutil,subprocess,os
from pathlib import Path
import torch,yaml
from .calibration import approved_indices
from .common import atomic_json,environment,object_sha,sha_file
from .nm import CONVENTION

ROOT=Path(__file__).resolve().parents[1]

def source_identity():
    records={}
    for folder in ('benchmark','configs','scripts','patches'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml','.json','.patch'):
                records[path.relative_to(ROOT).as_posix()]=sha_file(path)
    return {'files':records,'sha256':object_sha(records)}

def gates(config,method='OBC',gpu=False):
    failures=[]
    if method!='OBC':failures.append(method+' is blocked/literature-only; no equivalent verified runner')
    p=config['pruning']
    if (p['n'],p['m'],p['layout'],p['scope'],p['tail'],p['bn_affine'])!=(2,4,CONVENTION,'exclude_first_last','dense','excluded'):
        failures.append('Candidate N:M grouping/scope/tail semantics are not accepted')
    if config.get('nm_acceptance',{}).get(method)!='CPU_ADAPTER_PASS_CUDA_PENDING':
        failures.append('Candidate N:M adapter acceptance missing/blocked')
    try:approved_indices(config)
    except (ValueError,KeyError,FileNotFoundError) as exc:failures.append(str(exc))
    if config['execution']['approved'] is not True:failures.append('GPU execution is not yet authorized')
    if gpu:
        receipt=config['execution'].get('cuda_smoke_receipt')
        if config['execution'].get('smoke_lineage'):
            try:
                from .smoke_lineage import verify
                verify(config,source_identity())
            except (FileNotFoundError,ValueError,KeyError) as exc:
                failures.append('Reviewed CUDA smoke lineage: '+str(exc))
        elif receipt is None:failures.append('Reviewed CUDA smoke receipt is missing')
        else:
            try:
                check=json.loads(Path(receipt).read_text(encoding='utf-8'))
                if check.get('status')!='PASS' or check.get('source_sha256')!=source_identity()['sha256'] or check.get('config_object_sha256')!=object_sha(config):
                    failures.append('CUDA smoke receipt/source/config mismatch')
            except (FileNotFoundError,ValueError):failures.append('CUDA smoke receipt is unreadable')
        if not torch.cuda.is_available():failures.append('No CUDA available; CPU substitution prohibited')
        else:
            if config['device']!='cuda:0':failures.append('Unapproved CUDA logical device')
            props=torch.cuda.get_device_properties(0)
            actual=str(props.uuid)
            if actual.removeprefix('GPU-')!=config['gpu_uuid'].removeprefix('GPU-'):failures.append('GPU UUID mismatch')
            if props.name!=config.get('gpu_name','NVIDIA GeForce RTX 5090'):failures.append('Reviewed GPU model mismatch')
            if config.get('minimum_free_cuda_bytes') is not None and torch.cuda.mem_get_info(0)[0]<config['minimum_free_cuda_bytes']:
                failures.append('Insufficient CUDA memory reserve')
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if any(p.strip()!=str(os.getpid()) for p in processes.splitlines()):failures.append('Existing compute process; no concurrent GPU task allowed')
        if shutil.disk_usage(ROOT).free<config['minimum_free_disk_bytes']:failures.append('Insufficient disk space')
        if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
            failures.append('Source tree is not clean/frozen')
    return failures

def require_config_hash(path,expected):
    if expected is None or sha_file(path)!=expected:raise ValueError('Reviewed configuration SHA256 missing/mismatch')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/post_training_nm_benchmark.yaml')
    p.add_argument('--gpu',action='store_true');p.add_argument('--output',default='artifacts/preflight.json');args=p.parse_args()
    c=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));errors=gates(c,gpu=args.gpu)
    report={'status':'BLOCKED' if errors else 'PASS','failures':errors,'environment':environment(),
        'source':source_identity(),'config_sha256':sha_file(args.config),'gpu_checked':args.gpu,
        'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
    atomic_json(args.output,report);print(json.dumps({'status':report['status'],'failures':errors}))
    raise SystemExit(2 if errors else 0)

if __name__=='__main__':main()
