"""Source-bound schema-2 checkpoints with verified atomic publication."""
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda:file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def tensor_sha(state):
    h = hashlib.sha256()
    for name,tensor in state.items():
        t=tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(list(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def atomic_text(path, text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(path,value):
    atomic_text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def seed_all(seed, threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG unavailable')
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_payload(model,optimizer,scheduler,epoch,history,identity,best):
    return {'schema':2,'identity':identity,'model':model.state_dict(),'model_sha256':tensor_sha(model.state_dict()),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,
            'steps':history[-1]['steps'] if history else 0,'history':history,'best':best,'rng':rng_state(),
            'parameter_location':'unperturbed_after_sgd','checkpoint_status':'healthy'}


def validate_checkpoint(value,identity=None):
    if value['schema']!=2 or value['checkpoint_status']!='healthy' or value['parameter_location']!='unperturbed_after_sgd':
        raise ValueError('Checkpoint schema/health/location')
    if identity is not None and value['identity']!=identity:
        raise ValueError('Checkpoint source/config/data/environment identity mismatch')
    if tensor_sha(value['model'])!=value['model_sha256']:
        raise ValueError('Checkpoint model digest')
    if not all(torch.isfinite(t).all() for t in value['model'].values()):
        raise FloatingPointError('Checkpoint tensor')


def atomic_checkpoint(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    loaded=torch.load(tmp,map_location='cpu',weights_only=False)
    validate_checkpoint(loaded,value['identity'])
    tmp.replace(path)
    # Reopen the published path; a temporary-file-only check is insufficient.
    published=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(published,value['identity'])
    if published['epoch']!=value['epoch'] or published['steps']!=value['steps']:
        raise ValueError('Published checkpoint metadata')
    return sha_file(path)


def load_checkpoint(path,model,optimizer,scheduler,identity):
    value=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(value,identity)
    model.load_state_dict(value['model']); optimizer.load_state_dict(value['optimizer'])
    scheduler.load_state_dict(value['scheduler']); restore_rng(value['rng'])
    return value


def source_files(root):
    return {str(p.relative_to(root)).replace('\\','/'):sha_file(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()
            and p.suffix in ('.py','.yaml','.md','.json')
            and not any(part in ('artifacts','data','.git','__pycache__','.pytest_cache') for part in p.relative_to(root).parts)
            and p.name!='SOURCE_FREEZE.json'}


def source_freeze(root):
    frozen=json.loads((Path(root)/'SOURCE_FREEZE.json').read_text())
    if frozen['source_sha256']!=source_files(root):
        raise ValueError('Frozen source files changed')
    return frozen


def gpu_check(config, allow_own=False):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True).strip().splitlines()
    if len(rows)!=1:
        raise RuntimeError('Single-GPU mapping must be re-audited')
    uuid,name,free=[part.strip() for part in rows[0].split(',')]
    if uuid!=config['gpu_uuid'] or '5090' not in name or config['device']!='cuda:0':
        raise RuntimeError('Unapproved GPU')
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    pids={int(p) for p in procs if p.strip().isdigit()}
    if pids-({os.getpid()} if allow_own else set()):
        raise RuntimeError(f'Other GPU processes present: {sorted(pids)}')
    return {'uuid':uuid,'name':name,'free':free,'processes':sorted(pids)}


def disk_check(root,config):
    free=shutil.disk_usage(root).free
    if free<config['minimum_free_disk_bytes']:
        raise RuntimeError(f'Insufficient free disk bytes: {free}')
    return free


def environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'platform':platform.platform(),'torchvision':__import__('torchvision').__version__,
            'device_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip() if torch.cuda.is_available() else None,
            'cudnn':torch.backends.cudnn.version(),'numpy':np.__version__,
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'omp_threads':os.environ.get('OMP_NUM_THREADS'),'mkl_threads':os.environ.get('MKL_NUM_THREADS'),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'deterministic':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
