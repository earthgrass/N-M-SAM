"""File and tensor identities; compatible with historical schema2 digests."""
import hashlib,json,os,platform,random
from pathlib import Path
import numpy as np
import torch

def sha_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def tensor_sha(values):
    h=hashlib.sha256()
    for name,value in values.items():
        t=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode())
        h.update(str(list(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def finite(values,label):
    for v in values:
        if not bool(torch.isfinite(v).all()):raise FloatingPointError(label)

def seed_all(seed,threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(threads)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['cpu'])
    if value['cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG required')
        torch.cuda.set_rng_state_all(value['cuda'])

def environment():
    import torchvision
    return {'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled()}
