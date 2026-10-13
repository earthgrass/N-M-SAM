from pathlib import Path
import hashlib
import json
import platform
import subprocess
from uuid import uuid4
import torch
import torchvision
from nmsam import __version__


def config_fingerprint(config):
    scientific=dict(config)
    for key in ("archive_path","dataset_cache_dir","output_dir","checkpoint","resume"):
        scientific.pop(key,None)
    return hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()


def save_result(kind,config,payload,output_dir="results"):
    root=Path(__file__).resolve().parents[3]
    def git(*args):
        r=subprocess.run(["git","-C",str(root),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else "unavailable"
    raw=json.dumps(config,sort_keys=True)
    result={"run_id":uuid4().hex,"git_sha":git("rev-parse","HEAD"),
        "git_dirty":bool(git("status","--porcelain")),"config":config,
        "config_sha256":config_fingerprint(config),"source_config_sha256":hashlib.sha256(payload.get("config_raw",raw).encode()).hexdigest(),
        "seed":config.get("seed",config.get("seeds",[0])),"code_version":__version__,
        "python":platform.python_version(),"torch":torch.__version__,"torchvision":torchvision.__version__,
        "device":config.get("device","cpu"),**payload}
    path=Path(output_dir)/f"{kind}_{result['run_id']}.json"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return path,result
