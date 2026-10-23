"""Explicit synthetic CPU/CUDA smoke: train/recovery resume, step time and memory.

No SSH, downloads, formal configs or job submission. CUDA is never selected by
default. Passing on CPU does not certify any CUDA path.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from vpr_sam.config import load_config
from vpr_sam.repro import atomic_json,environment,project_path
from vpr_sam.training import train,prune
from vpr_sam.training.engine import read_checkpoint


def identical(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


def compare(a,b):
    for field in ("model","stepper","scheduler","rng","order_hash"):
        if not identical(a[field],b[field]):raise AssertionError(f"Resume trajectory mismatch: {field}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--model",choices=("tiny","resnet18"),default="resnet18")
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args();root=args.root.resolve()
    destination=project_path(root,f"runs/{args.run_id}")
    if destination.exists():raise FileExistsError(destination)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    results=[]
    for method in ("sam","vpr"):
        config=load_config(root/"configs/smoke/resnet18.json")
        config.update(model=args.model,device=args.device,output=f"runs/{args.run_id}/{method}_continuous")
        config["optimizer"]["method"]=method
        continuous=train(config,root)
        left,_=read_checkpoint(config["output"]+"/final.pt",root)
        split=copy.deepcopy(config);split["output"]=f"runs/{args.run_id}/{method}_resume"
        train(split,root,stop_after=1)
        train(split,root,resume=split["output"]+"/last.pt")
        right,_=read_checkpoint(split["output"]+"/final.pt",root)
        compare(left,right)
        hard=prune(config["output"]+"/final.pt",root)
        parent,parent_path=read_checkpoint(hard["checkpoint"],root)
        recovered=train(parent["config"],root,recover_from=hard["checkpoint"])
        recovery_left,_=read_checkpoint(parent_path.parent/"recovery/final.pt",root)
        clone=parent_path.parent/"resume_copy"
        clone.mkdir();shutil.copy2(parent_path,clone/"pruned.pt")
        train(parent["config"],root,recover_from=clone/"pruned.pt",stop_after=1)
        train(parent["config"],root,recover_from=clone/"pruned.pt",resume=clone/"recovery/last.pt")
        recovery_right,_=read_checkpoint(clone/"recovery/final.pt",root)
        compare(recovery_left,recovery_right)
        steps=json.loads((root/config["output"]/"steps.json").read_text(encoding="utf-8"))
        results.append({"method":method,"train_resume_exact":True,"recovery_resume_exact":True,
                        "training":continuous,"recovery":recovered,
                        "step_seconds":[s["step_seconds"] for s in steps],
                        "d_p_max":[s["d_p_max"] for s in steps],
                        "max_abs_delta_s":[s["max_abs_delta_s"] for s in steps]})
        print(f"PASS {method} {args.device}: train/recovery exact resume",flush=True)
    atomic_json(destination/"device_acceptance.json",{"device":args.device,"model":args.model,
                "environment":environment(),"dataset":"SYNTHETIC-SMOKE-ONLY","results":results,
                "formal_training_started":False})


if __name__=="__main__":main()
