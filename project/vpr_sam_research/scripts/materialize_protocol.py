"""Expand ONE user-approved complete common config into paired method/seed files.

This script does not choose hyperparameters or launch jobs. Existing files are
never overwritten. Static_std must be positive if static is requested.
"""
import argparse
import copy
import json
from pathlib import Path
from vpr_sam.config import load_config,validate
from vpr_sam.repro import atomic_json,project_path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path.cwd())
    p.add_argument("--common",required=True)
    p.add_argument("--seeds",nargs="+",required=True,type=int)
    p.add_argument("--methods",nargs="+",default=["sgd","sam","vpr"])
    p.add_argument("--output",default="configs/approved")
    args=p.parse_args()
    common=load_config(project_path(args.root,args.common))
    pending=[]
    for seed in args.seeds:
        for method in args.methods:
            c=copy.deepcopy(common)
            c.update(seed=seed,output=f"runs/formal/{method}_seed{seed}")
            c["optimizer"]["method"]=method
            validate(c,formal_allowed=True)
            target=project_path(args.root,f"{args.output}/{method}_seed{seed}.json")
            if target.exists():raise FileExistsError(target)
            pending.append((target,c))
    if len({str(path) for path,_ in pending})!=len(pending):raise ValueError("Duplicate seed/method")
    for path,c in pending:atomic_json(path,c)
    print(json.dumps({"generated":[str(p.relative_to(args.root)) for p,_ in pending]},indent=2))


if __name__=="__main__":main()
