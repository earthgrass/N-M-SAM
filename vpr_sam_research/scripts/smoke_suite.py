"""Explicit short CPU-only engineering acceptance suite. Never formal training."""
import argparse
import copy
from pathlib import Path

from vpr_sam.config import load_config
from vpr_sam.training import train,prune,evaluate_checkpoint
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.analysis.report import summarize,plot
from vpr_sam.repro import atomic_json,project_path,environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path.cwd())
    p.add_argument("--run-id",required=True,help="New child directory under runs/")
    p.add_argument("--include-resnet",action="store_true")
    p.add_argument("--seeds",nargs="+",type=int,help="Explicit smoke seeds; not a formal seed recommendation")
    p.add_argument("--methods",nargs="+",choices=("sgd","sam","vpr","static","positive","fixed_l2"),
                   default=["sgd","sam","vpr","static","positive","fixed_l2"])
    args=p.parse_args()
    root=args.root.resolve()
    destination=project_path(root,f"runs/{args.run_id}")
    if destination.exists():raise FileExistsError(destination)
    paths=[];reports=[]
    for method,seed in [(m,s) for m in args.methods for s in (args.seeds or [17])]:
        c=load_config(root/f"configs/smoke/{method}.json")
        c["seed"]=seed
        tag=method if args.seeds is None else f"{method}_seed{seed}"
        c["output"]=f"runs/{args.run_id}/{tag}"
        result=train(c,root)
        pruned=prune(c["output"]+"/final.pt",root)
        state,_=read_checkpoint(pruned["checkpoint"],root)
        recovery=train(state["config"],root,recover_from=pruned["checkpoint"])
        evaluated=evaluate_checkpoint(pruned["checkpoint"],root,"test")
        evaluate_checkpoint(c["output"]+"/final.pt",root,"test")
        evaluate_checkpoint(str((root/pruned["checkpoint"]).parent/"recovery/final.pt"),root,"test")
        reports.append({"training":result,"pruning":pruned,"recovery":recovery,"evaluation":evaluated})
        paths.append(c["output"])
        print(f"PASS {method} seed {seed}: train -> prune 1:3 -> recover -> evaluate",flush=True)
    summarize(root,paths,f"runs/{args.run_id}/analysis")
    plot(root,f"runs/{args.run_id}/analysis/summary.json",f"runs/{args.run_id}/analysis/figures")
    summarize(root,paths,f"runs/{args.run_id}/test_analysis",split="test")
    if args.include_resnet:
        c=load_config(root/"configs/smoke/resnet18.json")
        c["output"]=f"runs/{args.run_id}/resnet18"
        result=train(c,root)
        reports.append({"resnet18_two_step_training":result})
        print("PASS ResNet18 FP32 VPR-SAM two-step smoke",flush=True)
    atomic_json(destination/"acceptance.json",{"environment":environment(),"reports":reports,
                "formal_training_started":False,"dataset":"SYNTHETIC-SMOKE-ONLY"})
    print(f"Acceptance artifacts: {destination}")


if __name__=="__main__":main()
