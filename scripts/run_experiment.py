"""Default dry-run. --execute is the ONLY path to datasets, checkpoints or CUDA."""
import argparse
import json
import os
from nmsam.config.experiment import load_experiment_config,plan
from nmsam.utils.capabilities import capabilities


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/gpu_baselines.yaml")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--method");parser.add_argument("--seed",type=int);parser.add_argument("--resume")
    args=parser.parse_args(argv);config,raw=load_experiment_config(args.config)
    if not args.execute:
        print(json.dumps({**plan(config),"capabilities":capabilities()},indent=2));return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    from nmsam.experiment.deployment import execute_training,execute_probe
    path,record=(execute_training(config,raw,selected_method=args.method,selected_seed=args.seed,resume=args.resume)
                 if config.kind=="baseline" else execute_probe(config,raw))
    print(json.dumps({"result":str(path),"run_id":record["run_id"]}));return 0


if __name__=="__main__":raise SystemExit(main())
