"""Run from an editable installation of nmsam-rebuild."""
import argparse
import json
from nmsam.config.loader import load_dense_config
from nmsam.experiment.runner import run_dense_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_sam_official_toy.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    config,_=load_dense_config(args.config)
    if not args.execute:
        print(json.dumps({"dry_run":True,"status":"GPU_VALIDATION_PENDING","config":args.config}));raise SystemExit(0)
    run_dense_cpu(args.config, args.output_dir)

