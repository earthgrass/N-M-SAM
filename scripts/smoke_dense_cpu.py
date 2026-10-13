"""Run from an editable installation of nmsam-rebuild."""
import argparse
from nmsam.experiment.runner import run_dense_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_sam_official_toy.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    run_dense_cpu(args.config, args.output_dir)

