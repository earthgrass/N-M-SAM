"""Explicit server preflight; does not train or download data."""
import argparse
import json
from pathlib import Path
import torch
from vpr_sam.config import load_config, validate
from vpr_sam.data import build_data
from vpr_sam.repro import environment, project_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--config")
    p.add_argument("--require-5090", action="store_true")
    args = p.parse_args()
    result = environment()
    if args.require_5090:
        if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("RTX5090 CUDA device required")
        if torch.cuda.get_device_capability(0) != (12, 0):
            raise RuntimeError("Unexpected RTX5090 compute capability")
        x = torch.ones(32, 32, device="cuda")
        if float((x@x)[0, 0]) != 32:
            raise RuntimeError("CUDA kernel preflight failed")
    if args.config:
        config = load_config(project_path(args.root, args.config))
        validate(config, formal_allowed=True)
        if config["data"]["download"]:
            raise ValueError("Preflight requires already prepared data; download must be false")
        _, manifest = build_data(config, args.root)
        result["dataset"] = {k: manifest[k] for k in ("identity", "fingerprint", "test_size")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
