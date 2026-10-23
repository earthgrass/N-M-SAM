"""Explicit server preflight; does not train or download data."""
import argparse
import json
from pathlib import Path
import torch
from vpr_sam.config import load_config, validate
from vpr_sam.data import build_data
from vpr_sam.repro import environment, project_path, execution_record, seed_all


def target_record(device, threads=1):
    """Resolve the requested logical index, including its actual UUID."""
    requested = torch.device(device)
    if requested.type != "cuda" or requested.index is None:
        raise ValueError("5090 preflight requires explicit cuda:N")
    return execution_record({"device": device, "threads": threads, "purpose": "formal"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--config")
    p.add_argument("--require-5090", action="store_true")
    p.add_argument("--device", default="cuda:0", help="Explicit logical CUDA target for hardware preflight")
    args = p.parse_args()
    result = environment()
    if args.require_5090:
        seed_all(0)
        result["execution"] = target_record(args.device, torch.get_num_threads())
        result["gpu"] = result["execution"]["gpu_name"]
        x = torch.ones(32, 32, device=args.device)
        if float((x@x)[0, 0]) != 32:
            raise RuntimeError("CUDA kernel preflight failed")
    if args.config:
        config = load_config(project_path(args.root, args.config))
        validate(config, formal_allowed=True)
        if args.require_5090 and config["device"] != args.device:
            raise ValueError("Preflight target differs from config device")
        if config["data"]["download"]:
            raise ValueError("Preflight requires already prepared data; download must be false")
        _, manifest = build_data(config, args.root)
        result["dataset"] = {k: manifest[k] for k in ("identity", "fingerprint", "test_size")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
