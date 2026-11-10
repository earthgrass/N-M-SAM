"""Explicit non-training precision/performance microbenchmark. No dataset/model forward."""
import argparse
import time
from pathlib import Path

import torch
from vpr_sam.optim import Geometry
from vpr_sam.repro import atomic_json, environment, project_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--device", default="cpu")
    p.add_argument("--dimensions", nargs="+", type=int, default=[10000, 100000])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output", default="artifacts/geometry_benchmark.json")
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = []
    for d in args.dimensions:
        references = {}
        for dtype in ("float64", "float32"):
            gen = torch.Generator().manual_seed(73)
            param = torch.nn.Parameter(torch.zeros(d, device=args.device))
            param.grad = torch.randn(d, generator=gen).to(args.device)
            g = Geometry([("weight", param)], {"dtype": dtype, "gamma": 1e-5, "warmup_steps": 0,
                         "volume_atol": 1e-5, "volume_rtol": 1e-6, "static_std": 0.})
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for step in range(args.steps):
                eps, info = g.perturbations(.05)
                g.prepare(eps, step, info["zero_perturbation"])
                g.commit()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            references[dtype] = g.logs[0].double().cpu()
            rows.append({"d": d, "dtype": dtype, "steps": args.steps,
                         "seconds_per_update": elapsed/args.steps,
                         "state_bytes": sum(s.numel()*s.element_size() for s in g.logs),
                         "log_volume": g.diagnostics()["log_volume"],
                         "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None})
        rows[-1]["max_abs_difference_from_fp64"] = float((references["float32"]-references["float64"]).abs().max())
    atomic_json(project_path(args.root, args.output), {"environment": environment(), "rows": rows,
                "note": "Microbenchmark only; CPU timings do not predict RTX5090 performance."})
    print(rows)


if __name__ == "__main__":
    main()
