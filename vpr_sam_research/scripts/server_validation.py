"""Bounded, explicitly invoked CUDA acceptance/diagnostics. Never formal training.

Observers wrap public operations without changing their inputs, outputs or RNG.
Freeze this script and all source/config inputs before generating artifacts.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from vpr_sam.config import load_config
from vpr_sam.data import build_data, loader
from vpr_sam.models import build_model
from vpr_sam.optim import Stepper
from vpr_sam.optim.geometry import Geometry
from vpr_sam.repro import (atomic_json, atomic_save, seed_all, seed_for, source_hashes,
                          tensor_hash, execution_record, environment, file_hash)
from vpr_sam.training import train, prune, evaluate_checkpoint
from vpr_sam.training.engine import read_checkpoint
from vpr_sam.analysis.summary import summarize


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite diagnostic distribution")
    return dict(zip(("min", "mean", "median", "p90", "p95", "p99", "max"),
                    map(float, (x.min(), x.mean(), *np.quantile(x, [.5, .9, .95, .99]), x.max()))))


@torch.no_grad()
def gradient_energy(named_parameters):
    """Global fixed d; grad=None is zero. Independent of SAM's placeholder logs."""
    parameters = list(named_parameters)
    d = sum(p.numel() for _, p in parameters)
    energies, maxima, nonzero, minimum = [], [], 0, float("inf")
    for name, p in parameters:
        if p.grad is None:
            energies.append((name, 0.)); maxima.append(0.); minimum = 0.
            continue
        g = p.grad.detach().double()
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError("Nonfinite diagnostic g1")
        squared = g.square()
        energies.append((name, float(squared.sum())))
        maxima.append(float(squared.max()))
        minimum = min(minimum, float(squared.min()))
        nonzero += int(torch.count_nonzero(g))
    total = sum(value for _, value in energies)
    if not total:
        return {"d": d, "zero": True, "nonzero_fraction": 0., "g1_norm": 0.,
                "c_max": 0., "max_abs_one_minus_c": 0., "layer_energy": {n: 0. for n, _ in energies}}
    cmax, cmin = d * max(maxima) / total, d * minimum / total
    return {"d": d, "zero": False, "nonzero_fraction": nonzero / d, "g1_norm": math.sqrt(total),
            "c_max": cmax, "max_abs_one_minus_c": max(abs(1-cmax), abs(1-cmin)),
            "layer_energy": {name: value / total for name, value in energies}}


class Observer:
    """Per-instance wrappers; no extra forward/backward or mutation of tensors."""
    def __init__(self, stepper, sample_energy=False):
        self.stepper, self.sample_energy = stepper, sample_energy
        self.times, self.energy, self.forwards, self.backwards = {}, None, [], []
        self.inputs, self.base_calls = [], 0
        self.device = stepper.geometry.params[0].device
        self.handles = [stepper.model.register_forward_pre_hook(self.before_forward),
                        stepper.model.register_forward_hook(self.after_forward)]
        self.originals = []
        self.wrap(stepper.geometry, "perturbations", self.perturbations)
        self.wrap(stepper.geometry, "prepare", self.prepare)
        self.wrap(stepper.geometry, "commit", self.commit)
        self.wrap(stepper, "_check_grads", self.check_grads)
        self.wrap(stepper.base, "step", self.base_step)

    def sync(self):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)

    def wrap(self, owner, name, function):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, lambda *a, **k: function(original, *a, **k))

    def before_forward(self, model, args):
        self.sync(); self.forward_start = time.perf_counter()
        self.inputs.append((args[0].data_ptr(), tuple(args[0].shape)))

    def after_forward(self, model, args, output):
        self.sync(); self.forwards.append(time.perf_counter()-self.forward_start)
        self.backward_start = time.perf_counter()

    def check_grads(self, original, stage):
        self.sync(); self.backwards.append(time.perf_counter()-self.backward_start)
        return original(stage)

    def timed(self, label, original, *a, **k):
        self.sync(); start = time.perf_counter()
        result = original(*a, **k)
        self.sync(); self.times[label] = self.times.get(label, 0.) + time.perf_counter()-start
        return result

    def perturbations(self, original, *a, **k):
        if self.sample_energy:
            start = time.perf_counter()
            self.energy = gradient_energy(self.stepper.model.named_parameters())
            self.times["energy_observer_seconds"] = time.perf_counter()-start
        result = self.timed("perturbation_seconds", original, *a, **k)
        eps = result[0]
        self.underflow_coordinates = sum(int(((p.grad != 0) & (e == 0)).sum())
                                        for p, e in zip(self.stepper.geometry.params, eps) if p.grad is not None)
        if self.stepper.geometry.logs:
            total = sum(e.double().square().sum() for e in eps)
            self.layer_budget = {name: float(e.double().square().sum()/total) if float(total) else 0.
                                 for (name, _), e in zip(self.stepper.geometry.mapping, eps)}
        return result

    def prepare(self, original, *a, **k):
        return self.timed("feedback_seconds", original, *a, **k)

    def commit(self, original, *a, **k):
        return self.timed("commit_seconds", original, *a, **k)

    def base_step(self, original, *a, **k):
        self.base_calls += 1
        return original(*a, **k)

    def begin(self):
        self.times, self.forwards, self.backwards, self.inputs = {}, [], [], []
        self.base_calls = 0

    def finish(self):
        assert len(self.inputs) == 2 and self.inputs[0] == self.inputs[1], "Two-pass input mismatch"
        assert self.base_calls == 1, "Base optimizer must update once"
        return {**self.times, "forward_seconds": sum(self.forwards),
                "backward_and_loss_seconds": sum(self.backwards),
                "underflow_coordinates": self.underflow_coordinates,
                "layer_budget": getattr(self, "layer_budget", {})}

    def close(self):
        for h in self.handles: h.remove()
        for owner, name, original in reversed(self.originals): setattr(owner, name, original)


def common(root, device, method="sam", dtype="float64", real=False):
    c = load_config(root/"configs/smoke/resnet18.json")
    c.update(device=device, threads=4)
    c["optimizer"].update(method=method, lr=.1, rho=.05)
    c["geometry"].update(gamma=1e-7, dtype=dtype)
    c["data"].update(workers=2, cutout=0, validation_size=5000 if real else 8,
                     synthetic_train=40, synthetic_test=16, kind="cifar10" if real else "synthetic")
    c["training"].update(epochs=2, batch_size=128 if real else 8)
    c["evaluation"].update(batch_size=256 if real else 8, probe_batches=1)
    return c


def identical(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return a == b


def kernel_acceptance(root, output, device):
    torch.set_num_threads(4); seed_all(17)
    record = execution_record({"device": device, "purpose": "formal", "threads": 4})
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"], text=True)
    assert record["gpu_uuid"] in smi, "CUDA UUID not present in physical GPU inventory"
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.BatchNorm2d(8)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(8, 3, 16, 16, device=device)
    loss = model(x).square().mean(); loss.backward(); optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    matrix = torch.ones(32, 32, device=device)
    assert float((matrix@matrix)[0, 0]) == 32
    z = torch.arange(1, 1025, dtype=torch.float64, device=device)
    assert torch.allclose(z.log().exp(), z, rtol=1e-13, atol=1e-13)
    torch.cuda.synchronize(device)
    raw = {"execution": record, "environment": environment(), "nvidia_smi": smi,
           "python": subprocess.check_output([os.sys.executable, "--version"], text=True).strip(),
           "pip_freeze": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True),
           "torch_path": torch.__file__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "deterministic": torch.are_deterministic_algorithms_enabled(), "kernels": "PASS"}
    atomic_json(output/"hardware_raw.json", raw)
    return {"kernels": "PASS", "device": device, "gpu": record["gpu_name"], "compute_capability": record["compute_capability"]}


def mathematical_acceptance(root, output, device):
    torch.set_num_threads(4); seed_all(17)
    c = common(root, device, "vpr")
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .5, .1], device=device))
    geo = Geometry([("w", parameter)], c["geometry"])
    s = torch.tensor([-.3, .1, .4, -.2], dtype=torch.float64, device=device)
    geo.logs = [s.clone()]
    parameter.grad = torch.tensor([1., -2., 0., 4.], device=device)
    a = s.exp(); g = parameter.grad.double(); rho = c["optimizer"]["rho"]
    reference = rho*a.square()*g/(a*g).norm()
    eps, stats = geo.perturbations(rho)
    assert torch.allclose(eps[0].double(), reference, rtol=2e-7, atol=1e-10)
    expected = s+c["geometry"]["gamma"]*(1-4*eps[0].double().square()/eps[0].double().square().sum())
    geo.prepare(eps, 0); geo.commit()
    assert torch.equal(geo.logs[0], expected)
    assert math.isclose(stats["ellipsoid_norm"], rho, rel_tol=2e-5)
    # ResNet actual step: restored parameters checked at base-step entry.
    model = build_model("resnet18").to(device); stepper = Stepper(model, c)
    original = [p.detach().clone() for p in model.parameters()]
    bn_before = {n: int(v) for n, v in model.named_buffers() if n.endswith("num_batches_tracked")}
    base = stepper.base.step
    def checked_base(*args, **kwargs):
        assert all(torch.equal(p, q) for p, q in zip(model.parameters(), original)), "Parameter restoration failed"
        return base(*args, **kwargs)
    stepper.base.step = checked_base
    observer = Observer(stepper); observer.begin()
    info = stepper.step(torch.randn(8, 3, 32, 32, device=device), torch.arange(8, device=device)%10)
    observer.finish(); observer.close()
    assert all(int(v) == bn_before[n]+1 for n, v in model.named_buffers() if n in bn_before)
    assert stepper.geometry.updates == 1 and info["d"] == sum(p.numel() for p in model.parameters())
    assert math.isfinite(info["g1_norm"]) and math.isfinite(info["g2_norm"])
    return {"nonunit_ellipsoid_reference": "PASS", "restoration": "PASS", "two_pass_bn_base": "PASS", "step": info}


def data_acceptance(root, output, device):
    c = common(root, device, real=True); c["data"]["download"] = True
    datasets, manifest = build_data(c, root)
    assert len(datasets["train"])+len(datasets["validation"]) == 50000
    assert len(datasets["test"]) == 10000
    assert not set(manifest["train_indices"]) & set(manifest["validation_indices"])
    atomic_json(output/"data_manifest.json", manifest)
    return {"identity": manifest["identity"], "fingerprint": manifest["fingerprint"],
            "train": len(datasets["train"]), "validation": len(datasets["validation"]), "test": len(datasets["test"]),
            "root": str(root/c["data"]["root"]), "augmentation": manifest["augmentation"], "test_used_for_selection": False}


def resume_smoke(root, output, device):
    records, paths = [], []
    for method in ("sam", "vpr"):
        c = common(root, device, method); c["output"] = f"runs/server_acceptance/{method}_continuous"
        train(c, root); left, _ = read_checkpoint(c["output"]+"/final.pt", root)
        for boundary in ("batch", "epoch"):
            split = copy.deepcopy(c); split["output"] = f"runs/server_acceptance/{method}_{boundary}_resume"
            # Last batch of an epoch is checkpointed before validation/scheduler.
            # Also exercise a true post-scheduler checkpoint via callback on save.
            if boundary == "batch":
                train(split, root, stop_after=1)
            else:
                from vpr_sam.training import engine
                save = engine.atomic_save
                class BoundaryStop(Exception): pass
                def save_boundary(path, payload):
                    save(path, payload)
                    if path.name == "last.pt" and payload["epoch"] == 1 and payload["next_batch"] == 0:
                        raise BoundaryStop()
                engine.atomic_save = save_boundary
                try:
                    try: train(split, root)
                    except BoundaryStop: pass
                finally: engine.atomic_save = save
                # The deliberate harness interruption is recorded by engine,
                # while the completed, sealed boundary checkpoint remains healthy.
            train(split, root, resume=split["output"]+"/last.pt")
            right, _ = read_checkpoint(split["output"]+"/final.pt", root)
            fields = ("model", "stepper", "scheduler", "rng", "order_hash", "epoch", "next_batch", "accumulator", "initialization_hash")
            checks = {field: identical(left[field], right[field]) for field in fields}
            assert all(checks.values()), f"Exact resume failed {method} {boundary}: {checks}"
            # Step timing is deliberately excluded from trajectory equality.
            rows = lambda state: [{k:v for k,v in row.items() if k != "step_seconds"} for row in state["step_rows"]]
            assert identical(rows(left), rows(right)), "Step history mismatch"
            records.append({"method": method, "boundary": boundary, "workers": 2, "checks": checks})
        hard = prune(c["output"]+"/final.pt", root)
        parent, parent_path = read_checkpoint(hard["checkpoint"], root)
        train(parent["config"], root, recover_from=hard["checkpoint"])
        recovered, _ = read_checkpoint(parent_path.parent/"recovery/final.pt", root)
        for name, mask in recovered["masks"].items():
            assert not torch.count_nonzero(recovered["model"][name][~mask.bool()])
        assert parent["schema"] == recovered["schema"] == left["schema"] == 2
        paths.append(c["output"])
    summary = summarize(root, paths, "runs/server_acceptance/analysis")
    real = []
    for method in ("sam", "vpr"):
        c = common(root, device, method, real=True); c["output"] = f"runs/server_acceptance/{method}_real"
        real.append(train(c, root, stop_after=3))
    return {"resume": records, "prune_recover_schema2": "PASS", "real_short": real, "paired_n": summary["paired"]["vpr"]["n"]}


def diagnostic_run(root, output, device, method, dtype, steps, gamma, tag):
    c = common(root, device, method, dtype, real=True); c["geometry"]["gamma"] = gamma
    # Diagnostic budget is explicitly steps, cosine horizon is recorded; never a formal run.
    c["training"]["epochs"] = 2
    torch.set_num_threads(c["threads"]); seed_all(seed_for(c["seed"], "initialization"))
    datasets, manifest = build_data(c, root)
    model = build_model(c["model"]).to(device)
    init_hash = tensor_hash(model.state_dict()); stepper = Stepper(model, c)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base, 2)
    observer = Observer(stepper, sample_energy=method == "sam" and not tag.startswith("performance"))
    destination = output/tag
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir()
    atomic_json(destination/"config.json", c); atomic_json(destination/"source_hashes.json", source_hashes(root))
    records, energies = [], []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        step = 0
        for epoch in range(2):
            iterator = iter(loader(datasets["train"], 128, c["seed"], epoch, workers=2, shuffle=True))
            while step < steps:
                load_start = time.perf_counter()
                try: x, y, ids = next(iterator)
                except StopIteration: break
                load_seconds = time.perf_counter()-load_start
                bn_before = {n:int(v) for n,v in model.named_buffers() if n.endswith("num_batches_tracked")}
                observer.begin(); torch.cuda.synchronize(device); t = time.perf_counter()
                info = stepper.step(x.to(device), y.to(device)); torch.cuda.synchronize(device)
                wall = time.perf_counter()-t
                assert all(int(v) == bn_before[n]+1 for n,v in model.named_buffers() if n in bn_before)
                observed = observer.finish()
                step += 1
                records.append({"step": step, "epoch": epoch, "step_seconds_instrumented": wall,
                                "data_loader_seconds": load_seconds, "ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                                **info, **observed})
                if observer.energy: energies.append({"step": step, **observer.energy})
                if step % 10 == 0 or step <= 3:
                    atomic_json(destination/"progress.json", {"step":step, "elapsed_seconds":time.perf_counter()-started})
                    print(f"{tag} step {step}/{steps}: delta={info['max_abs_delta_s']:.6g}, volume={info['log_volume']:.6g}", flush=True)
                if step >= steps: break
            if step >= steps: break
            scheduler.step()
        assert step == steps, "Diagnostic budget not completed"
        observer.close()
        t = time.perf_counter()
        atomic_save(destination/"diagnostic.pt", {"kind":"DIAGNOSTIC_ONLY_NOT_TRAIN_RESUME_OR_FORMAL_INIT", "model":model.state_dict(),
                    "stepper":stepper.state_dict(), "scheduler":scheduler.state_dict(), "source_hashes":source_hashes(root)})
        save_seconds = time.perf_counter()-t
        result = {"status":"PASS", "steps":step, "initialization_hash":init_hash, "dataset_fingerprint":manifest["fingerprint"],
                  "config":c, "peak_cuda_bytes":torch.cuda.max_memory_allocated(device),
                  "checkpoint_seconds":save_seconds, "checkpoint_bytes":(destination/"diagnostic.pt").stat().st_size,
                  "elapsed_seconds":time.perf_counter()-started, "records":records, "energy":energies,
                  "timing_scope":"synchronized wall timings including observer overhead; backward includes loss"}
        atomic_json(destination/"result.json", result)
        return result
    except BaseException as exc:
        # Do not save a poisoned Stepper or claim partial state is recoverable.
        atomic_json(destination/"failure.json", {"step":stepper.steps, "type":type(exc).__name__, "message":str(exc),
                    "config":c, "checkpoint":"none; failed object cannot be serialized", "records":records})
        raise
    finally:
        observer.close()


def gradient_calibration(root, output, device):
    result = diagnostic_run(root, output, device, "sam", "float64", 300, 1e-7, "sam_g1_300")
    rows = result["energy"]; valid = [x for x in rows if not x["zero"]]
    denominator = distribution([x["max_abs_one_minus_c"] for x in valid])
    gamma0 = .05/denominator["p99"]
    assert math.isfinite(gamma0) and 0 < gamma0 < .25
    summary = {"d": rows[0]["d"], "steps":len(rows), "independent_seeds":1,
               "c_max":distribution([x["c_max"] for x in valid]), "denominator":denominator,
               "nonzero_fraction":distribution([x["nonzero_fraction"] for x in rows]),
               "gradient_norm":distribution([x["g1_norm"] for x in rows]),
               "zero_steps":sum(x["zero"] for x in rows), "target_delta":.05, "gamma0":gamma0,
               "candidates":[gamma0*.5, gamma0, gamma0*2], "not_accuracy_selection":True,
               "robustness":{"using_max_instead_q99":.05/denominator["max"],
                             "using_q95":.05/denominator["p95"]}}
    atomic_json(output/"gamma_estimate.json", summary)
    return summary


def candidates(root, output, device):
    estimate = json.loads((output/"gamma_estimate.json").read_text())
    results = []
    for index, gamma in enumerate(estimate["candidates"]):
        # Each starts afresh; immediate failures propagate and are preserved.
        results.append(diagnostic_run(root, output, device, "vpr", "float64", 100, gamma, f"vpr_candidate_{index}"))
    assert len({r["initialization_hash"] for r in results}) == 1
    sam = json.loads((output/"sam_g1_300/result.json").read_text())
    assert results[0]["initialization_hash"] == sam["initialization_hash"]
    return {"candidates":[{"gamma":r["config"]["geometry"]["gamma"], "status":r["status"], "steps":r["steps"]} for r in results]}


def performance(root, output, device):
    results = []
    for method, dtype in (("sam", "float64"), ("vpr", "float64"), ("vpr", "float32")):
        result = diagnostic_run(root, output, device, method, dtype, 12, 1e-7, f"performance_{method}_{dtype}")
        rows = result["records"][2:]
        results.append({"method":method, "dtype":dtype, "steps_measured":len(rows), "warmup_discarded":2,
                        "peak_cuda_bytes":result["peak_cuda_bytes"], "checkpoint_seconds":result["checkpoint_seconds"],
                        "checkpoint_bytes":result["checkpoint_bytes"],
                        "timings":{key:distribution([row.get(key,0.) for row in rows]) for key in
                                   ("step_seconds_instrumented","forward_seconds","backward_and_loss_seconds","perturbation_seconds",
                                    "feedback_seconds","commit_seconds","data_loader_seconds","energy_observer_seconds")},
                        "max_abs_volume":max(abs(row["log_volume"]) for row in rows),
                        "max_ellipsoid_error":max(abs(row["ellipsoid_norm"]-.05) for row in rows)})
    return {"results":results, "not_exclusive_gpu_certification":True, "gamma":1e-7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("kernels","math","data","smoke","performance","gradients","candidates"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", required=True)
    args = parser.parse_args(); root = args.root.resolve()
    output = root/"artifacts/server_acceptance"; output.mkdir(parents=True, exist_ok=True)
    if (output/f"{args.phase}.json").exists(): raise FileExistsError("Refusing to overwrite phase evidence")
    phases = {"kernels":kernel_acceptance,"math":mathematical_acceptance,"data":data_acceptance,
              "smoke":resume_smoke,"performance":performance,"gradients":gradient_calibration,"candidates":candidates}
    try:
        before = source_hashes(root)
        result = phases[args.phase](root, output, args.device)
        assert before == source_hashes(root), "Source/config set changed during phase"
        atomic_json(output/f"{args.phase}.json", {"status":"PASS", "source_hashes":before, "result":result})
    except BaseException as exc:
        atomic_json(output/f"{args.phase}_failure.json", {"status":"FAIL", "type":type(exc).__name__, "message":str(exc),
                    "traceback":traceback.format_exc(), "source_hashes":source_hashes(root)})
        raise


if __name__ == "__main__": main()
