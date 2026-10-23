"""Random-init training, exact batch-boundary resume, hard pruning and recovery."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from ..config import validate, validate_pruning, recovery_config
from ..data import build_data, loader
from ..evaluation import evaluate, sphere_probe, output_change
from ..models import build_model
from ..optim import Stepper
from ..pruning import make_masks, apply_masks, validate_masks
from ..repro import (atomic_json, atomic_save, environment, file_hash, project_path,
                     restore_rng, rng_state, seed_all, seed_for, source_hashes, tensor_hash)


def read_checkpoint(path, root):
    path = project_path(root, str(path))
    # Only load locally generated/trusted checkpoints; torch pickle is not a safe interchange format.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return state, path


def _eval_loader(datasets, split, c, epoch=0):
    return loader(datasets[split], c["evaluation"]["batch_size"], c["seed"], epoch,
                  workers=c["data"]["workers"])


def _check_resume(saved, c, root, manifest, stage, parent):
    if saved["config"] != c or saved["stage"] != stage:
        raise ValueError("Resume config/stage mismatch")
    if saved["manifest"] != manifest:
        raise ValueError("Resume dataset/split mismatch")
    if saved["source_hashes"] != source_hashes(root):
        raise ValueError("Source/config/scripts changed since checkpoint; exact resume refused")
    env = environment()
    for key in ("python", "platform", "packages", "cuda_runtime", "gpu", "cuda_capabilities", "deterministic"):
        if saved["environment"][key] != env[key]:
            raise ValueError(f"Exact resume environment mismatch: {key}")
    if saved["parent"] != parent:
        raise ValueError("Recovery parent mismatch")


def train(c, root: Path, *, formal_allowed=False, resume=None, recover_from=None, stop_after=None):
    """No implicit checkpoint discovery or warm start. stop_after is smoke-only."""
    c = copy.deepcopy(c)
    validate(c, formal_allowed=formal_allowed)
    if stop_after is not None and (c["purpose"] != "smoke" or stop_after < 1):
        raise ValueError("Positive stop_after is only allowed for explicit smoke runs")
    root = root.resolve()
    stage, parent, parent_payload = "train", None, None
    if recover_from:
        parent_payload, parent_path = read_checkpoint(recover_from, root)
        if parent_payload.get("stage") != "pruned":
            raise ValueError("Recovery must start from an explicit hard-pruned checkpoint")
        if parent_payload["config"] != c:
            raise ValueError("Recovery input config must match pruning artifact")
        stage, parent = "recovery", {"path": str(parent_path.relative_to(root)), "sha256": file_hash(parent_path)}
        c = recovery_config(c)
    output = (parent_path.parent / "recovery" if stage == "recovery" else project_path(root, c["output"]))
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    if resume is not None and not output.exists():
        raise ValueError("Resume requires the original run directory")
    torch.set_num_threads(c["threads"])
    seed_all(seed_for(c["seed"], "initialization"), c["deterministic"])
    datasets, manifest = build_data(c, root)
    device = torch.device(c["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; CPU fallback forbidden")
    model = build_model(c["model"]).to(device)
    init_hash = tensor_hash(model.state_dict())
    masks = None
    if stage == "recovery":
        model.load_state_dict(parent_payload["model"])
        masks = parent_payload["masks"]
        if parent_payload["manifest"] != manifest:
            raise ValueError("Pruning/recovery data identity mismatch")
        init_hash = parent_payload["initialization_hash"]
        apply_masks(model, masks)
        validate_masks(model, masks, c["pruning"])
    stepper = Stepper(model, c, masks)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stepper.base, c["training"]["epochs"], eta_min=c["training"]["eta_min"])
    epoch, next_batch = 0, 0
    epoch_rows, step_rows = [], []
    accumulator = {"loss": 0., "adversarial_loss": 0., "correct": 0, "count": 0}
    best = float("-inf")
    order_hash = hashlib.sha256(b"vpr-order-v1").hexdigest()
    elapsed_before = 0.
    provenance, env = source_hashes(root), environment()
    if resume:
        state, resume_path = read_checkpoint(resume, root)
        if resume_path.parent != output:
            raise ValueError("Resume checkpoint must belong to the configured output directory")
        _check_resume(state, c, root, manifest, stage, parent)
        model.load_state_dict(state["model"])
        stepper.load_state_dict(state["stepper"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, next_batch = state["epoch"], state["next_batch"]
        epoch_rows, step_rows, accumulator = state["epoch_rows"], state["step_rows"], state["accumulator"]
        best, order_hash = state["best_validation_accuracy"], state["order_hash"]
        init_hash, elapsed_before = state["initialization_hash"], state["elapsed_seconds"]
        restore_rng(state["rng"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", c)
    atomic_json(output / "data_manifest.json", manifest)
    atomic_json(output / "environment.json", env)
    atomic_json(output / "source_sha256.json", provenance)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save(name="last.pt"):
        payload = {"schema": 1, "stage": stage, "config": c, "model": model.state_dict(),
                   "stepper": stepper.state_dict(), "scheduler": scheduler.state_dict(),
                   "epoch": epoch, "next_batch": next_batch, "epoch_rows": epoch_rows,
                   "step_rows": step_rows, "accumulator": accumulator,
                   "best_validation_accuracy": best, "order_hash": order_hash,
                   "initialization_hash": init_hash, "manifest": manifest, "parent": parent,
                   "masks": masks, "rng": rng_state(), "source_hashes": provenance, "environment": env,
                   "elapsed_seconds": elapsed_before + time.perf_counter()-started}
        atomic_save(output / name, payload)
        atomic_json(output / "epochs.json", epoch_rows)
        atomic_json(output / "steps.json", step_rows)

    try:
        while epoch < c["training"]["epochs"]:
            train_loader = loader(datasets["train"], c["training"]["batch_size"], c["seed"], epoch,
                                  next_batch, c["data"]["workers"], shuffle=True)
            for x, y, ids in train_loader:
                order_hash = hashlib.sha256(bytes.fromhex(order_hash) + ids.numpy().tobytes()).hexdigest()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                info = stepper.step(x.to(device), y.to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                info["step_seconds"] = time.perf_counter()-t0
                info.pop("layer_log_volume")  # epoch-level per-layer logs below
                info.update(step=stepper.steps, epoch=epoch, batch=next_batch,
                            lr=stepper.base.param_groups[0]["lr"], order_hash=order_hash)
                step_rows.append(info)
                for key in ("loss", "adversarial_loss"):
                    accumulator[key] += info["center_loss" if key == "loss" else key] * info["count"]
                accumulator["correct"] += info["correct"]
                accumulator["count"] += info["count"]
                next_batch += 1
                if masks:
                    validate_masks(model, masks, c["pruning"])
                interval = c["training"]["checkpoint_interval_steps"]
                if interval and stepper.steps % interval == 0:
                    save()
                if stop_after is not None and stepper.steps >= stop_after:
                    save()
                    return {"complete": False, "stage": stage, "steps": stepper.steps, "output": str(output)}
            val = evaluate(model, _eval_loader(datasets, "validation", c), device)
            count = accumulator["count"]
            row = {"epoch": epoch+1, "step": stepper.steps, "lr": stepper.base.param_groups[0]["lr"],
                   "train_loss": accumulator["loss"]/count, "adversarial_train_loss": accumulator["adversarial_loss"]/count,
                   "train_accuracy": accumulator["correct"]/count, "validation": val,
                   "geometry": stepper.geometry.diagnostics()}
            epoch_rows.append(row)
            is_best = val["accuracy"] > best
            best = max(best, val["accuracy"])
            scheduler.step()
            epoch += 1
            next_batch = 0
            accumulator = {"loss": 0., "adversarial_loss": 0., "correct": 0, "count": 0}
            save()
            if is_best:
                save("best_validation.pt")
        save("final.pt")
        result = {"complete": True, "stage": stage, "purpose": c["purpose"], "method": c["optimizer"]["method"],
                  "seed": c["seed"], "steps": stepper.steps, "epochs": epoch,
                  "best_validation_accuracy": best, "final_validation": epoch_rows[-1]["validation"],
                  "initialization_hash": init_hash, "order_hash": order_hash,
                  "dataset_fingerprint": manifest["fingerprint"],
                  "elapsed_seconds": elapsed_before+time.perf_counter()-started,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "output": str(output), "parent": parent}
        atomic_json(output / "summary.json", result)
        return result
    except BaseException as exc:
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                             "step": stepper.steps, "last_checkpoint_may_precede_failure": True})
        raise


def evaluate_checkpoint(checkpoint, root, split="validation", probe=True):
    state, path = read_checkpoint(checkpoint, root)
    c = state["config"]
    torch.set_num_threads(c["threads"])
    seed_all(seed_for(c["seed"], "evaluation"), c["deterministic"])
    datasets, manifest = build_data(c, root)
    if state["manifest"] != manifest:
        raise ValueError("Evaluation data mismatch")
    device = torch.device(c["device"])
    model = build_model(c["model"]).to(device)
    model.load_state_dict(state["model"])
    masks = state.get("masks")
    result = {"split": split, "checkpoint_sha256": file_hash(path), "purpose": c["purpose"],
              "metrics": evaluate(model, _eval_loader(datasets, split, c), device),
              "probe_protocol": "common first-order L2 sphere; masked subspace after hard pruning"}
    if probe:
        values = []
        for index, (x, y, _) in enumerate(_eval_loader(datasets, split, c)):
            if index >= c["evaluation"]["probe_batches"]:
                break
            item = sphere_probe(model, x.to(device), y.to(device), c["evaluation"]["probe_rho"], masks)
            item["count"] = len(y)
            values.append(item)
        result["probe_batches"] = values
    atomic_json(path.parent / f"evaluation_{path.stem}_{split}.json", result)
    return result


def prune(checkpoint, root, protocol=None):
    state, path = read_checkpoint(checkpoint, root)
    if state["stage"] != "train":
        raise ValueError("Pruning requires a dense training checkpoint")
    c = copy.deepcopy(state["config"])
    if protocol is not None:
        c["pruning"] = protocol
    validate_pruning(c["pruning"])
    torch.set_num_threads(c["threads"])
    seed_all(seed_for(c["seed"], "evaluation"), c["deterministic"])
    datasets, manifest = build_data(c, root)
    if manifest != state["manifest"]:
        raise ValueError("Pruning data mismatch")
    device = torch.device(c["device"])
    original = build_model(c["model"]).to(device)
    original.load_state_dict(state["model"])
    pruned = copy.deepcopy(original)
    masks, layer_report = make_masks(pruned, c["pruning"])
    apply_masks(pruned, masks)
    validate_masks(pruned, masks, c["pruning"])
    p = c["pruning"]
    tag = f"n{p['n']}_m{p['m']}_{p['scope']}_{p['score']}_{p['tail']}_{path.stem}"
    output = path.parent / "pruning" / tag
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pruning artifacts: {output}")
    dense_result = evaluate(original, _eval_loader(datasets, "validation", c), device)
    sparse_result = evaluate(pruned, _eval_loader(datasets, "validation", c), device)
    change = output_change(original, pruned, _eval_loader(datasets, "validation", c), device)
    report = {"protocol": p, "dense_validation": dense_result, "hard_validation": sparse_result,
              "hard_loss_increase": sparse_result["loss"]-dense_result["loss"],
              "accuracy_drop": dense_result["accuracy"]-sparse_result["accuracy"],
              "output_change": change, "layers": layer_report, "bn_calibration": "none",
              "global_parameter_sparsity": sum(r["masked"] for r in layer_report)/sum(p.numel() for p in pruned.parameters()),
              "parent_sha256": file_hash(path), "purpose": c["purpose"]}
    payload = {"schema": 1, "stage": "pruned", "config": c, "model": pruned.state_dict(),
               "masks": {name: mask.cpu() for name, mask in masks.items()}, "manifest": manifest,
               "initialization_hash": state["initialization_hash"], "report": report,
               "source_hashes": source_hashes(root), "environment": environment(),
               "parent": {"path": str(path.relative_to(root)), "sha256": file_hash(path)}}
    output.mkdir(parents=True)
    atomic_save(output / "pruned.pt", payload)
    atomic_json(output / "pruning.json", report)
    return {"checkpoint": str((output / "pruned.pt").relative_to(root)), "report": report}
