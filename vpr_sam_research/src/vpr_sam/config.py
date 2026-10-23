"""Strict configuration: unresolved formal protocols cannot execute."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

METHODS = {"sgd", "sam", "vpr", "static", "positive", "fixed_l2"}


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(c, *, formal_allowed=False):
    expected = {"schema", "purpose", "seed", "device", "threads", "deterministic", "output", "model",
                "data", "optimizer", "geometry", "training", "pruning", "recovery", "evaluation"}
    if set(c) != expected:
        raise ValueError(f"Config keys mismatch: missing={expected-set(c)}, extra={set(c)-expected}")
    if c["schema"] != 1 or c["purpose"] not in {"smoke", "formal"}:
        raise ValueError("Unsupported schema/purpose")
    for key, keys in {
        "data": {"kind", "root", "download", "validation_size", "synthetic_train", "synthetic_test", "augment", "cutout", "workers"},
        "optimizer": {"method", "lr", "momentum", "weight_decay", "nesterov", "rho"},
        "geometry": {"gamma", "dtype", "warmup_steps", "volume_atol", "volume_rtol", "static_std"},
        "training": {"epochs", "batch_size", "scheduler", "eta_min", "checkpoint_interval_steps"},
        "pruning": {"n", "m", "scope", "score", "tail"},
        "recovery": {"epochs", "method", "lr", "momentum", "weight_decay", "rho", "scheduler", "eta_min"},
        "evaluation": {"batch_size", "probe_rho", "probe_batches"},
    }.items():
        if set(c[key]) != keys:
            raise ValueError(f"Unexpected/missing fields in {key}")
    formal = c["purpose"] == "formal"
    if formal and not formal_allowed:
        raise ValueError("Formal execution requires explicit --allow-formal after protocol approval")
    def required(value, label):
        if value is None:
            raise ValueError(f"Unresolved protocol field: {label}")
    for key in ("seed", "device", "threads", "deterministic", "output", "model"):
        required(c[key], key)
    for section in ("optimizer", "geometry", "training", "evaluation"):
        for key, value in c[section].items():
            required(value, f"{section}.{key}")
    if c["optimizer"]["method"] not in METHODS:
        raise ValueError("Unknown method")
    if c["model"] not in {"resnet18", "tiny"}:
        raise ValueError("Unknown model")
    if c["data"]["kind"] not in {"cifar10", "synthetic"}:
        raise ValueError("Unknown dataset")
    if formal and (c["model"] != "resnet18" or c["data"]["kind"] != "cifar10"):
        raise ValueError("Formal runs require ResNet18 and real CIFAR-10")
    if formal and not c["deterministic"]:
        raise ValueError("Formal protocol requires deterministic execution")
    for key in ("validation_size", "augment", "cutout", "workers", "root", "download"):
        required(c["data"][key], f"data.{key}")
    if c["training"]["scheduler"] != "cosine" or c["training"]["epochs"] <= 0:
        raise ValueError("Training requires positive epochs and cosine scheduler")
    for value in (c["training"]["batch_size"], c["evaluation"]["batch_size"], c["threads"]):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Batch sizes and threads must be positive integers")
    if c["data"]["workers"] < 0 or c["data"]["cutout"] < 0:
        raise ValueError("workers/cutout must be nonnegative")
    if not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if not isinstance(c["data"]["augment"], bool) or not isinstance(c["data"]["download"], bool):
        raise ValueError("augment/download must be explicit booleans")
    if not 0 <= c["data"]["cutout"] <= 32:
        raise ValueError("cutout must be in [0,32]")
    for value in (c["training"]["epochs"], c["training"]["checkpoint_interval_steps"],
                  c["data"]["workers"], c["geometry"]["warmup_steps"], c["evaluation"]["probe_batches"]):
        if not isinstance(value, int):
            raise ValueError("Epoch/step/worker budgets must be integers")
    if c["training"]["checkpoint_interval_steps"] < 0 or c["evaluation"]["probe_batches"] < 0:
        raise ValueError("Invalid interval/probe budget")
    o, g = c["optimizer"], c["geometry"]
    for label, value in {"lr": o["lr"], "rho": o["rho"], "gamma": g["gamma"],
                         "momentum": o["momentum"], "weight_decay": o["weight_decay"],
                         "eta_min": c["training"]["eta_min"], "probe_rho": c["evaluation"]["probe_rho"],
                         "volume_atol": g["volume_atol"], "volume_rtol": g["volume_rtol"],
                         "static_std": g["static_std"]}.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid finite nonnegative {label}")
    if o["lr"] <= 0 or o["rho"] <= 0 or not 0 <= o["momentum"] < 1:
        raise ValueError("Require lr,rho>0 and 0<=momentum<1")
    if c["training"]["eta_min"] > o["lr"] or (o["nesterov"] and o["momentum"] <= 0):
        raise ValueError("Invalid eta_min or Nesterov momentum")
    if not 0 <= g["gamma"] < .5:
        raise ValueError("Require 0<=gamma<1/2 (local criterion, not a global guarantee)")
    if o["method"] in {"vpr", "positive", "fixed_l2"} and g["gamma"] == 0 and formal:
        raise ValueError("Dynamic formal method requires gamma>0")
    if g["dtype"] not in {"float32", "float64"} or g["warmup_steps"] < 0:
        raise ValueError("Invalid geometry precision/warmup")
    if o["method"] == "static" and g["static_std"] <= 0:
        raise ValueError("Static ablation requires explicit positive static_std")
    if formal:
        # The user requires a complete workflow protocol before formal training.
        for section in ("pruning", "recovery"):
            for key, value in c[section].items():
                required(value, f"{section}.{key}")
        validate_recovery(c["recovery"])
    if formal or c["pruning"]["n"] is not None or c["pruning"]["m"] is not None:
        validate_pruning(c["pruning"])
    return c


def validate_pruning(p):
    n, m = p["n"], p["m"]
    if not isinstance(n, int) or not isinstance(m, int) or not 0 < n < m:
        raise ValueError("Specify integers 0<N<M; no default N:M")
    if p["scope"] not in {"all", "exclude_first", "exclude_first_last", "exclude_first_and_pointwise", "residual_and_head"}:
        raise ValueError("Explicit pruning scope required")
    if p["score"] != "magnitude" or p["tail"] not in {"dense", "error"}:
        raise ValueError("First version uses magnitude; tail must be dense/error")


def canonical_pruning(p):
    validate_pruning(p)
    result = copy.deepcopy(p)
    if result["scope"] == "residual_and_head":
        result["scope"] = "exclude_first_and_pointwise"
    return result


def validate_recovery(r):
    if any(v is None for v in r.values()) or r["method"] not in {"sgd", "sam"}:
        raise ValueError("Explicit positive recovery budget and common sgd/sam protocol required")
    if type(r["epochs"]) is not int or r["epochs"] <= 0 or r["scheduler"] != "cosine":
        raise ValueError("Recovery requires positive integer epochs and cosine scheduler")
    for key in ("lr", "momentum", "weight_decay", "rho", "eta_min"):
        if type(r[key]) not in (int, float) or not math.isfinite(r[key]) or r[key] < 0:
            raise ValueError(f"Invalid recovery.{key}")
    if r["lr"] <= 0 or r["rho"] <= 0 or r["momentum"] >= 1 or r["eta_min"] > r["lr"]:
        raise ValueError("Invalid recovery optimizer/scheduler range")


def recovery_config(c):
    r = c["recovery"]
    validate_recovery(r)
    out = copy.deepcopy(c)
    out["optimizer"].update({k: r[k] for k in ("method", "lr", "momentum", "weight_decay", "rho")})
    out["training"].update({k: r[k] for k in ("epochs", "scheduler", "eta_min")})
    return validate(out, formal_allowed=True)


def comparison_protocol(c):
    """Exclude only research variables and paths, never fairness variables."""
    result = copy.deepcopy(c)
    for key in ("output", "device", "threads"):
        result.pop(key)
    result["data"].pop("root")
    result["optimizer"].pop("method")
    if result["pruning"]["n"] is not None and result["pruning"]["m"] is not None:
        result["pruning"] = canonical_pruning(result["pruning"])
    result["geometry"] = {}  # geometry is the declared research variable
    return result


def method_protocol(c):
    """Within one method, seed changes must not hide hyperparameter changes."""
    result = comparison_protocol(c)
    result.pop("seed")
    result["geometry"] = copy.deepcopy(c["geometry"])
    return result
