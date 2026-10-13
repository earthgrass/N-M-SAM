"""Trusted local full-state checkpoints, validated config/split on resume."""
from pathlib import Path
from copy import deepcopy
import os
import torch
from nmsam.utils.seed import capture_rng,restore_rng


def save_checkpoint(path,trainer,sampler,metadata):
    required={"config_sha256","split_sha256","git_sha"}
    if not required.issubset(metadata):raise ValueError("checkpoint provenance incomplete")
    state={"format_version":2,"model":trainer.model.state_dict(),"optimizer":trainer.optimizer.state_dict(),
        "scheduler":trainer.scheduler.state_dict(),"mask":trainer.registry.state_dict() if trainer.registry else None,
        "method":trainer.method,"step":trainer.step,"rng":capture_rng(),
        "sampler":sampler.state_dict(),"metadata":metadata,
        "modes":{n:m.training for n,m in trainer.model.named_modules()},
        "grads":[None if p.grad is None else p.grad.detach() for p in trainer.model.parameters()]}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    try:
        torch.save(state,temporary);os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path,trainer,sampler,expected):
    # Only load self-produced trusted files. torch pickle is not an untrusted format.
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("format_version")!=2:raise ValueError("unsupported checkpoint format")
    for key in ("config_sha256","split_sha256"):
        if state["metadata"].get(key)!=expected.get(key):raise ValueError(f"resume {key} mismatch")
    for key in ("archive_sha256","git_sha"):
        if key in expected and state["metadata"].get(key)!=expected[key]:raise ValueError(f"resume {key} mismatch")
    if state["method"]!=trainer.method:raise ValueError("resume method mismatch")
    backup={"model":deepcopy(trainer.model.state_dict()),"optimizer":deepcopy(trainer.optimizer.state_dict()),
            "scheduler":deepcopy(trainer.scheduler.state_dict()),"mask":trainer.registry.state_dict() if trainer.registry else None,
            "step":trainer.step,"sampler":deepcopy(sampler.state_dict()),"rng":capture_rng(),
            "grads":[None if p.grad is None else p.grad.clone() for p in trainer.model.parameters()],
            "modes":{n:m.training for n,m in trainer.model.named_modules()}}
    try:
        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        if trainer.registry:
            parameters=dict(trainer.model.named_parameters())
            state["mask"]["masks"]={k:v.to(parameters[k].device) for k,v in state["mask"]["masks"].items()}
            trainer.registry.load_state_dict(state["mask"])
        trainer.step=state["step"];sampler.load_state_dict(state["sampler"])
        if len(state["grads"])!=len(tuple(trainer.model.parameters())):raise ValueError("gradient layout mismatch")
        for p,g in zip(trainer.model.parameters(),state["grads"]):p.grad=None if g is None else g.to(p)
        for n,m in trainer.model.named_modules():m.training=state["modes"][n]
        restore_rng(state["rng"])
    except BaseException:
        trainer.model.load_state_dict(backup["model"]);trainer.optimizer.load_state_dict(backup["optimizer"])
        trainer.scheduler.load_state_dict(backup["scheduler"])
        if trainer.registry:trainer.registry.load_state_dict(backup["mask"])
        trainer.step=backup["step"];sampler.load_state_dict(backup["sampler"])
        for p,g in zip(trainer.model.parameters(),backup["grads"]):p.grad=g
        for n,m in trainer.model.named_modules():m.training=backup["modes"][n]
        restore_rng(backup["rng"]);raise
    return state["metadata"]
