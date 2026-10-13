"""Trusted local full-state checkpoints, validated config/split on resume."""
from pathlib import Path
import os
import torch
from nmsam.utils.seed import capture_rng,restore_rng


def save_checkpoint(path,trainer,sampler,metadata):
    required={"config_sha256","split_sha256","git_sha"}
    if not required.issubset(metadata):raise ValueError("checkpoint provenance incomplete")
    state={"model":trainer.model.state_dict(),"optimizer":trainer.optimizer.state_dict(),
        "scheduler":trainer.scheduler.state_dict(),"mask":trainer.registry.state_dict() if trainer.registry else None,
        "method":trainer.method,"step":trainer.step,"rng":capture_rng(),
        "sampler":sampler.state_dict(),"metadata":metadata}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp");torch.save(state,temporary);os.replace(temporary,path)


def load_checkpoint(path,trainer,sampler,expected):
    # Only load self-produced trusted files. torch pickle is not an untrusted format.
    state=torch.load(path,map_location="cpu",weights_only=False)
    for key in ("config_sha256","split_sha256"):
        if state["metadata"].get(key)!=expected.get(key):raise ValueError(f"resume {key} mismatch")
    if state["method"]!=trainer.method:raise ValueError("resume method mismatch")
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    if trainer.registry:trainer.registry.load_state_dict(state["mask"])
    trainer.step=state["step"];sampler.load_state_dict(state["sampler"]);restore_rng(state["rng"])
    return state["metadata"]
