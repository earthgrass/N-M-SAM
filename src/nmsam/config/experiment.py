"""GPU deployment configuration validation; never opens datasets or devices."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import math
import os
import yaml
from nmsam.experiment.training import METHODS


@dataclass(frozen=True)
class ExperimentConfig:
    kind: str = "baseline"
    model: str = "resnet18"
    device: str = "cuda"
    precision: str = "fp32"
    amp: bool = False
    ddp: bool = False
    seeds: list[int] = field(default_factory=lambda:[0,1,2])
    split_seed: int = 0
    methods: list[str] = field(default_factory=lambda:list(METHODS))
    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 0
    lr: float = .1
    momentum: float = .9
    nesterov: bool = True
    rho: float = .05
    l2_reg: float = .001
    gradient_clipping: float = 5.
    scheduler: str = "cosine"
    mask_interval: int = 20
    ste: bool = True
    max_batches: int | None = None
    eval_max_batches: int | None = None
    archive_path: str = "${NMSAM_ARCHIVE}"
    dataset_cache_dir: str = "${NMSAM_CACHE}"
    output_dir: str = "results"
    checkpoint: str = "${NMSAM_CHECKPOINT}"
    resume: str | None = None
    state_mode: str = "eval_fixed_buffers"
    target_layer: str = "layer1.0.conv1.weight"
    block_ids: list[int] = field(default_factory=lambda:[0])
    inner_steps: int = 1
    outer_steps: int = 1
    restarts: int = 1
    compensation_budget: float = .1
    max_hvps: int = 4
    damping: float = .001
    joint_candidate_limit: int = 6
    perturbation_domain: str = "new_support"

    def __post_init__(self):
        if self.kind not in ("baseline","direction_gap","objective_gap","compensation"):
            raise ValueError("invalid experiment kind")
        if self.model not in ("resnet18","tiny") or self.device not in ("cuda","cpu") or self.precision not in ("fp32","fp64"):
            raise ValueError("invalid model/device/precision")
        if self.scheduler not in ("constant","cosine") or self.state_mode not in ("eval_fixed_buffers","train_fixed_batch"):
            raise ValueError("invalid scheduler/probe mode")
        if self.perturbation_domain not in ("new_support","old_support_control"):
            raise ValueError("explicit perturbation domain required")
        if not self.seeds or len(set(self.seeds))!=len(self.seeds) or any(type(s) is not int or not 0<=s<2**32 for s in self.seeds):
            raise ValueError("unique nonnegative 32-bit seeds required")
        if not self.methods or len(set(self.methods))!=len(self.methods) or any(m not in METHODS for m in self.methods):
            raise ValueError("invalid methods")
        for key in ("amp","ddp","nesterov","ste"):
            if type(getattr(self,key)) is not bool:raise ValueError(f"{key} must be boolean")
        for key in ("epochs","batch_size","mask_interval","inner_steps","outer_steps","restarts","max_hvps","joint_candidate_limit"):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f"invalid {key}")
        if self.batch_size<2 or self.num_workers!=0:raise ValueError("exact-resume protocol requires batch>=2, num_workers=0")
        if type(self.split_seed) is not int or not 0<=self.split_seed<2**32:raise ValueError("invalid split seed")
        for key in ("archive_path","dataset_cache_dir","output_dir","checkpoint","target_layer"):
            if not isinstance(getattr(self,key),str) or not getattr(self,key):raise ValueError(f"invalid {key}")
        for key in ("max_batches","eval_max_batches"):
            value=getattr(self,key)
            if value is not None and (type(value) is not int or value<1):raise ValueError(f"invalid {key}")
        for key in ("lr","momentum","rho","l2_reg","gradient_clipping","compensation_budget","damping"):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError(f"invalid {key}")
        if self.lr==0 or self.momentum>=1 or (self.nesterov and self.momentum==0):raise ValueError("invalid SGD hyperparameters")
        if not self.block_ids or len(set(self.block_ids))!=len(self.block_ids) or any(type(b) is not int or b<0 for b in self.block_ids):
            raise ValueError("unique nonnegative block ids required")
        if self.kind=="compensation" and len(self.block_ids)!=2:raise ValueError("C requires exactly two blocks")


def load_experiment_config(path):
    raw=Path(path).read_text(encoding="utf-8");value=yaml.safe_load(raw)
    if not isinstance(value,dict):raise ValueError("configuration mapping required")
    return ExperimentConfig(**value),raw


def resolved_path(value):
    result=os.path.expandvars(value)
    if "${" in result or not result:raise ValueError("unresolved deployment path variable")
    return Path(result).expanduser()


def plan(config):
    updates=(45000//config.batch_size)*config.epochs
    if config.max_batches is not None:updates=min(updates,config.max_batches)
    matrix=([{"seed":s,"method":m} for s in config.seeds for m in config.methods]
            if config.kind=="baseline" else [{"seed":s,"probe":config.kind} for s in config.seeds])
    return {"dry_run":True,"config":asdict(config),"status":"GPU_VALIDATION_PENDING",
        "dataset_status":"DATASET_PENDING","matrix":matrix,
        "estimated_updates_per_run":updates if config.kind=="baseline" else 0,
        "amp_status":"UNSUPPORTED_FAIL_CLOSED","ddp_status":"UNSUPPORTED_FAIL_CLOSED",
        "execution_requires":"explicit --execute; separate human resource authorization"}
