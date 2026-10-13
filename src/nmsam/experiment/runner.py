"""Bounded, no-network CPU FakeData smoke; not a scientific accuracy benchmark."""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import hashlib
import json
import platform
import subprocess
from uuid import uuid4
import torch
import torchvision
from nmsam import __version__
from nmsam.config.loader import load_dense_config
from nmsam.core.batch import StepBatch
from nmsam.core.loss import classification_loss
from nmsam.data.fake_data import fake_loader
from nmsam.models.tiny_mlp import TinyMLP
from nmsam.sam.reference_sgd import create_optimizer
from nmsam.sam.train_step import train_step
from nmsam.utils.seed import seed_everything
from .metrics import step_metrics


def run_dense_cpu(config_path: str | Path, output_dir: str | Path = "results") -> dict:
    config, raw = load_dense_config(config_path)
    if config.steps > 100:
        raise ValueError("smoke runner is limited to 100 steps")
    seed_everything(config.seed)
    model = TinyMLP().cpu()
    optimizer = create_optimizer(model.parameters(), lr=config.lr,
                                 beta=config.momentum, nesterov=config.nesterov)
    loss = partial(classification_loss, l2_reg=config.l2_reg,
                   no_weight_decay_on_bn=config.no_weight_decay_on_bn)
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    run_id = uuid4().hex
    record = {
        "run_id": run_id, "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "config_raw": raw, "config_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "config": asdict(config), "seed": config.seed, "code_version": __version__,
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": "cpu", "dtype": "float32",
        "model": "TinyMLP(16,8,3; BatchNorm1d; Tanh)",
        "dataset": "torchvision.datasets.FakeData", "split": "synthetic_train_smoke_only",
        "augmentation": "ToTensor only; same materialized batch for both forwards",
        "task_loss": "mean cross_entropy", "optimizer_loss": "CE + explicit lambda/2 sum p^2",
        "optimizer": "torch.SGD, dampening=0, weight_decay=0",
        "mask_scope": "dense (none)", "metrics": [],
    }
    for step, (inputs, targets) in enumerate(fake_loader(batch_size=config.batch_size,
            steps=config.steps, seed=config.seed), 1):
        report = train_step(model, optimizer, StepBatch(inputs, targets), loss,
                            rho=config.rho, gradient_clipping=config.gradient_clipping)
        record["metrics"].append(step_metrics(report, step))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"dense_{run_id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "result_path": str(path.resolve()),
                      "steps": len(record["metrics"]), "final": record["metrics"][-1]}, indent=2))
    return record

