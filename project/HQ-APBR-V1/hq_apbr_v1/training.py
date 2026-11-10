"""Independent two-pass HQ-SAM runtime; calibration is committed at Epoch boundaries.

The geometry is immutable during an Epoch and there is no APBR energy-feedback
update here. This module never creates a validation or Test dataset.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .math_core import hq_global_l2_perturbation


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_sha(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def freeze_bn_running_stats(model):
    modules = [(module, module.track_running_stats) for module in model.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in modules:
            module.track_running_stats = False
        yield
    finally:
        for module, value in modules:
            module.track_running_stats = value


def _finite_tensors(values, label: str) -> None:
    checks = [torch.isfinite(value).all() for value in values]
    if checks and not bool(torch.stack(checks).all()):
        raise FloatingPointError(f"Nonfinite {label}")


class HQStepper:
    def __init__(self, model: nn.Module, geometry: Mapping[str, torch.Tensor], *,
                 rho: float = .05, lr: float = .1, momentum: float = .9,
                 weight_decay: float = .0005, radius_rtol: float = 2e-5):
        self.model = model
        self.params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not self.params:
            raise ValueError("No trainable parameters")
        self.device = next(iter(self.params.values())).device
        if any(p.device != self.device or p.dtype not in {torch.float32, torch.float64}
               for p in self.params.values()):
            raise ValueError("HQ supports one device and FP32/FP64 parameters")
        if not 0 < rho < float("inf"):
            raise ValueError("rho must be finite and positive")
        if not np.isfinite(radius_rtol) or radius_rtol < 0:
            raise ValueError("Invalid perturbation radius tolerance")
        self.rho, self.radius_rtol = float(rho), float(radius_rtol)
        self.optimizer = torch.optim.SGD(self.params.values(), lr=lr, momentum=momentum,
                                         weight_decay=weight_decay, nesterov=False)
        self.steps, self.active, self.failed = 0, False, False
        self.set_geometry(geometry)

    def set_geometry(self, geometry: Mapping[str, torch.Tensor]) -> None:
        if self.active or self.failed:
            raise RuntimeError("Cannot change geometry of an active/failed stepper")
        if not geometry or not geometry.keys() <= self.params.keys():
            raise ValueError("Invalid geometry parameter mapping")
        self.geometry = {}
        for name, value in geometry.items():
            if value.shape != self.params[name].shape or value.dtype != torch.float64:
                raise ValueError(f"Invalid FP64 geometry mapping: {name}")
            self.geometry[name] = value.detach().to(self.device).clone()
        _finite_tensors(self.geometry.values(), "geometry")
        total = torch.stack([value.sum() for value in self.geometry.values()]).sum()
        absolute = torch.stack([value.abs().sum() for value in self.geometry.values()]).sum()
        if float(total.abs()) > 1e-12 * max(1., float(absolute)):
            raise ValueError("Geometry violates eligible global zero-sum constraint")
        self.geometry_sha256 = tensor_sha(self.geometry)
        # Include noneligible s=0 in the common scale. This cached algebra is
        # equivalent to exp(2s)g / ||exp(2s)g|| and avoids per-batch log/exp.
        extrema = torch.stack([value.min() for value in self.geometry.values()] +
                              [value.max() for value in self.geometry.values()] +
                              [torch.zeros((), dtype=torch.float64, device=self.device)])
        low, high = float(extrema.min()), float(extrema.max())
        self.cached_scales = None
        if 2 * (high - low) <= 600:
            self.cached_scales = {name: torch.exp(2 * (value - high))
                                  for name, value in self.geometry.items()}
            self.noneligible_scale = torch.exp(torch.tensor(-2 * high, dtype=torch.float64, device=self.device))

    def perturbations(self):
        gradients = {name: p.grad for name, p in self.params.items()}
        if self.cached_scales is None:
            return hq_global_l2_perturbation(self.params, gradients, self.geometry,
                                            self.rho, radius_rtol=self.radius_rtol)
        direction = {
            name: (torch.zeros_like(p, dtype=torch.float64) if p.grad is None else
                   p.grad.detach().double() * self.cached_scales.get(name, self.noneligible_scale))
            for name, p in self.params.items()
        }
        _finite_tensors(direction.values(), "scaled direction")
        norm = torch.stack([value.square().sum() for value in direction.values()]).sum().sqrt()
        if float(norm) == 0:
            if any(p.grad is not None and bool((p.grad != 0).any()) for p in self.params.values()):
                raise FloatingPointError("Nonzero gradients underflowed in cached direction")
            return {name: torch.zeros_like(p) for name, p in self.params.items()}, 0., True
        epsilon = {name: (self.rho * value / norm).to(self.params[name].dtype)
                   for name, value in direction.items()}
        actual = float(torch.stack([value.double().square().sum() for value in epsilon.values()]).sum().sqrt())
        if not np.isfinite(actual) or abs(actual / self.rho - 1) > self.radius_rtol:
            raise FloatingPointError(f"Incorrect global perturbation radius: {actual}")
        return epsilon, actual, False

    def _gradient_norm(self, stage):
        gradients = [p.grad for p in self.params.values() if p.grad is not None]
        _finite_tensors(gradients, stage)
        return float(torch.stack([grad.double().square().sum() for grad in gradients]).sum().sqrt()) if gradients else 0.

    def step(self, inputs, targets, *, audit: bool = False) -> dict:
        if self.active or self.failed:
            raise RuntimeError("Stepper is active/failed; reload a valid checkpoint")
        self.active = True
        saved = None
        bn_before = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        audit_data = {}
        try:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            center = nn.functional.cross_entropy(logits, targets)
            _finite_tensors([center, logits], "first forward")
            center.backward()
            g1_norm = self._gradient_norm("first gradient")
            epsilon, eps_norm, zero = self.perturbations()
            if audit:
                audit_data["first_gradients"] = {name: None if p.grad is None else p.grad.detach().clone()
                                                  for name, p in self.params.items()}
                audit_data["epsilon"] = {name: value.detach().clone() for name, value in epsilon.items()}
                audit_data["bn_after_center"] = {name: value.detach().clone() for name, value in self.model.named_buffers()}
            saved = {name: p.detach().clone() for name, p in self.params.items()}
            with torch.no_grad():
                for name, p in self.params.items():
                    p.add_(epsilon[name])
            _finite_tensors(self.params.values(), "perturbed parameters")
            physical_norm = float(torch.stack([(p.detach().double() - saved[name].double()).square().sum()
                                               for name, p in self.params.items()]).sum().sqrt())
            self.optimizer.zero_grad(set_to_none=True)
            with freeze_bn_running_stats(self.model):
                adversarial_logits = self.model(inputs)
                adversarial = nn.functional.cross_entropy(adversarial_logits, targets)
                _finite_tensors([adversarial, adversarial_logits], "adversarial forward")
                adversarial.backward()
            g2_norm = self._gradient_norm("second gradient")
            with torch.no_grad():
                for name, p in self.params.items():
                    p.copy_(saved[name])
            if audit:
                audit_data["restored_parameters"] = {name: p.detach().clone() for name, p in self.params.items()}
                audit_data["optimizer_gradients"] = {name: None if p.grad is None else p.grad.detach().clone()
                                                      for name, p in self.params.items()}
            saved = None
            self.optimizer.step()
            _finite_tensors(self.params.values(), "updated parameters")
            _finite_tensors((value for state in self.optimizer.state.values() for value in state.values()
                             if torch.is_tensor(value)), "optimizer state")
            self.steps += 1
            energies = {name: value.double().square().sum() for name, value in epsilon.items()}
            energy_names = list(self.geometry) + ["noneligible"]
            noneligible = sum((energy for name, energy in energies.items() if name not in self.geometry),
                              torch.zeros((), dtype=torch.float64, device=self.device))
            energy_values = torch.stack([energies[name] for name in self.geometry] + [noneligible]).cpu().tolist()
            info = {"center_loss": float(center.detach()), "adversarial_loss": float(adversarial.detach()),
                    "count": len(targets), "correct": int((logits.argmax(1) == targets).sum()),
                    "g1_norm": g1_norm, "g2_norm": g2_norm, "eps_norm": eps_norm,
                    "physical_addition_norm": physical_norm, "zero_perturbation": zero,
                    "perturbation_energy_distribution": {name: value / self.rho**2 for name, value in zip(energy_names, energy_values)},
                    "direction_path": "cached_fp64_scale" if self.cached_scales is not None else "log_reference"}
            if audit:
                info["audit"] = audit_data
            return info
        except BaseException:
            if saved is not None:
                with torch.no_grad():
                    for name, p in self.params.items():
                        p.copy_(saved[name])
            with torch.no_grad():
                for name, value in self.model.named_buffers():
                    value.copy_(bn_before[name])
            self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.active = False

    def state_dict(self):
        if self.active or self.failed:
            raise RuntimeError("Only a healthy completed step may be checkpointed")
        return {"version": 1, "steps": self.steps, "rho": self.rho,
                "radius_rtol": self.radius_rtol, "parameter_mapping": [(name, tuple(p.shape)) for name, p in self.params.items()],
                "optimizer": cpu_copy(self.optimizer.state_dict()), "geometry": cpu_copy(self.geometry),
                "geometry_sha256": self.geometry_sha256}

    def load_state_dict(self, state):
        if self.active:
            raise RuntimeError("Cannot load during an active step")
        if state["version"] != 1 or state["rho"] != self.rho or state["radius_rtol"] != self.radius_rtol:
            raise ValueError("Stepper configuration mismatch")
        if state["parameter_mapping"] != [(name, tuple(p.shape)) for name, p in self.params.items()]:
            raise ValueError("Stepper parameter mapping mismatch")
        if tensor_sha(state["geometry"]) != state["geometry_sha256"]:
            raise ValueError("Geometry SHA256 mismatch")
        self.failed = False
        self.set_geometry(state["geometry"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state["steps"])


def train_epoch(stepper: HQStepper, loader, device, epoch: int, *, expected_samples=50_000,
                expected_batches=391, progress_callback=None) -> dict:
    """One Epoch only. Caller refreshes geometry and steps scheduler afterward."""
    synchronize(device)
    start = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    count = correct = batches = zero_steps = 0
    loss_sum = adversarial_sum = 0.
    eps_min, eps_max = float("inf"), 0.
    order = hashlib.sha256()
    energy_sum = {}
    for batch in loader:
        inputs, targets = batch[:2]
        if len(batch) > 2:
            order.update(torch.as_tensor(batch[2]).numpy().astype("<i8").tobytes())
        info = stepper.step(inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True))
        count += info["count"]
        correct += info["correct"]
        loss_sum += info["center_loss"] * info["count"]
        adversarial_sum += info["adversarial_loss"] * info["count"]
        batches += 1
        zero_steps += int(info["zero_perturbation"])
        eps_min, eps_max = min(eps_min, info["eps_norm"]), max(eps_max, info["eps_norm"])
        for name, value in info["perturbation_energy_distribution"].items():
            energy_sum[name] = energy_sum.get(name, 0.) + value
        if progress_callback is not None:
            progress_callback(batches, count, info)
    synchronize(device)
    if count != expected_samples or batches != expected_batches:
        raise ValueError(f"Incomplete Epoch coverage: {count} samples / {batches} batches")
    result = {"epoch": epoch, "completed_epochs": epoch + 1, "steps": stepper.steps,
              "train_samples": count, "batches": batches, "train_order_sha256": order.hexdigest(),
              "train_loss": loss_sum / count, "adversarial_train_loss": adversarial_sum / count,
              "train_accuracy": correct / count, "train_seconds": time.perf_counter() - start,
              "learning_rate": stepper.optimizer.param_groups[0]["lr"],
              "eps_norm_min": eps_min, "eps_norm_max": eps_max, "zero_steps": zero_steps,
              "current_geometry_hash": stepper.geometry_sha256,
              "perturbation_energy_distribution": {name: value / batches for name, value in energy_sum.items()},
              "test_accessed": False}
    if torch.device(device).type == "cuda":
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def build_checkpoint(model, stepper, scheduler, *, epoch: int, history: list, identity: dict,
                     calibration_ids: list, q_diags: dict, calibration_report: dict,
                     skipped_calibration_epochs=()) -> dict:
    if epoch < 0 or epoch >= 200:
        raise ValueError("Epoch index must be in 0..199")
    model_state = cpu_copy(model.state_dict())
    return {"schema": "hq_apbr_v1_training_v1", "status": "healthy", "seed": 0,
            "epoch": epoch, "epoch_index": epoch, "completed_epochs": epoch + 1, "next_epoch": epoch + 1,
            "steps": stepper.steps, "model": model_state, "model_sha256": tensor_sha(model_state),
            "stepper": stepper.state_dict(), "scheduler": cpu_copy(scheduler.state_dict()),
            "history": history, "identity": identity, "rng": capture_rng(),
            "calibration_ids": list(calibration_ids), "q_diags": cpu_copy(q_diags),
            "q_diags_sha256": tensor_sha(q_diags), "calibration_report": calibration_report,
            "skipped_calibration_epochs": list(skipped_calibration_epochs),
            "test_accessed": False, "parameter_location": "unperturbed_after_sgd"}


def validate_checkpoint(payload: dict, identity: dict | None = None) -> None:
    if payload["schema"] != "hq_apbr_v1_training_v1" or payload["status"] != "healthy":
        raise ValueError("Checkpoint schema or status mismatch")
    if payload["test_accessed"] is not False or payload["parameter_location"] != "unperturbed_after_sgd":
        raise ValueError("Invalid training checkpoint provenance")
    if payload["completed_epochs"] != payload["epoch"] + 1 or payload["next_epoch"] != payload["completed_epochs"]:
        raise ValueError("Checkpoint Epoch accounting mismatch")
    if identity is not None and payload["identity"] != identity:
        raise ValueError("Checkpoint source/config identity mismatch")
    for key, hash_key in (("model", "model_sha256"), ("q_diags", "q_diags_sha256")):
        if tensor_sha(payload[key]) != payload[hash_key]:
            raise ValueError(f"Checkpoint {key} hash mismatch")
        _finite_tensors(payload[key].values(), f"checkpoint {key}")
    if tensor_sha(payload["stepper"]["geometry"]) != payload["stepper"]["geometry_sha256"]:
        raise ValueError("Checkpoint geometry hash mismatch")
    if payload["steps"] != payload["stepper"]["steps"]:
        raise ValueError("Checkpoint optimizer step mismatch")


def atomic_checkpoint(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    validate_checkpoint(payload)
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    validate_checkpoint(torch.load(temp, map_location="cpu", weights_only=False), payload["identity"])
    temp.replace(path)
    return file_sha(path)


def load_checkpoint(path, model, stepper, scheduler, *, identity=None, restore_random_state=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, identity)
    model.load_state_dict(payload["model"], strict=True)
    stepper.load_state_dict(payload["stepper"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_random_state:
        restore_rng(payload["rng"])
    return payload
