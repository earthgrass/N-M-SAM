"""Isolated APBR-GPT joint training step.

The original APBR geometry object is supplied by the caller and is not
modified.  Weight optimisation uses adversarial CE plus the frozen-Hessian
pruning regularizer.  Geometry retains the original energy feedback and adds
only the pre-registered projected pruning-gradient update.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping

import torch
from torch import Tensor, nn

from baseline_adapters.apbr_gpt_joint import CandidateFactors, softmin_pruning_regularizer
from joint_optimization.calibration import combine_scale_updates
from joint_optimization.gradient_handoff import GradientHandoff, materialize_none_as_zeros
from joint_optimization.training_objective import differentiable_apbr_perturbations, joint_training_loss


@contextmanager
def freeze_bn(model: nn.Module):
    modules = [(m, m.track_running_stats) for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _value in modules:
            module.track_running_stats = False
        yield
    finally:
        for module, value in modules:
            module.track_running_stats = value


class JointStepper:
    """Transactional joint step around the frozen original APBR interfaces."""

    def __init__(self, model: nn.Module, geometry, optimizer: torch.optim.Optimizer, *,
                 eligible_parameter_names: tuple[str, ...], lambda_p: float, eta_s: float,
                 temperature: float = 0.1, rho: float = 0.05) -> None:
        self.model, self.geometry, self.optimizer = model, geometry, optimizer
        self.names = [name for name, _ in geometry.mapping]
        self.parameters = list(geometry.params)
        if self.names != [name for name, _ in model.named_parameters()]:
            raise ValueError("Geometry/model parameter mapping mismatch")
        self.eligible = tuple(eligible_parameter_names)
        if not set(self.eligible) <= set(self.names):
            raise ValueError("Unknown eligible parameter")
        if lambda_p < 0 or eta_s < 0 or temperature <= 0 or rho <= 0:
            raise ValueError("Invalid frozen joint constants")
        self.lambda_p, self.eta_s = float(lambda_p), float(eta_s)
        self.temperature, self.rho = float(temperature), float(rho)
        self.factors: dict[str, CandidateFactors] = {}
        self.references: dict[str, float] = {}
        self.handoff = GradientHandoff(self.parameters)
        self.steps, self.failed, self.active = 0, False, False

    def refresh(self, factors: Mapping[str, CandidateFactors], references: Mapping[str, float]) -> None:
        if self.active or self.failed:
            raise RuntimeError("Cannot refresh active/failed stepper")
        if set(factors) != set(self.eligible) or set(references) != set(self.eligible):
            raise ValueError("Incomplete Hessian/reference cache")
        for name in self.eligible:
            item = factors[name]
            parameter = dict(zip(self.names, self.parameters))[name]
            columns = parameter[0].numel()
            if item.inverse_qdd.shape != (columns // 4, 6, 2, 2):
                raise ValueError(f"Factor coordinate mismatch: {name}")
            if not 0 < float(references[name]) < float("inf"):
                raise ValueError(f"Invalid reference cost: {name}")
        self.factors = dict(factors); self.references = {k: float(v) for k, v in references.items()}

    def _regularizer(self, scales: list[Tensor], first_grads: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        if set(self.factors) != set(self.eligible):
            raise RuntimeError("Hessian cache has not been refreshed")
        probes = differentiable_apbr_perturbations(first_grads, scales, rho=self.rho)
        by_name = dict(zip(self.names, zip(self.parameters, probes)))
        weighted, count = None, 0
        for name in self.eligible:
            parameter, probe = by_name[name]
            rows = parameter.shape[0]; groups = parameter[0].numel() // 4
            value = softmin_pruning_regularizer(
                parameter.reshape(rows, -1), probe.reshape(rows, -1),
                self.factors[name].inverse_qdd, self.references[name],
                temperature=self.temperature,
            )
            blocks = rows * groups
            weighted = value * blocks if weighted is None else weighted + value * blocks
            count += blocks
        return weighted / count, probes

    def step(self, inputs: Tensor, targets: Tensor, *, pruning_active: bool) -> dict:
        if self.active or self.failed:
            raise RuntimeError("Stepper failed or active; reload a completed checkpoint")
        self.active = True; saved = None
        buffers = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        try:
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            center_logits = self.model(inputs)
            center = nn.functional.cross_entropy(center_logits, targets)
            first_raw = torch.autograd.grad(center, self.parameters, allow_unused=True)
            first = materialize_none_as_zeros(self.parameters, first_raw)
            first_grad_norm = float(sum(gradient.double().square().sum() for gradient in first).sqrt())
            # Legacy APBR reads p.grad.  Install the functional first gradients
            # only for perturbation generation and energy-feedback preparation.
            with self.handoff.install(first_raw):
                original_eps, stats = self.geometry.perturbations(self.rho)
                self.geometry.prepare(original_eps, self.steps, stats["zero_perturbation"])

            saved = [p.detach().clone() for p in self.parameters]
            with torch.no_grad():
                for parameter, epsilon in zip(self.parameters, original_eps):
                    parameter.add_(epsilon)
            # This is the same boundary as original Stepper.base.zero_grad:
            # adversarial gradients start from a clean destination.
            self.optimizer.zero_grad(set_to_none=True)
            with freeze_bn(self.model):
                adversarial_logits = self.model(inputs)
                adversarial_ce = nn.functional.cross_entropy(adversarial_logits, targets)
            adversarial_ce.backward()
            with torch.no_grad():
                for parameter, before in zip(self.parameters, saved):
                    parameter.copy_(before)
                restoration_max_abs = max(float((parameter - before).abs().max())
                                          for parameter, before in zip(self.parameters, saved))
            saved = None
            adversarial_grad_norm = float(sum(
                (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).double().square().sum()
                for parameter in self.parameters
            ).sqrt())

            joint_active = pruning_active and self.lambda_p > 0
            scale_work = [scale.detach().clone().requires_grad_(joint_active)
                          for scale in self.geometry.logs]
            if joint_active:
                regularizer, _probes = self._regularizer(scale_work, first)
                loss = joint_training_loss(adversarial_ce, regularizer, self.lambda_p)
                joint_gradients = torch.autograd.grad(
                    regularizer, [*self.parameters, *scale_work], allow_unused=True
                )
                weight_grads = joint_gradients[:len(self.parameters)]
                scale_grads = joint_gradients[len(self.parameters):]
                for parameter, gradient in zip(self.parameters, weight_grads):
                    if gradient is not None:
                        if parameter.grad is None:
                            parameter.grad = self.lambda_p * gradient.detach()
                        else:
                            parameter.grad.add_(gradient.detach(), alpha=self.lambda_p)
                scale_grads = [torch.zeros_like(scale) if gradient is None else gradient
                               for scale, gradient in zip(scale_work, scale_grads)]
                pruning_weight_grad_norm = float(sum(
                    (torch.zeros_like(parameter) if gradient is None else gradient).double().square().sum()
                    for parameter, gradient in zip(self.parameters, weight_grads)
                ).sqrt())
            else:
                regularizer = adversarial_ce.detach() * 0
                loss = adversarial_ce
                scale_grads = [torch.zeros_like(scale) for scale in self.geometry.logs]
                pruning_weight_grad_norm = 0.0
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.parameters):
                raise FloatingPointError("Non-finite joint weight gradient")

            # Original APBR energy proposal was prepared from the installed g1
            # before perturbation.  Add only the separately calibrated pruning
            # scale update; no CE gradient is applied to s.
            candidate, zero = self.geometry.candidate
            energy = ([torch.zeros_like(s) for s in self.geometry.logs] if candidate is None
                      else [new.double() - old.double() for new, old in zip(candidate, self.geometry.logs)])
            energy_update_norm = float(sum(value.double().square().sum() for value in energy).sqrt())
            pruning_scale_update_norm = 0.0
            combined_scale_update_norm = energy_update_norm
            if joint_active and self.eta_s > 0:
                updated, _delta = combine_scale_updates(
                    self.geometry.logs, energy, scale_grads,
                    eta_s=self.eta_s, lambda_p=self.lambda_p,
                )
                baseline = ([old.double() for old in self.geometry.logs] if candidate is None
                            else [new.double() for new in candidate])
                pruning_scale_update_norm = float(sum(
                    (new.double() - base).square().sum() for new, base in zip(updated, baseline)
                ).sqrt())
                combined_scale_update_norm = float(sum(
                    (new.double() - old.double()).square().sum()
                    for new, old in zip(updated, self.geometry.logs)
                ).sqrt())
                self.geometry.candidate = ([value.to(old.dtype) for value, old in zip(updated, self.geometry.logs)], zero)

            final_optimizer_grad_norm = float(sum(
                (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).double().square().sum()
                for parameter in self.parameters
            ).sqrt())
            self.optimizer.step()
            self.geometry.commit(); self.steps += 1
            if not all(bool(torch.isfinite(p).all()) for p in self.parameters):
                raise FloatingPointError("Non-finite joint parameter")
            return {
                "center_loss": float(center.detach()), "adversarial_loss": float(adversarial_ce.detach()),
                "pruning_regularizer": float(regularizer.detach()), "joint_loss": float(loss.detach()),
                "pruning_active": bool(joint_active), "eps_norm": stats["eps_norm"],
                "ellipsoid_norm": stats["ellipsoid_norm"],
                "first_grad_norm": first_grad_norm,
                "adversarial_grad_norm": adversarial_grad_norm,
                "pruning_weight_grad_norm": pruning_weight_grad_norm,
                "final_optimizer_grad_norm": final_optimizer_grad_norm,
                "energy_update_norm": energy_update_norm,
                "pruning_scale_update_norm": pruning_scale_update_norm,
                "combined_scale_update_norm": combined_scale_update_norm,
                "restoration_max_abs": restoration_max_abs,
                "correct": int((center_logits.argmax(1) == targets).sum()), "count": len(targets),
            }
        except BaseException:
            if saved is not None:
                with torch.no_grad():
                    for parameter, before in zip(self.parameters, saved):
                        parameter.copy_(before)
            with torch.no_grad():
                for name, value in self.model.named_buffers():
                    value.copy_(buffers[name])
            self.geometry.discard(); self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True); self.active = False

    def state_dict(self) -> dict:
        if self.active or self.failed:
            raise RuntimeError("Only a healthy completed step can be checkpointed")
        return {"version": 2, "optimizer": self.optimizer.state_dict(),
                "geometry": self.geometry.state_dict(), "steps": self.steps,
                "eligible": self.eligible, "lambda_p": self.lambda_p, "eta_s": self.eta_s,
                "temperature": self.temperature, "rho": self.rho}

    def load_state_dict(self, state: Mapping) -> None:
        for key, expected in (("version", 2), ("eligible", self.eligible),
                              ("lambda_p", self.lambda_p), ("eta_s", self.eta_s),
                              ("temperature", self.temperature), ("rho", self.rho)):
            if state[key] != expected:
                raise ValueError(f"Joint stepper checkpoint mismatch: {key}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.geometry.load_state_dict(state["geometry"])
        self.steps, self.failed = int(state["steps"]), False
