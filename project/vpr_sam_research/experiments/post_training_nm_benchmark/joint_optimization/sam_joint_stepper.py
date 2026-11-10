"""Standard spherical SAM plus the frozen groupwise second-order regularizer."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from baseline_adapters.apbr_gpt_joint import CandidateFactors, softmin_pruning_regularizer
from joint_optimization.gradient_handoff import GradientHandoff, materialize_none_as_zeros
from joint_optimization.joint_stepper import freeze_bn
from joint_optimization.training_objective import joint_training_loss


class SAMJointStepper:
    """Transactional standard-SAM step with no learned APBR geometry."""

    def __init__(self, model: nn.Module, geometry, optimizer: torch.optim.Optimizer, *,
                 eligible_parameter_names: tuple[str, ...], lambda_p: float,
                 temperature: float = 0.1, rho: float = 0.05) -> None:
        if geometry.method != "sam" or geometry.logs:
            raise ValueError("SAMJointStepper requires spherical SAM geometry without learned scales")
        self.model, self.geometry, self.optimizer = model, geometry, optimizer
        self.names = [name for name, _shape in geometry.mapping]
        self.parameters = list(geometry.params)
        if self.names != [name for name, _parameter in model.named_parameters()]:
            raise ValueError("Geometry/model parameter mapping mismatch")
        self.eligible = tuple(eligible_parameter_names)
        if not set(self.eligible) <= set(self.names):
            raise ValueError("Unknown eligible parameter")
        if lambda_p < 0 or temperature <= 0 or rho <= 0:
            raise ValueError("Invalid frozen SAM-GPT constants")
        self.lambda_p, self.temperature, self.rho = float(lambda_p), float(temperature), float(rho)
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
            parameter = dict(zip(self.names, self.parameters))[name]
            columns = parameter[0].numel()
            if factors[name].inverse_qdd.shape != (columns // 4, 6, 2, 2):
                raise ValueError(f"Factor coordinate mismatch: {name}")
            if not 0 < float(references[name]) < float("inf"):
                raise ValueError(f"Invalid reference cost: {name}")
        self.factors = dict(factors); self.references = {name: float(value) for name, value in references.items()}

    def _regularizer(self, probes: list[Tensor]) -> Tensor:
        if set(self.factors) != set(self.eligible):
            raise RuntimeError("Hessian cache has not been refreshed")
        by_name = dict(zip(self.names, zip(self.parameters, probes)))
        weighted, count = None, 0
        for name in self.eligible:
            parameter, probe = by_name[name]
            rows, groups = parameter.shape[0], parameter[0].numel() // 4
            value = softmin_pruning_regularizer(
                parameter.reshape(rows, -1), probe.detach().reshape(rows, -1),
                self.factors[name].inverse_qdd, self.references[name], temperature=self.temperature,
            )
            blocks = rows * groups
            weighted = value * blocks if weighted is None else weighted + value * blocks
            count += blocks
        return weighted / count

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
            with self.handoff.install(first_raw):
                epsilon, stats = self.geometry.perturbations(self.rho)
                self.geometry.prepare(epsilon, self.steps, stats["zero_perturbation"])

            saved = [parameter.detach().clone() for parameter in self.parameters]
            with torch.no_grad():
                for parameter, value in zip(self.parameters, epsilon): parameter.add_(value)
            self.optimizer.zero_grad(set_to_none=True)
            with freeze_bn(self.model):
                adversarial_logits = self.model(inputs)
                adversarial_ce = nn.functional.cross_entropy(adversarial_logits, targets)
            adversarial_ce.backward()
            with torch.no_grad():
                for parameter, before in zip(self.parameters, saved): parameter.copy_(before)
                restoration_max_abs = max(float((parameter - before).abs().max())
                                          for parameter, before in zip(self.parameters, saved))
            saved = None

            joint_active = pruning_active and self.lambda_p > 0
            if joint_active:
                regularizer = self._regularizer(epsilon)
                loss = joint_training_loss(adversarial_ce, regularizer, self.lambda_p)
                gradients = torch.autograd.grad(regularizer, self.parameters, allow_unused=True)
                pruning_weight_grad_norm = float(sum(
                    (torch.zeros_like(parameter) if gradient is None else gradient).double().square().sum()
                    for parameter, gradient in zip(self.parameters, gradients)).sqrt())
                for parameter, gradient in zip(self.parameters, gradients):
                    if gradient is not None:
                        if parameter.grad is None: parameter.grad = self.lambda_p * gradient.detach()
                        else: parameter.grad.add_(gradient.detach(), alpha=self.lambda_p)
            else:
                regularizer = adversarial_ce.detach() * 0; loss = adversarial_ce
                pruning_weight_grad_norm = 0.0
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                       for parameter in self.parameters):
                raise FloatingPointError("Non-finite SAM-GPT weight gradient")
            self.optimizer.step(); self.geometry.commit(); self.steps += 1
            if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters):
                raise FloatingPointError("Non-finite SAM-GPT parameter")
            return {"center_loss": float(center.detach()),
                "adversarial_loss": float(adversarial_ce.detach()),
                "pruning_regularizer": float(regularizer.detach()), "joint_loss": float(loss.detach()),
                "pruning_active": bool(joint_active), "eps_norm": stats["eps_norm"],
                "pruning_weight_grad_norm": pruning_weight_grad_norm,
                "restoration_max_abs": restoration_max_abs,
                "correct": int((center_logits.argmax(1) == targets).sum()), "count": len(targets)}
        except BaseException:
            if saved is not None:
                with torch.no_grad():
                    for parameter, before in zip(self.parameters, saved): parameter.copy_(before)
            with torch.no_grad():
                for name, value in self.model.named_buffers(): value.copy_(buffers[name])
            self.geometry.discard(); self.failed = True
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True); self.active = False

    def state_dict(self) -> dict:
        if self.active or self.failed:
            raise RuntimeError("Only a healthy completed step can be checkpointed")
        return {"version": 1, "optimizer": self.optimizer.state_dict(),
            "geometry": self.geometry.state_dict(), "steps": self.steps, "eligible": self.eligible,
            "lambda_p": self.lambda_p, "temperature": self.temperature, "rho": self.rho}

    def load_state_dict(self, state: Mapping) -> None:
        for key, expected in (("version", 1), ("eligible", self.eligible),
                              ("lambda_p", self.lambda_p), ("temperature", self.temperature),
                              ("rho", self.rho)):
            if state[key] != expected: raise ValueError(f"SAM-GPT checkpoint mismatch: {key}")
        self.optimizer.load_state_dict(state["optimizer"]); self.geometry.load_state_dict(state["geometry"])
        self.steps, self.failed = int(state["steps"]), False
