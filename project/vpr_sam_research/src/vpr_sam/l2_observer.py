"""Read-only diagnostics around the frozen Stepper; no new optimizer formula.

The observer uses the existing perturbation result, reads the actual parameter
addition before the second forward, and never consumes RNG or changes buffers.
"""
import math
import torch
from .optim import Stepper

L2_RTOL = 2e-5  # same approved FP32 relative tolerance as frozen Geometry
L2_ATOL = 1e-12
DIAGNOSTIC_INTERVAL = 352  # step 1,353,...: one fixed batch per epoch


class ObservedStepper(Stepper):
    def __init__(self, model, config, masks=None):
        super().__init__(model, config, masks)
        if self.method != "fixed_l2":
            raise ValueError("V2 observer is authorized only for fixed_l2")
        original = self.geometry.perturbations

        def observe(rho, masks=None):
            eps, stats = original(rho, masks)
            norm = stats["eps_norm"]
            if not stats["zero_perturbation"] and not math.isclose(norm, rho, rel_tol=L2_RTOL, abs_tol=L2_ATOL):
                raise FloatingPointError("fixed_l2 FP32 perturbation radius gate failed")
            self._eps = eps
            self._center = [p.detach().clone() for p in self.geometry.params]
            self._extra = {"epsilon_l2_over_rho": norm / rho}
            if self.steps % DIAGNOSTIC_INTERVAL == 0:
                gs = [torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in self.geometry.params]
                gnorm = sum(g.double().square().sum() for g in gs).sqrt()
                dot = sum((g.double()*e.double()).sum() for g, e in zip(gs, eps))
                self._extra.update(linear_strength=float(dot),
                    direction_cosine=float(dot/(gnorm*norm)) if norm and float(gnorm) else None,
                    linear_strength_over_spherical=float(dot/(rho*gnorm)) if float(gnorm) else None)
            return eps, stats
        self.geometry.perturbations = observe

    def step(self, inputs, targets):
        self._eps = self._center = None
        self._extra = {}

        @torch.no_grad()
        def before_forward(_model, _args):
            if self._center is None:  # first forward
                return
            actual_sq = torch.zeros((), dtype=torch.float64, device=inputs.device)
            error_sq = actual_sq.clone()
            bound_sq = actual_sq.clone()
            for p, center, eps in zip(self.geometry.params, self._center, self._eps):
                delta = p.double()-center.double()
                actual_sq += delta.square().sum()
                error = delta-eps.double()
                error_sq += error.square().sum()
                # Conservative IEEE FP32 round-to-nearest error bound for a+b.
                bound = torch.finfo(torch.float32).eps*(center.double().abs()+eps.double().abs())
                bound += torch.finfo(torch.float32).tiny
                bound_sq += bound.square().sum()
                if not bool((error.abs() <= bound).all()):
                    raise FloatingPointError("FP32 parameter-addition rounding bound failed")
            self._extra.update(delta_w_l2=float(actual_sq.sqrt()),
                observed_addition_roundoff_l2=float(error_sq.sqrt()),
                addition_roundoff_bound_l2=float(bound_sq.sqrt()))
            self._center = self._eps = None

        hook = self.model.register_forward_pre_hook(before_forward)
        try:
            info = super().step(inputs, targets)
            info.update(self._extra)
            if not all(math.isfinite(v) for v in self._extra.values() if v is not None):
                self.failed = True
                raise FloatingPointError("Non-finite read-only diagnostic")
            if inputs.device.type == "cuda":
                info["gpu_allocated_bytes"] = torch.cuda.memory_allocated(inputs.device)
                info["gpu_peak_bytes"] = torch.cuda.max_memory_allocated(inputs.device)
            return info
        finally:
            hook.remove()
            self._center = self._eps = None
