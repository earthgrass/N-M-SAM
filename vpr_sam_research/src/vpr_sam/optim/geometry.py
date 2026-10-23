"""Frozen global VPR update. No EMA, clipping or volume recentering."""
from __future__ import annotations
import math

import torch

from ..repro import finite, seed_for


class Geometry:
    def __init__(self, named_parameters, config, method="vpr", seed=0):
        self.params, self.mapping, seen = [], [], set()
        for name, p in named_parameters:
            if id(p) not in seen:
                if p.dtype != torch.float32 or not p.requires_grad:
                    raise ValueError("First version requires fixed trainable FP32 parameters")
                seen.add(id(p))
                self.params.append(p)
                self.mapping.append((name, tuple(p.shape)))
        self.d = sum(p.numel() for p in self.params)
        if not self.d or len({p.device for p in self.params}) != 1:
            raise ValueError("Require nonempty parameters on a single device")
        self.config, self.method = dict(config), method
        self.dtype = getattr(torch, config["dtype"])
        self.logs = ([torch.zeros_like(p, dtype=self.dtype) for p in self.params]
                     if method not in {"sgd", "sam"} else [])
        self.updates, self.zero_skips = 0, 0
        self.candidate = None
        if method == "static":
            gen = torch.Generator(device="cpu").manual_seed(seed_for(seed, "static-geometry"))
            values = [torch.randn(p.shape, generator=gen, dtype=torch.float64).to(p.device) * config["static_std"]
                      for p in self.params]
            mean = sum(v.sum() for v in values) / self.d
            self.logs = [(v - mean).to(self.dtype) for v in values]
        self.check_volume(self.logs)

    def check_volume(self, logs):
        total = torch.zeros((), dtype=torch.float64, device=self.params[0].device)
        absolute = total.clone()
        for s in logs:
            finite(s, "log-radius")
            total += s.double().sum()
            absolute += s.double().abs().sum()
        bound = self.config["volume_atol"] + self.config["volume_rtol"] * float(absolute)
        if abs(float(total)) > bound:
            raise FloatingPointError(f"log-volume drift {float(total)} exceeds tolerance {bound}; no recentering applied")
        return float(total)

    @torch.no_grad()
    def perturbations(self, rho, masks=None):
        """Stable log-domain elliptic formula; feedback later uses these FP32 buffers.

        masks are only for explicitly fixed-mask SAM recovery/evaluation. Dense
        VPR training never restricts coordinates or changes d.
        """
        if self.candidate is not None:
            raise RuntimeError("Uncommitted geometry update")
        if self.method in {"sam", "sgd"}:
            directions = []
            for i, p in enumerate(self.params):
                g = torch.zeros_like(p) if p.grad is None else p.grad.detach()
                finite(g, "g1")
                if masks is not None and i in masks:
                    g = g * masks[i]
                directions.append(g)
            norm = sum(g.double().square().sum() for g in directions).sqrt()
            if float(norm) == 0:
                return [torch.zeros_like(p) for p in self.params], {"zero_perturbation": True, "eps_norm": 0., "ellipsoid_norm": 0.}
            eps = [(g.double() * (rho/norm)).float() for g in directions]
            actual = float(sum(e.double().square().sum() for e in eps).sqrt())
            return eps, {"zero_perturbation": False, "eps_norm": actual, "ellipsoid_norm": actual}
        logs = self.logs if self.method not in {"sam", "sgd"} else [torch.zeros_like(s) for s in self.logs]
        h, b, signs = [], [], []
        for index, (p, s) in enumerate(zip(self.params, logs)):
            grad = torch.zeros_like(p) if p.grad is None else p.grad.detach()
            finite(grad, "g1")
            if masks is not None and index in masks:
                grad = grad * masks[index]
            g = grad.double()
            logg = g.abs().log()  # log(0)=-inf is a deliberate masked contribution
            h.append(s.double() + logg)
            b.append(2 * s.double() + logg)
            signs.append(g.sign())
        maximum = torch.stack([x.max() for x in h]).max()
        if bool(torch.isneginf(maximum)):
            return [torch.zeros_like(p) for p in self.params], {"zero_perturbation": True, "eps_norm": 0.0, "ellipsoid_norm": 0.0}
        finite(maximum, "gradient/geometry exponent")
        norm_log = maximum + .5 * torch.log(sum(torch.exp(2 * (x - maximum)).sum() for x in h))
        eps, ellipsoid_parts = [], []
        for p, s, exponent, sign in zip(self.params, logs, b, signs):
            value = sign * torch.exp(math.log(rho) + exponent - norm_log)
            finite(value, "perturbation before FP32 cast")
            applied = value.to(p.dtype)
            finite(applied, "FP32 perturbation")
            eps.append(applied)
            # Evaluate ||A^-1 epsilon|| using log exponents to avoid inverse overflow.
            ellipsoid_parts.append(torch.exp(2 * (applied.double().abs().log() - s.double())).sum())
        eps_sq = sum(e.double().square().sum() for e in eps)
        if float(eps_sq) == 0:
            raise FloatingPointError("Nonzero mathematical perturbation underflowed completely in FP32")
        if self.method == "fixed_l2":
            scale = rho / eps_sq.sqrt()
            eps = [(e.double() * scale).float() for e in eps]
            eps_sq = sum(e.double().square().sum() for e in eps)
            # This ablation intentionally changes the ellipse boundary.
            ellipsoid_parts = [torch.exp(2 * (e.double().abs().log() - s.double())).sum()
                               for e, s in zip(eps, logs)]
        ellipsoid_norm = float(sum(ellipsoid_parts).sqrt())
        if not math.isfinite(ellipsoid_norm):
            raise FloatingPointError("Non-finite ellipsoid norm")
        if self.method != "fixed_l2" and not math.isclose(ellipsoid_norm, rho, rel_tol=2e-5, abs_tol=1e-12):
            raise FloatingPointError("FP32 perturbation does not satisfy elliptic boundary tolerance")
        return eps, {"zero_perturbation": False, "eps_norm": float(eps_sq.sqrt()),
                     "ellipsoid_norm": ellipsoid_norm}

    @torch.no_grad()
    def prepare(self, eps, successful_steps, zero=False):
        dynamic = self.method in {"vpr", "positive", "fixed_l2"}
        if zero or not dynamic or successful_steps < self.config["warmup_steps"]:
            self.candidate = (None, zero and dynamic)
            return {"p_sum_error": 0.0, "d_p_max": 0.0, "max_abs_delta_s": 0.0}
        for e in eps:
            finite(e, "feedback epsilon")
        denominator = sum(e.double().square().sum() for e in eps)
        finite(denominator, "global feedback energy")
        if float(denominator) <= 0:
            raise FloatingPointError("Invalid feedback energy")
        shares = [e.double().square() / denominator for e in eps]
        sign = -1 if self.method == "positive" else 1
        deltas = [sign * self.config["gamma"] * (1 - self.d * p) for p in shares]
        candidate = [(s.double() + delta).to(self.dtype) for s, delta in zip(self.logs, deltas)]
        self.check_volume(candidate)
        self.candidate = (candidate, False)
        return {"p_sum_error": float(abs(sum(p.sum() for p in shares) - 1)),
                "d_p_max": self.d * max(float(p.max()) for p in shares),
                "max_abs_delta_s": max(float(delta.abs().max()) for delta in deltas)}

    def commit(self):
        if self.candidate is None:
            raise RuntimeError("No prepared update")
        logs, zero = self.candidate
        if logs is not None:
            self.logs = logs
            self.updates += 1
        self.zero_skips += int(zero)
        self.candidate = None

    def discard(self):
        self.candidate = None

    @torch.no_grad()
    def diagnostics(self):
        volume = self.check_volume(self.logs)
        if not self.logs:
            return {"d": self.d, "log_volume": 0., "log_radius_min": 0., "log_radius_max": 0.,
                    "log_radius_rms": 0., "geometry_updates": 0, "zero_geometry_skips": 0,
                    "layer_log_volume": {}}
        lo, hi = min(float(s.min()) for s in self.logs), max(float(s.max()) for s in self.logs)
        square = sum(float(s.double().square().sum()) for s in self.logs)
        return {"d": self.d, "log_volume": volume, "log_radius_min": lo,
                "log_radius_max": hi, "log_radius_rms": math.sqrt(square / self.d),
                "geometry_updates": self.updates, "zero_geometry_skips": self.zero_skips,
                "layer_log_volume": {name: float(s.double().sum()) for (name, _), s in zip(self.mapping, self.logs)}}

    def state_dict(self):
        if self.candidate is not None:
            raise RuntimeError("Cannot checkpoint pending geometry transaction")
        return {"version": 1, "mapping": self.mapping, "d": self.d, "config": self.config,
                "method": self.method, "logs": [s.detach().cpu().clone() for s in self.logs],
                "updates": self.updates, "zero_skips": self.zero_skips}

    def load_state_dict(self, state):
        for key, expected in (("version", 1), ("mapping", self.mapping), ("d", self.d),
                              ("config", self.config), ("method", self.method)):
            if state[key] != expected:
                raise ValueError(f"Geometry checkpoint mismatch: {key}")
        if len(state["logs"]) != len(self.logs):
            raise ValueError("Geometry tensor count mismatch")
        result = []
        for saved, current in zip(state["logs"], self.logs):
            if saved.dtype != current.dtype or saved.shape != current.shape:
                raise ValueError("Geometry dtype/shape mismatch; no silent conversion")
            result.append(saved.to(current.device).clone())
        self.check_volume(result)
        self.logs, self.updates, self.zero_skips = result, state["updates"], state["zero_skips"]
        self.candidate = None
