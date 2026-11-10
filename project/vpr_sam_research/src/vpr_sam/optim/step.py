from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn

from .geometry import Geometry
from ..repro import finite


@contextmanager
def freeze_bn(model):
    modules = [(m, m.track_running_stats) for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for m, _ in modules:
            m.track_running_stats = False
        yield
    finally:
        for m, value in modules:
            m.track_running_stats = value


class Stepper:
    def __init__(self, model, config, masks=None):
        self.model, self.config = model, config
        self.method = config["optimizer"]["method"]
        self.geometry = Geometry(model.named_parameters(), config["geometry"], self.method, config["seed"])
        o = config["optimizer"]
        self.base = torch.optim.SGD(self.geometry.params, lr=o["lr"], momentum=o["momentum"],
                                    weight_decay=o["weight_decay"], nesterov=o["nesterov"])
        self.steps, self.active, self.failed = 0, False, False
        self.masks = {i: masks[name].to(p.device) for i, ((name, _), p) in
                      enumerate(zip(self.geometry.mapping, self.geometry.params))
                      if masks is not None and name in masks}

    def _check_grads(self, stage):
        total = torch.zeros((), dtype=torch.float64, device=self.geometry.params[0].device)
        for p in self.geometry.params:
            if p.grad is not None:
                finite(p.grad, stage)
                total += p.grad.double().square().sum()
        return float(total.sqrt())

    @torch.no_grad()
    def project_mask(self):
        for i, mask in self.masks.items():
            p = self.geometry.params[i]
            p.mul_(mask)
            if p.grad is not None:
                p.grad.mul_(mask)
            buffer = self.base.state[p].get("momentum_buffer")
            if buffer is not None:
                buffer.mul_(mask)

    def step(self, inputs, targets):
        if self.failed or self.active:
            raise RuntimeError("Stepper failed or active; reload a valid checkpoint")
        self.active = True
        saved = None
        bn_saved = {name: value.clone() for name, value in self.model.named_buffers()}
        base_started = False
        try:
            self.model.train()
            self.base.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            center = nn.functional.cross_entropy(logits, targets)
            finite(center, "center loss")
            center.backward()
            g1 = self._check_grads("g1")
            info = {"eps_norm": 0.0, "ellipsoid_norm": 0.0, "p_sum_error": 0.0,
                    "d_p_max": 0.0, "max_abs_delta_s": 0.0}
            loss, adversarial_logits, g2 = center, logits, g1
            if self.method != "sgd":
                eps, stats = self.geometry.perturbations(self.config["optimizer"]["rho"], self.masks)
                info.update(stats)
                info.update(self.geometry.prepare(eps, self.steps, stats["zero_perturbation"]))
                saved = [p.detach().clone() for p in self.geometry.params]
                with torch.no_grad():
                    for p, e in zip(self.geometry.params, eps):
                        p.add_(e)
                        finite(p, "perturbed parameter")
                    info["addition_roundoff_l2"] = float(sum(
                        ((p.double() - before.double()) - e.double()).square().sum()
                        for p, before, e in zip(self.geometry.params, saved, eps)).sqrt())
                self.base.zero_grad(set_to_none=True)
                with freeze_bn(self.model):
                    adversarial_logits = self.model(inputs)
                    loss = nn.functional.cross_entropy(adversarial_logits, targets)
                    finite(loss, "adversarial loss")
                    loss.backward()
                g2 = self._check_grads("g2")
                with torch.no_grad():
                    for p, before in zip(self.geometry.params, saved):
                        p.copy_(before)
                saved = None
            self.project_mask()
            base_started = True
            self.base.step()
            self.project_mask()
            for p in self.geometry.params:
                finite(p, "updated parameter")
            for state in self.base.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        finite(value, "base optimizer state")
            if self.method != "sgd":
                self.geometry.commit()
            self.steps += 1
            info.update(center_loss=float(center.detach()), adversarial_loss=float(loss.detach()),
                        loss_increase=float((loss-center).detach()), g1_norm=g1, g2_norm=g2,
                        correct=int((logits.argmax(1) == targets).sum()), count=len(targets))
            info.update(self.geometry.diagnostics())
            return info
        except BaseException:
            if saved is not None:
                with torch.no_grad():
                    for p, before in zip(self.geometry.params, saved):
                        p.copy_(before)
            with torch.no_grad():
                for name, value in self.model.named_buffers():
                    value.copy_(bn_saved[name])
            self.geometry.discard()
            # A failed base update is not claimed to be atomic. All failures
            # poison this instance, so partially changed momentum cannot resume.
            self.failed = True
            raise
        finally:
            self.base.zero_grad(set_to_none=True)
            self.active = False

    def state_dict(self):
        if self.active or self.failed:
            raise RuntimeError("Only a healthy completed step can be checkpointed")
        return {"version": 1, "base": self.base.state_dict(), "geometry": self.geometry.state_dict(),
                "steps": self.steps, "mask_names": sorted(self.masks)}

    def load_state_dict(self, state):
        if state["version"] != 1 or state["mask_names"] != sorted(self.masks):
            raise ValueError("Stepper version/mask mismatch")
        self.geometry.load_state_dict(state["geometry"])
        self.base.load_state_dict(state["base"])
        self.steps, self.failed = state["steps"], False
        self.project_mask()
