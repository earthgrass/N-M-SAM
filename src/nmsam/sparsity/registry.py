"""Explicit master names/support, functional sparse forward, step-time lock."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from torch.func import functional_call
from .layout import MaskLayout
from .mask import build_magnitude_mask, validate_mask
from .ste import effective_weight


class MaskRegistry:
    def __init__(self, model: nn.Module, names=None, *, ste=True):
        self.model, self.ste, self.version, self.locked = model, ste, 0, False
        parameters = dict(model.named_parameters())
        self.masks, self.excluded = {}, {}
        if names is None:
            names = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    key = f"{name}.weight" if name else "weight"
                    if "downsample" in name or "shortcut" in name or name == "conv1":
                        self.excluded[key] = "stem/shortcut policy"
                    else:
                        names.append(key)
        for key in names:
            if key not in parameters:
                raise ValueError(f"unknown sparse parameter: {key}")
            try:
                self.masks[key] = build_magnitude_mask(parameters[key])
            except ValueError as error:
                self.excluded[key] = str(error)
        if not self.masks:
            raise ValueError("scope has no legal N:M target weights")

    def effective(self):
        return {name: effective_weight(p, self.masks[name], self.ste) if name in self.masks else p
                for name, p in self.model.named_parameters()}

    def forward(self, inputs):
        return functional_call(self.model, self.effective(), (inputs,), strict=False)

    def supports(self):
        return tuple(self.masks[name] if name in self.masks else torch.ones_like(p,dtype=torch.bool)
                     for name,p in self.model.named_parameters() if p.requires_grad)

    def directions(self, gradients):
        supports = self.supports()
        if len(supports) != len(gradients):
            raise ValueError("gradient/master order mismatch")
        return tuple(g*m for g,m in zip(gradients,supports))

    def state_dict(self):
        return {"version": self.version, "ste":self.ste,
                "masks":{k:v.clone() for k,v in self.masks.items()}, "excluded":dict(self.excluded)}

    def load_state_dict(self, state):
        if self.locked:
            raise RuntimeError("mask cannot change during SAM step")
        if state["masks"].keys() != self.masks.keys():
            raise ValueError("mask scope mismatch")
        for name, mask in state["masks"].items():
            p=dict(self.model.named_parameters())[name]
            validate_mask(mask, MaskLayout(tuple(p.shape)))
            if mask.device != p.device:
                raise ValueError("mask/master device mismatch")
        self.masks={k:v.detach().to(dtype=torch.bool).clone() for k,v in state["masks"].items()}
        self.version=state["version"]; self.ste=state["ste"]
        self.excluded=dict(state["excluded"])

    def update(self, masks):
        state=self.state_dict(); state["masks"]=masks; state["version"]+=1
        self.load_state_dict(state)

    @contextmanager
    def frozen(self):
        if self.locked:
            raise RuntimeError("nested sparse step")
        parameters=dict(self.model.named_parameters())
        for name,mask in self.masks.items():validate_mask(mask,MaskLayout(tuple(parameters[name].shape)))
        snapshot=self.state_dict(); self.locked=True
        try:
            yield snapshot
            self.assert_unchanged(snapshot)
        except BaseException:
            self.locked=False; self.load_state_dict(snapshot)
            raise
        finally:
            self.locked=False

    def assert_unchanged(self, snapshot):
        if (self.version != snapshot["version"] or self.ste != snapshot["ste"]
            or self.masks.keys()!=snapshot["masks"].keys()
            or any(not torch.equal(self.masks[k],v) for k,v in snapshot["masks"].items())):
            raise RuntimeError("mask mutated inside step")
