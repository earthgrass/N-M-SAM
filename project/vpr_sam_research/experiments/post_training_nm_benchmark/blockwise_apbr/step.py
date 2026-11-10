"""Independent Stepper binding the frozen APBR training step to new geometries."""
from __future__ import annotations

import torch

from vpr_sam.optim.step import Stepper

from .geometry import BlockwiseGeometry


class BlockwiseStepper(Stepper):
    def __init__(self, model, config, scheme: str, eligible_names):
        self.model, self.config = model, config
        self.method = scheme
        self.geometry = BlockwiseGeometry(model.named_parameters(), config["geometry"], scheme,
                                          config["seed"], eligible_names)
        o = config["optimizer"]
        self.base = torch.optim.SGD(self.geometry.params, lr=o["lr"], momentum=o["momentum"],
                                    weight_decay=o["weight_decay"], nesterov=o["nesterov"])
        self.steps, self.active, self.failed = 0, False, False
        self.masks = {}
