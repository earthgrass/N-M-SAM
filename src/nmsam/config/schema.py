"""Single-device dense CPU toy configuration; research configs are separate."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DenseConfig:
    seed: int = 0
    batch_size: int = 8
    steps: int = 3
    lr: float = 0.05
    rho: float = 0.05
    l2_reg: float = 0.001
    gradient_clipping: float = 5.0
    momentum: float = 0.9
    nesterov: bool = True
    no_weight_decay_on_bn: bool = False

    def __post_init__(self):
        for key in ("seed", "batch_size", "steps"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be a valid integer")
        if self.batch_size < 2:
            raise ValueError("training BatchNorm requires batch_size >= 2")
        for key in ("lr", "rho", "l2_reg", "gradient_clipping", "momentum"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.lr == 0 or self.momentum >= 1:
            raise ValueError("lr > 0 and 0 <= momentum < 1 required")
        for key in ("nesterov", "no_weight_decay_on_bn"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.nesterov and self.momentum == 0:
            raise ValueError("Nesterov requires positive momentum")

