from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class StepBatch:
    """Already transformed tensors; callers/models must not mutate them."""
    inputs: torch.Tensor
    targets: torch.Tensor

    def __post_init__(self):
        if self.inputs.shape[0] != self.targets.shape[0] or self.inputs.shape[0] == 0:
            raise ValueError("batch must be nonempty with matching sample counts")
        if self.inputs.device != self.targets.device:
            raise ValueError("inputs and targets must share a device")

