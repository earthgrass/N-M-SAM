import torch
from torch import nn


def validate_dense_device(model: nn.Module, batch_device: torch.device) -> None:
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("no trainable parameters")
    if any(p.device != batch_device for p in model.parameters()) or any(
        b.device != batch_device for b in model.buffers()
    ):
        raise ValueError("single shared model/batch device required")
    if len({p.dtype for p in parameters}) != 1 or parameters[0].dtype not in (
        torch.float32, torch.float64
    ):
        raise ValueError("bootstrap supports uniform FP32/FP64 without AMP")

