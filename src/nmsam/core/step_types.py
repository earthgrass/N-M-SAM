from dataclasses import dataclass
import torch

TensorTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StepTrace:
    """Opt-in detached snapshots, intentionally expensive and used by tiny tests."""
    base: TensorTuple
    g1: TensorTuple
    epsilon: TensorTuple
    probe: TensorTuple
    g2: TensorTuple
    restored: TensorTuple
    clipped: TensorTuple


@dataclass(frozen=True)
class SAMStepReport:
    task_loss: float
    optimizer_loss: float
    l2_penalty: float
    accuracy: float | None
    g1_norm: float
    g2_norm: float
    epsilon_norm: float
    clipped_gradient_norm: float
    restoration_error: float
    forward_passes: int
    zero_direction: bool
    trace: StepTrace | None = None

