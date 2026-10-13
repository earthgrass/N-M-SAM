"""Capture and replay torch, Python and NumPy stochastic state."""
from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class RNGState:
    cpu: torch.Tensor
    cuda: list[torch.Tensor] | None
    python: object
    numpy: tuple


def capture_rng() -> RNGState:
    return RNGState(torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                    random.getstate(), np.random.get_state())


def restore_rng(state: RNGState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda is not None:
        torch.cuda.set_rng_state_all(state.cuda)
    random.setstate(state.python)
    np.random.set_state(state.numpy)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

