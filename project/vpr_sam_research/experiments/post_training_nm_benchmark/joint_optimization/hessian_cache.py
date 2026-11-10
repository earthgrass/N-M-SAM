"""CPU/GPU training-time GPT Gram cache construction with independent RNG."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import torch
from torch import Tensor, nn

from baseline_adapters.apbr_gpt_joint import CandidateFactors, build_candidate_factors
from baseline_adapters.common.conv2d import grouped_patch_matrices


@dataclass(frozen=True)
class LayerTrainingFactors:
    factors: CandidateFactors
    sampled_patch_ids_sha256: str
    available_patches: int
    sampled_patches: int
    gram_seconds: float
    factor_seconds: float


@torch.no_grad()
def capture_layer_inputs(
    model: nn.Module, layers: dict[str, nn.Conv2d], inputs: Tensor, *,
    storage_device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    captured: dict[str, Tensor] = {}
    hooks = []
    for name, layer in layers.items():
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, values, key=name: captured.__setitem__(
                key, values[0].detach().to(storage_device).clone()
            )
        ))
    try:
        model.eval()
        model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != set(layers):
        raise RuntimeError("Failed to capture every eligible layer input")
    return captured


@torch.no_grad()
def build_training_factors(
    layer: nn.Conv2d,
    layer_input: Tensor,
    *,
    sample_seed: int,
    max_patches: int = 8192,
    percdamp: float = 0.01,
    compute_dtype: torch.dtype = torch.float32,
) -> LayerTrainingFactors:
    matrices = grouped_patch_matrices(layer, layer_input.to(layer.weight.device))
    if len(matrices) != 1:
        raise ValueError("Frozen ResNet-18 eligible layers must have groups=1")
    matrix = matrices[0]
    available = matrix.shape[1]
    count = min(int(max_patches), available)
    if count < 1:
        raise RuntimeError("No activation patches")
    # Sampling is deliberately generated on CPU and then transferred so the
    # selected patch IDs are hardware-independent.
    generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
    ids_cpu = torch.randperm(available, generator=generator)[:count]
    ids_sha = hashlib.sha256(ids_cpu.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    ids = ids_cpu.to(matrix.device)
    x = matrix[:, ids].to(compute_dtype)
    started = time.perf_counter()
    hessian = 2 * (x @ x.T) / count
    gram_seconds = time.perf_counter() - started
    del x, matrix
    started = time.perf_counter()
    factors = build_candidate_factors(hessian, percdamp=percdamp, compute_dtype=compute_dtype)
    factor_seconds = time.perf_counter() - started
    return LayerTrainingFactors(factors, ids_sha, available, count, gram_seconds, factor_seconds)
