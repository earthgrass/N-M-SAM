"""General N:M magnitude masks; fixed masks for hard-pruning and recovery."""
from __future__ import annotations
import torch
from torch import nn
from ..config import validate_pruning, canonical_pruning
from ..repro import finite


def nm_mask(weight, n, m, tail):
    if not isinstance(n, int) or not isinstance(m, int) or not 0 < n < m:
        raise ValueError("Require explicit integers 0<N<M")
    if weight.ndim not in (2, 4) or tail not in {"dense", "error"}:
        raise ValueError("Expected linear/conv weight and explicit tail policy")
    finite(weight, "pruning weights")
    flat = weight.detach().reshape(weight.shape[0], -1)
    complete = flat.shape[1] // m * m
    if complete != flat.shape[1] and tail == "error":
        raise ValueError(f"Incomplete M-block in shape {tuple(weight.shape)}")
    mask = torch.ones_like(flat, dtype=torch.bool)
    if complete:
        blocks = flat[:, :complete].reshape(flat.shape[0], -1, m)
        # Stable ties: lower index wins; portable audit-friendly ordering.
        chosen = blocks.abs().argsort(dim=-1, descending=True, stable=True)[..., :n]
        head = torch.zeros_like(blocks, dtype=torch.bool)
        head.scatter_(-1, chosen, True)
        mask[:, :complete] = head.reshape(flat.shape[0], complete)
    return mask.reshape_as(weight)


def make_masks(model, protocol):
    validate_pruning(protocol)
    protocol = canonical_pruning(protocol)
    if protocol["score"] != "magnitude":
        raise ValueError("geometry_guided requires the audited frozen mask import; never regenerate as magnitude")
    layers = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, (nn.Linear, nn.Conv2d))]
    first, last = layers[0][0], layers[-1][0]
    masks, report = {}, []
    for name, mod in layers:
        scope = protocol["scope"]
        selected = not (scope in {"exclude_first", "exclude_first_last", "exclude_first_and_pointwise"} and name == first)
        selected &= not (scope == "exclude_first_last" and name == last)
        selected &= not (scope == "exclude_first_and_pointwise" and isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1))
        if selected:
            key = f"{name}.weight" if name else "weight"
            mask = nm_mask(mod.weight, protocol["n"], protocol["m"], protocol["tail"])
            masks[key] = mask
            flat = mask.reshape(mask.shape[0], -1)
            complete = flat.shape[1] // protocol["m"]
            report.append({"name": key, "shape": list(mask.shape), "full_blocks": flat.shape[0]*complete,
                           "tail_weights": flat.shape[0]*(flat.shape[1] % protocol["m"]),
                           "masked": int((~mask).sum()), "total": mask.numel()})
    if not masks or not any(row["full_blocks"] for row in report):
        raise ValueError("Selected scope has no complete N:M blocks")
    return masks, report


def sparsity_summary(model, layer_report):
    """Report mask-induced sparsity with explicit independent denominators."""
    deleted = sum(row["masked"] for row in layer_report)
    targeted = sum(row["total"] for row in layer_report)
    tails = sum(row["tail_weights"] for row in layer_report)
    full_blocks = targeted - tails
    all_parameters = sum(p.numel() for p in model.parameters())
    return {"mask_deleted_parameters": deleted, "target_layer_weight_parameters": targeted,
            "target_complete_block_parameters": full_blocks, "target_tail_parameters": tails,
            "all_model_parameters": all_parameters,
            "target_layer_sparsity": deleted / targeted,
            "target_complete_block_sparsity": deleted / full_blocks,
            "global_parameter_sparsity": deleted / all_parameters,
            "definition": "mask-induced zeros; natural zeros are excluded from the deletion count"}


@torch.no_grad()
def apply_masks(model, masks):
    params = dict(model.named_parameters())
    for name, mask in masks.items():
        if name not in params or mask.shape != params[name].shape or mask.dtype != torch.bool:
            raise ValueError(f"Mask mismatch: {name}")
        params[name].mul_(mask.to(params[name].device))


def validate_masks(model, masks, protocol):
    validate_pruning(protocol)
    params = dict(model.named_parameters())
    for name, mask in masks.items():
        p = params[name]
        if mask.dtype != torch.bool or mask.shape != p.shape:
            raise ValueError("Bad mask shape/dtype")
        flat = mask.reshape(mask.shape[0], -1)
        head = flat.shape[1] // protocol["m"] * protocol["m"]
        if head and not bool((flat[:, :head].reshape(flat.shape[0], -1, protocol["m"]).sum(-1) == protocol["n"]).all()):
            raise ValueError("Illegal N:M mask")
        if not bool(flat[:, head:].all()):
            raise ValueError("Tail must remain dense")
        if bool((p.detach()[~mask.to(p.device)] != 0).any()):
            raise ValueError("Masked parameters regrew")
