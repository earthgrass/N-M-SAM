"""Independent blockwise-feedback geometries built on the frozen APBR formula."""
from __future__ import annotations

import math
from typing import Iterable

import torch

from vpr_sam.optim.geometry import Geometry
from vpr_sam.repro import finite


class BlockwiseGeometry(Geometry):
    """APBR with 2:4-block feedback and optional equal block energy.

    The frozen parent APBR enforces an ellipsoid boundary, not a fixed L2 norm.
    Scheme 4 returns that perturbation byte-for-byte. Scheme 1 preserves its
    *current actual L2 energy* while redistributing only eligible block energy.
    Residual coordinates form one additional feedback group, which preserves
    the original zero-log-volume invariant without changing the parent source.
    A block with an exactly zero perturbation has no directional feedback
    information: it keeps zero perturbation and receives the neutral update.
    """

    def __init__(self, named_parameters, config, scheme: str, seed: int, eligible_names: Iterable[str]):
        if scheme not in {"global_block_feedback", "equal_block_energy"}:
            raise ValueError(f"Unknown blockwise scheme: {scheme}")
        super().__init__(named_parameters, config, method="vpr", seed=seed)
        self.scheme = scheme
        self.eligible_names = frozenset(eligible_names)
        self.eligible_indices = {i for i, (name, _) in enumerate(self.mapping) if name in self.eligible_names}
        if self.eligible_names != {self.mapping[i][0] for i in self.eligible_indices}:
            raise ValueError("Eligible parameter mapping mismatch")
        self.block_count = sum(self.params[i].reshape(self.params[i].shape[0], -1).shape[1] // 4
                               * self.params[i].shape[0] for i in self.eligible_indices)
        self.grouped_coordinates = self.block_count * 4
        self.residual_coordinates = self.d - self.grouped_coordinates
        if self.block_count <= 0 or self.residual_coordinates <= 0:
            raise ValueError("Expected nonempty 2:4 blocks and residual coordinates")
        self.degenerate_threshold = float(config.get("block_degenerate_threshold", 1e-30))
        if self.degenerate_threshold < 0 or not math.isfinite(self.degenerate_threshold):
            raise ValueError("Invalid fixed block-degeneracy threshold")
        self.last_block_stats: dict[str, float | int] = {}

    def _split_masks(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        flat = value.reshape(value.shape[0], -1)
        end = flat.shape[1] // 4 * 4
        return flat[:, :end], flat[:, end:] if end < flat.shape[1] else None

    @torch.no_grad()
    def perturbations(self, rho, masks=None):
        if masks:
            raise ValueError("Dense blockwise training does not accept pruning masks")
        original, stats = super().perturbations(rho, masks=None)
        stats = dict(stats)
        if stats["zero_perturbation"]:
            self.last_block_stats = {"block_count": self.block_count, "degenerate_blocks": self.block_count}
            return original, stats | self.last_block_stats

        original_l2 = float(sum(e.double().square().sum() for e in original).sqrt())
        if self.scheme == "global_block_feedback":
            applied = [e.clone() for e in original]
            block_norms = []
            for i in self.eligible_indices:
                head, _ = self._split_masks(applied[i])
                block_norms.append(head.double().reshape(head.shape[0], -1, 4).square().sum(-1).sqrt())
            all_norms = torch.cat([x.reshape(-1) for x in block_norms])
            active_norms = all_norms[all_norms > self.degenerate_threshold]
            block_stats = {
                "block_count": self.block_count, "active_blocks": int(active_norms.numel()),
                "degenerate_blocks": int((all_norms <= self.degenerate_threshold).sum()),
                "block_norm_min": float(all_norms.min()), "block_norm_max": float(all_norms.max()),
                "block_norm_mean": float(all_norms.mean()), "block_norm_relative_spread": float(all_norms.std(unbiased=False) / all_norms.mean()),
                "active_block_norm_relative_spread": (float(active_norms.std(unbiased=False) / active_norms.mean())
                                                       if active_norms.numel() else 0.0),
                "block_scale_min": 1.0, "block_scale_max": 1.0,
                "original_eps_l2": original_l2, "applied_eps_l2": original_l2,
                "l2_preservation_error": 0.0, "perturbation_matches_original": True,
            }
        else:
            eligible_energy = torch.zeros((), dtype=torch.float64, device=original[0].device)
            norms_by_index, active_by_index = {}, {}
            active_block_count = 0
            for i in self.eligible_indices:
                head, _ = self._split_masks(original[i])
                groups = head.double().reshape(head.shape[0], -1, 4)
                norms = groups.square().sum(-1).sqrt()
                if not bool(torch.isfinite(norms).all()):
                    raise FloatingPointError("Non-finite eligible block norm")
                active = norms > self.degenerate_threshold
                norms_by_index[i] = norms
                active_by_index[i] = active
                active_block_count += int(active.sum())
                eligible_energy += groups.square().sum()
            target = ((eligible_energy / active_block_count).sqrt() if active_block_count
                      else torch.zeros_like(eligible_energy))
            applied, scales = [], []
            for i, e in enumerate(original):
                value = e.clone()
                if i in self.eligible_indices:
                    flat = value.reshape(value.shape[0], -1)
                    end = flat.shape[1] // 4 * 4
                    groups = flat[:, :end].reshape(flat.shape[0], -1, 4)
                    scale = torch.zeros_like(norms_by_index[i])
                    active = active_by_index[i]
                    scale[active] = target / norms_by_index[i][active]
                    groups.mul_(scale.unsqueeze(-1).to(groups.dtype))
                    scales.append(scale[active].reshape(-1))
                applied.append(value)
            applied_l2 = float(sum(e.double().square().sum() for e in applied).sqrt())
            all_scales = torch.cat(scales) if scales and any(item.numel() for item in scales) else torch.zeros(1, device=original[0].device)
            new_norms = []
            for i in self.eligible_indices:
                head, _ = self._split_masks(applied[i])
                new_norms.append(head.double().reshape(head.shape[0], -1, 4).square().sum(-1).sqrt().reshape(-1))
            all_norms = torch.cat(new_norms)
            active_norms = all_norms[all_norms > self.degenerate_threshold]
            rel_error = abs(applied_l2 - original_l2) / max(original_l2, 1e-300)
            if rel_error > 2e-6:
                raise FloatingPointError(f"Equal-energy L2 preservation failed: {rel_error}")
            block_stats = {
                "block_count": self.block_count, "active_blocks": active_block_count,
                "degenerate_blocks": self.block_count - active_block_count,
                "block_norm_min": float(all_norms.min()), "block_norm_max": float(all_norms.max()),
                "block_norm_mean": float(all_norms.mean()), "block_norm_relative_spread": float(all_norms.std(unbiased=False) / all_norms.mean()),
                "active_block_norm_relative_spread": (float(active_norms.std(unbiased=False) / active_norms.mean())
                                                       if active_norms.numel() else 0.0),
                "block_scale_min": float(all_scales.min()), "block_scale_max": float(all_scales.max()),
                "original_eps_l2": original_l2, "applied_eps_l2": applied_l2,
                "l2_preservation_error": rel_error, "perturbation_matches_original": False,
            }
        for e in applied:
            finite(e, "blockwise applied perturbation")
        self.last_block_stats = block_stats
        return applied, stats | block_stats

    @torch.no_grad()
    def prepare(self, eps, successful_steps, zero=False):
        if zero or successful_steps < self.config["warmup_steps"]:
            self.candidate = (None, zero)
            return {"p_sum_error": 0.0, "d_p_max": 0.0, "max_abs_delta_s": 0.0,
                    "block_delta_sum_max": 0.0, "residual_delta_sum": 0.0}
        gamma = self.config["gamma"]
        deltas = [torch.zeros_like(s, dtype=torch.float64) for s in self.logs]
        residual_energy = torch.zeros((), dtype=torch.float64, device=eps[0].device)
        residual_count = 0
        block_sum_max = 0.0
        p_error_max = 0.0
        d_p_max = 0.0
        for i, (e, delta) in enumerate(zip(eps, deltas)):
            square = e.double().square()
            if i in self.eligible_indices:
                flat_square = square.reshape(square.shape[0], -1)
                flat_delta = delta.reshape(delta.shape[0], -1)
                end = flat_square.shape[1] // 4 * 4
                groups = flat_square[:, :end].reshape(flat_square.shape[0], -1, 4)
                energies = groups.sum(-1, keepdim=True)
                if not bool(torch.isfinite(energies).all()):
                    raise FloatingPointError(f"Non-finite block feedback energy: {self.mapping[i][0]}")
                active = energies > self.degenerate_threshold ** 2
                shares = torch.zeros_like(groups)
                shares[active.expand_as(groups)] = (groups / energies.clamp_min(torch.finfo(energies.dtype).tiny))[active.expand_as(groups)]
                # A block with no perturbation direction contains no relative
                # energy signal.  Its uniquely non-arbitrary, zero-sum update
                # is the neutral update.  Active blocks retain the exact rule.
                group_delta = torch.zeros_like(groups)
                group_delta[active.expand_as(groups)] = (gamma * (1 - 4 * shares))[active.expand_as(groups)]
                flat_delta[:, :end] = group_delta.reshape(flat_square.shape[0], end)
                block_sum_max = max(block_sum_max, float(group_delta.sum(-1).abs().max()))
                if bool(active.any()):
                    p_error_max = max(p_error_max, float((shares.sum(-1, keepdim=True)[active] - 1).abs().max()))
                    d_p_max = max(d_p_max, 4 * float(shares.max()))
                if end < flat_square.shape[1]:
                    residual_energy += flat_square[:, end:].sum()
                    residual_count += flat_square[:, end:].numel()
            else:
                residual_energy += square.sum()
                residual_count += square.numel()
        if residual_count != self.residual_coordinates:
            raise RuntimeError("Residual-coordinate accounting mismatch")
        residual_sum = 0.0
        if float(residual_energy) > 0:
            for i, (e, delta) in enumerate(zip(eps, deltas)):
                square = e.double().square()
                if i in self.eligible_indices:
                    flat_square = square.reshape(square.shape[0], -1)
                    flat_delta = delta.reshape(delta.shape[0], -1)
                    end = flat_square.shape[1] // 4 * 4
                    if end < flat_square.shape[1]:
                        share = flat_square[:, end:] / residual_energy
                        flat_delta[:, end:] = gamma * (1 - residual_count * share)
                else:
                    deltas[i] = gamma * (1 - residual_count * square / residual_energy)
            residual_sum = float(sum(delta.double().sum() for i, delta in enumerate(deltas)
                                     if i not in self.eligible_indices)
                                 + sum(delta.reshape(delta.shape[0], -1)[:, self.params[i].reshape(self.params[i].shape[0], -1).shape[1] // 4 * 4:].double().sum()
                                       for i, delta in enumerate(deltas) if i in self.eligible_indices))
        candidate = [(s.double() + delta).to(self.dtype) for s, delta in zip(self.logs, deltas)]
        self.check_volume(candidate)
        self.candidate = (candidate, False)
        return {"p_sum_error": p_error_max, "d_p_max": d_p_max,
                "max_abs_delta_s": max(float(delta.abs().max()) for delta in deltas),
                "block_delta_sum_max": block_sum_max, "residual_delta_sum": residual_sum,
                "residual_energy": float(residual_energy), "residual_coordinates": residual_count}

    def state_dict(self):
        state = super().state_dict()
        state.update(scheme=self.scheme, eligible_names=sorted(self.eligible_names),
                     block_count=self.block_count, grouped_coordinates=self.grouped_coordinates,
                     residual_coordinates=self.residual_coordinates,
                     degenerate_threshold=self.degenerate_threshold)
        return state

    def load_state_dict(self, state):
        for key, expected in (("scheme", self.scheme), ("eligible_names", sorted(self.eligible_names)),
                              ("block_count", self.block_count), ("grouped_coordinates", self.grouped_coordinates),
                              ("residual_coordinates", self.residual_coordinates),
                              ("degenerate_threshold", self.degenerate_threshold)):
            if state.get(key) != expected:
                raise ValueError(f"Blockwise geometry checkpoint mismatch: {key}")
        parent = {key: state[key] for key in ("version", "mapping", "d", "config", "method", "logs", "updates", "zero_skips")}
        super().load_state_dict(parent)
