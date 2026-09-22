"""Shared diagnostics for the Phase B controlled ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _mean_tensor(value: Any) -> Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value.detach().mean()


def actual_layer_weights(
    diagnostics: dict[str, Any],
) -> tuple[Tensor | None, tuple[int, ...] | None]:
    """Resolve weights that actually enter each ablation's token fusion."""

    stage = diagnostics.get("phase_b_moe")
    if stage is None:
        stage = diagnostics.get("phase_b_ablation")
    weights = getattr(stage, "layer_weights", None)
    if not torch.is_tensor(weights):
        weights = diagnostics.get("level_weights")
    if not torch.is_tensor(weights):
        return None, None
    if weights.ndim != 2:
        raise ValueError(
            f"layer weights must be [B, L], got {tuple(weights.shape)}."
        )

    level_ids = diagnostics.get("level_ids")
    if level_ids is None:
        return weights, None
    resolved_ids = tuple(int(level) for level in level_ids)
    if len(resolved_ids) != weights.shape[1]:
        raise ValueError(
            "level_ids must align with layer weights, got "
            f"{resolved_ids} and {tuple(weights.shape)}."
        )
    return weights, resolved_ids


def layer_fusion_metrics(
    diagnostics: dict[str, Any],
) -> dict[str, Tensor]:
    """Entropy, maximum and named mean weights for the actual fusion."""

    weights, level_ids = actual_layer_weights(diagnostics)
    if weights is None:
        return {}
    if weights.ndim != 2:
        raise ValueError(
            f"layer weights must be [B, L], got {tuple(weights.shape)}."
        )

    detached = weights.detach()
    entropy = -(
        detached.clamp_min(torch.finfo(detached.dtype).tiny).log() * detached
    ).sum(dim=1)
    metrics = {
        "level_weight_entropy": entropy.mean(),
        "level_weight_max": detached.max(dim=1).values.mean(),
    }
    if level_ids is not None:
        for index, level in enumerate(level_ids):
            metrics[f"layer_weight_{level}"] = detached[:, index].mean()
    return metrics


def routing_metrics(stage: Any) -> dict[str, Tensor]:
    """Routing diagnostics based on active-slot usage and dense probabilities."""

    routing = getattr(stage, "routing", None)
    if routing is None:
        return {}

    dense_probs = routing.dense_probs.detach()
    expert_indices = routing.expert_indices.detach()
    if dense_probs.ndim != 2 or expert_indices.ndim != 2:
        raise ValueError("routing tensors must be [B, K] and [B, k_e].")
    if expert_indices.shape[0] != dense_probs.shape[0]:
        raise ValueError("routing tensors must share batch size.")

    num_experts = dense_probs.shape[1]
    slot_counts = torch.bincount(
        expert_indices.reshape(-1),
        minlength=num_experts,
    ).to(dtype=dense_probs.dtype, device=dense_probs.device)
    usage = slot_counts / max(int(expert_indices.numel()), 1)
    usage_entropy = -(
        usage.clamp_min(torch.finfo(usage.dtype).tiny).log() * usage
    ).sum()

    sorted_probs = dense_probs.sort(dim=1, descending=True).values
    if num_experts > 1:
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margin = sorted_probs[:, 0]

    metrics = {
        "expert_usage_entropy": usage_entropy,
        "routing_max_probability": dense_probs.max(dim=1).values.mean(),
        "routing_top1_top2_margin": margin.mean(),
    }
    for expert_index in range(num_experts):
        metrics[f"expert_usage_fraction_{expert_index}"] = usage[expert_index]

    latent_kl = _mean_tensor(getattr(stage, "latent_kl", None))
    if latent_kl is not None:
        metrics["latent_kl"] = latent_kl
    balance = _mean_tensor(getattr(stage, "balance", None))
    if balance is not None:
        metrics["load_balance"] = balance
        metrics["load_balance_loss"] = balance
    return metrics


def enhancement_metrics(stage: Any) -> dict[str, Tensor]:
    """Norm diagnostics plus compatibility entropy aliases."""

    if stage is None:
        return {}

    metrics: dict[str, Tensor] = {}
    ratio = _mean_tensor(getattr(stage, "aux_norm_ratio", None))
    if ratio is not None:
        metrics["enhancement_aux_ratio"] = ratio
    fused_norm = _mean_tensor(getattr(stage, "fused_token_norm", None))
    if fused_norm is not None:
        metrics["fused_token_norm"] = fused_norm
    enhanced_norm = _mean_tensor(getattr(stage, "enhanced_token_norm", None))
    if enhanced_norm is not None:
        metrics["enhanced_token_norm"] = enhanced_norm

    weights = getattr(stage, "layer_weights", None)
    if torch.is_tensor(weights):
        entropy = -(
            weights.detach().clamp_min(torch.finfo(weights.dtype).tiny).log()
            * weights.detach()
        ).sum(dim=1).mean()
        if torch.is_tensor(
            getattr(stage, "shape_fusion_layer_weights", None)
        ):
            metrics["shape_fusion_layer_weight_entropy"] = entropy
        else:
            metrics["fused_level_weight_entropy"] = entropy
    return metrics
