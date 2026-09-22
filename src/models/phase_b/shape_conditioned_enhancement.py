"""Shape-conditioned level fusion followed by sparse expert enhancement.

This controlled Phase B ablation reverses the order used by the hierarchical
enhancement. The routing latent first conditions a shared scorer that fuses
the multi-level ViT tokens, then only the experts selected by the existing
Top-K router process that single fused representation::

    alpha_l = softmax_l(g_fuse([mean_P(X^(l)); z]))
    Z_fused = sum_l alpha_l X^(l)
    Z_hat = Z_fused + sum_{k in K_b} pi_k E_k(Z_fused)

The module deliberately owns no router or auxiliary loss. It consumes the
same sparse routing decision as the hierarchical implementation and reuses
the existing ``ExpertBank`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .experts import ExpertBank


FUSION_HIDDEN_DIM = 64


@dataclass
class ShapeConditionedFusionOutput:
    """Outputs of fuse-before-expert enhancement."""

    fused_tokens: Tensor
    """Z_fused before expert processing, shape [B, P, C]."""

    enhanced_fused_tokens: Tensor
    """Z_hat after sparse expert residuals, shape [B, P, C]."""

    layer_weights: Tensor
    """Shape-conditioned alpha over levels, shape [B, L]."""


class ShapeConditionedFusionMoEEnhancement(nn.Module):
    """Fuse levels using ``z``, then run only the selected shape experts."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        num_experts: int = 4,
        num_levels: int = 4,
        latent_dim: int = 64,
        expert_hidden_ratio: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.experts = ExpertBank(
            embed_dim=embed_dim,
            num_experts=num_experts,
            hidden_ratio=expert_hidden_ratio,
        )
        self.g_fuse = nn.Sequential(
            nn.Linear(embed_dim + latent_dim, FUSION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN_DIM, 1),
        )
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_levels = num_levels
        self.latent_dim = latent_dim

    def forward(
        self,
        level_tokens: tuple[Tensor, ...],
        latent: Tensor,
        routing_probs: Tensor,
        expert_indices: Tensor,
    ) -> ShapeConditionedFusionOutput:
        """Fuse ``level_tokens`` and apply the routed expert residuals."""

        if len(level_tokens) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} level-token tensors, got "
                f"{len(level_tokens)}."
            )

        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be [B, {self.latent_dim}], got "
                f"{tuple(latent.shape)}."
            )
        batch_size = latent.shape[0]

        if routing_probs.ndim != 2 or routing_probs.shape != (
            batch_size,
            self.num_experts,
        ):
            raise ValueError(
                f"routing_probs must be [B, {self.num_experts}], got "
                f"{tuple(routing_probs.shape)}."
            )
        if (
            expert_indices.ndim != 2
            or expert_indices.shape[0] != batch_size
            or expert_indices.shape[1] == 0
            or expert_indices.shape[1] > self.num_experts
        ):
            raise ValueError(
                "expert_indices must be non-empty [B, k_e] aligned with "
                f"routing_probs, got {tuple(expert_indices.shape)}."
            )
        if expert_indices.dtype != torch.long:
            raise ValueError("expert_indices must be a torch.long tensor.")
        if bool(
            ((expert_indices < 0) | (expert_indices >= self.num_experts)).any()
        ):
            raise ValueError(
                f"expert_indices must be in [0, {self.num_experts})."
            )

        stacked_tokens = []
        token_count = None
        for level, tokens in enumerate(level_tokens):
            if tokens.ndim != 3 or tokens.shape[0] != batch_size:
                raise ValueError(
                    f"level {level} tokens must be [B, P, C], got "
                    f"{tuple(tokens.shape)}."
                )
            if tokens.shape[2] != self.embed_dim:
                raise ValueError(
                    f"level {level} tokens must end with C={self.embed_dim}, "
                    f"got {tuple(tokens.shape)}."
                )
            if token_count is None:
                token_count = tokens.shape[1]
            elif tokens.shape[1] != token_count:
                raise ValueError(
                    "All levels must share token count P; got "
                    f"P={token_count} and P={tokens.shape[1]}."
                )
            stacked_tokens.append(tokens)

        # [B, L, P, C] and the corresponding global pools [B, L, C].
        stacked = torch.stack(stacked_tokens, dim=1)
        level_pools = stacked.mean(dim=2)
        expanded_latent = latent.unsqueeze(1).expand(-1, self.num_levels, -1)
        scores = self.g_fuse(torch.cat((level_pools, expanded_latent), dim=-1))
        layer_weights = torch.softmax(scores.squeeze(-1), dim=1)
        fused_tokens = (layer_weights[:, :, None, None] * stacked).sum(dim=1)

        # Sparse dispatch: group all (sample, active-slot) pairs by expert.
        # Each routed expert is called once for the batch subset assigned to it.
        active_probs = routing_probs.gather(1, expert_indices)
        num_active = expert_indices.shape[1]
        flat_indices = expert_indices.reshape(-1)
        routed_experts = torch.unique(flat_indices)
        delta = torch.zeros_like(fused_tokens)
        for expert_id in routed_experts.tolist():
            slot_positions = (flat_indices == expert_id).nonzero(as_tuple=True)[0]
            sample_ids = slot_positions // num_active
            slot_ids = slot_positions % num_active
            expert_output = self.experts.experts[int(expert_id)](
                fused_tokens[sample_ids]
            )
            pair_weights = active_probs[sample_ids, slot_ids]
            delta[sample_ids] += pair_weights.view(-1, 1, 1) * expert_output

        return ShapeConditionedFusionOutput(
            fused_tokens=fused_tokens,
            enhanced_fused_tokens=fused_tokens + delta,
            layer_weights=layer_weights,
        )
