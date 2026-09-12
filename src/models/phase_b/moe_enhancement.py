"""Shape-conditioned hierarchical MoE enhancement (proposal Eqs. 47-50).

Given the per-sample routing decision (``pi``, ``K_b``) and the multi-level
ViT tokens ``X^(l)``, this module refines the tokens with the routed experts
and fuses the levels into a single spatial map:

    X_hat^(l)_b = X^(l)_b + sum_{k in K_b} pi_{b,k} beta_{b,k,l} E_k(X^(l)_b)   (47)
    gamma_{b,l} = sum_{k in K_b} pi_{b,k} beta_{b,k,l}                          (48/49)
    Z_fused,b   = sum_l gamma_{b,l} X_hat^(l)_b                                 (50)

Only the ``k_e`` active experts are evaluated per sample (gather-then-run,
not loop-over-K): inactive experts contribute neither compute nor parameters
to the pass. ``Z_fused`` is returned flat, [B, P, C]; reshaping to spatial
layout and channel-reducing to the SAM decoder's 256-d embedding is the next
module's job (the neck, Eq. 53-54).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .experts import ExpertBank
from .layer_attention import LayerPreferenceScorer


@dataclass
class EnhancementOutput:
    """Everything the enhancement stage produces."""

    fused_tokens: Tensor
    """Z_fused, shape [B, P, C]."""

    enhanced_tokens: tuple[Tensor, ...]
    """X_hat^(l) per level, each [B, P, C]; kept for diagnostics/ablations."""

    layer_weights: Tensor
    """gamma_{b,l}, shape [B, L_levels]; sums to 1 by construction when
    pi sums to 1 over K_b and beta sums to 1 over levels."""

    expert_layer_weights: Tensor
    """beta_{b,k,l} for the active experts, [B, k_e, L_levels]."""


class HierarchicalMoEEnhancement(nn.Module):
    """Routed expert refinement of multi-level tokens + level fusion."""

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
        self.experts = ExpertBank(
            embed_dim=embed_dim,
            num_experts=num_experts,
            hidden_ratio=expert_hidden_ratio,
        )
        self.layer_scorer = LayerPreferenceScorer(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_experts=num_experts,
            num_levels=num_levels,
        )
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_levels = num_levels

    def forward(
        self,
        level_tokens: tuple[Tensor, ...],
        level_pools: Tensor,
        latent: Tensor,
        routing_probs: Tensor,
        expert_indices: Tensor,
    ) -> EnhancementOutput:
        """Run the routed enhancement.

        Args:
            level_tokens: X^(l), len-L tuple of [B, P_l, C] token tensors.
                All levels must share P (the same 16x16 grid) since their
                outputs are summed elementwise in Eq. 50.
            level_pools: v^(l), [B, L_levels, C]; the g_layer input.
            latent: z, [B, d_z].
            routing_probs: pi, [B, K]; sparse, zero off K_b, sums to 1.
            expert_indices: K_b, [B, k_e] (long).

        Returns:
            EnhancementOutput with Z_fused [B, P, C].
        """

        if len(level_tokens) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} level-token tensors, got "
                f"{len(level_tokens)}."
            )
        if level_pools.ndim != 3 or level_pools.shape[1] != self.num_levels:
            raise ValueError(
                f"level_pools must be [B, {self.num_levels}, C], got "
                f"{tuple(level_pools.shape)}."
            )
        if routing_probs.ndim != 2:
            raise ValueError(
                f"routing_probs must be [B, K], got {tuple(routing_probs.shape)}."
            )
        if routing_probs.shape[1] != self.num_experts:
            raise ValueError(
                f"routing_probs must have K={self.num_experts}, got "
                f"{routing_probs.shape[1]}."
            )
        if expert_indices.ndim != 2 or expert_indices.shape[0] != routing_probs.shape[0]:
            raise ValueError(
                "expert_indices must be [B, k_e] aligned with routing_probs, got "
                f"{tuple(expert_indices.shape)}."
            )

        batch_size = routing_probs.shape[0]
        reference_shape = None
        stacked_tokens = []
        for level, tokens in enumerate(level_tokens):
            if tokens.ndim != 3 or tokens.shape[0] != batch_size:
                raise ValueError(
                    f"level {level} tokens must be [B, P, C], got "
                    f"{tuple(tokens.shape)}."
                )
            if tokens.shape[2] != self.embed_dim:
                raise ValueError(
                    f"level {level} tokens must end with C={self.embed_dim}, got "
                    f"{tuple(tokens.shape)}."
                )
            if reference_shape is None:
                reference_shape = tokens.shape[1]
            elif tokens.shape[1] != reference_shape:
                raise ValueError(
                    "All levels must share the token grid P (their enhanced "
                    f"outputs are summed in Eq. 50); got P={reference_shape} "
                    f"and P={tokens.shape[1]}."
                )
            stacked_tokens.append(tokens)

        num_active = expert_indices.shape[1]

        # pi over the active set only: [B, k_e], sums to 1.
        active_probs = routing_probs.gather(1, expert_indices)
        # beta for the active experts: [B, k_e, L_levels], softmax over levels.
        beta = self.layer_scorer(level_pools, latent, expert_indices)

        # --- Run ONLY the active experts (gather-then-run, not loop-over-K).
        # Group samples by expert so each expert runs as a single batched call
        # on the samples routed to it — the standard sparse-MoE dispatch.
        flat_indices = expert_indices.reshape(-1)  # [B * k_e]
        # unique experts actually routed to in this batch (subset of K)
        routed_experts = torch.unique(flat_indices)

        enhanced_levels = []
        layer_weights = torch.stack(
            [(active_probs * beta[:, :, level]).sum(dim=1) for level in range(self.num_levels)],
            dim=1,
        )  # gamma_{b,l} = sum_k pi_{b,k} beta_{b,k,l}  ->  [B, L_levels]

        for level, tokens in enumerate(stacked_tokens):
            # Expert outputs only for the (sample, active-expert) pairs.
            # delta_j[b, j] = pi_{b,j} * beta_{b,j,l} * E_{K_b[b,j]}(tokens_b)
            delta = torch.zeros_like(tokens)  # [B, P, C]
            for expert_id in routed_experts.tolist():
                # (sample, slot) pairs routed to this expert
                pair_mask = (flat_indices == expert_id)
                slot_positions = pair_mask.nonzero(as_tuple=True)[0]
                if slot_positions.numel() == 0:
                    continue
                sample_ids = slot_positions // num_active
                slot_ids = slot_positions % num_active
                # One batched expert call for all pairs assigned to it.
                expert_out = self.experts.experts[int(expert_id)](
                    tokens[sample_ids]
                )  # [n_pairs, P, C]
                pair_weights = (
                    active_probs[sample_ids, slot_ids]
                    * beta[sample_ids, slot_ids, level]
                )  # [n_pairs]
                delta[sample_ids] += (
                    pair_weights.view(-1, 1, 1) * expert_out
                )
            enhanced_levels.append(tokens + delta)

        # --- Eq. 50: Z_fused = sum_l gamma_{b,l} * X_hat^(l).
        fused = None
        for level, enhanced in enumerate(enhanced_levels):
            term = layer_weights[:, level].view(batch_size, 1, 1) * enhanced
            fused = term if fused is None else fused + term

        return EnhancementOutput(
            fused_tokens=fused,
            enhanced_tokens=tuple(enhanced_levels),
            layer_weights=layer_weights,
            expert_layer_weights=beta,
        )
