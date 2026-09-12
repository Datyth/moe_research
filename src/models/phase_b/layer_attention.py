"""Hierarchical layer-preference scoring for Phase B (proposal Eqs. 40-44).

Where the fuse stage's ``g_level`` asks *"how relevant is level l to this
image?"* (one weight per level, expert-agnostic, ``alpha_l``), ``g_layer``
here asks *"how relevant is level l **to expert k** on **this sample**?"*.

Per expert k and level l, a score

    a_{b,k,l} = g_layer([ v^(l)_b ; z_b ; e_k ])

is computed from the globally pooled ViT level features ``v^(l) = GAP(F^(l))``
(raw, 768-d), the routing latent ``z_b``, and the learnable expert embedding
``e_k``; a softmax over levels yields the per-expert layer preference
``beta_{b,k,l}``. A sample routed to experts {0, 3} may therefore draw its
shape evidence from different levels per expert — early texture for one,
late semantics for the other — which is the "hierarchical" part of the
enhancement: level choice is delegated to each expert, not fixed globally.

The scorer is a single shared MLP over the concatenated argument, exactly like
``g_level`` in ``image_descriptor.py`` is shared across levels. Both v and e
live in the same C-dimensional token space; z is appended as the
sample-conditioning term.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


DEFAULT_SCORING_HIDDEN_RATIO = 12


class LayerPreferenceScorer(nn.Module):
    """g_layer: score [v^(l) ; z ; e_k] -> a_{b,k,l} (Eqs. 40-43)."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        latent_dim: int = 64,
        num_experts: int = 4,
        num_levels: int = 4,
        expert_embedding_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.num_levels = num_levels
        self.expert_embedding_dim = (
            embed_dim if expert_embedding_dim is None else expert_embedding_dim
        )
        in_dim = embed_dim + latent_dim + self.expert_embedding_dim
        hidden = (
            embed_dim // DEFAULT_SCORING_HIDDEN_RATIO
            if hidden_dim is None
            else hidden_dim
        )
        # One shared MLP scoring one (sample, expert, level) triple at a time;
        # a per-pair scorer could not be compared on a common scale, same
        # argument as g_level's sharing across levels.
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # e_k (Eq. 41): one learnable identity per expert, initialized small
        # so at init the level preferences are driven by the data terms.
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, self.expert_embedding_dim) * 0.02)

    def forward(
        self,
        level_pools: Tensor,
        latent: Tensor,
        expert_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute per-expert, per-level preferences beta_{b,k,l}.

        Args:
            level_pools: stacked v^(l), shape [B, L_levels, C] (raw ViT dims;
                the GAP of each level's tokens X^(l)).
            latent: routing latent z, shape [B, d_z].
            expert_indices: K_b, shape [B, k_e]. When None, all K experts are
                scored (used by tests and the dense ablation).

        Returns:
            beta, shape [B, K, L_levels] when expert_indices is None, else
            [B, k_e, L_levels] aligned with expert_indices — softmax over the
            level axis in both cases.
        """

        if level_pools.ndim != 3:
            raise ValueError(
                "level_pools must be [B, L_levels, C], got "
                f"{tuple(level_pools.shape)}."
            )
        if level_pools.shape[1] != self.num_levels:
            raise ValueError(
                f"level_pools must have L_levels={self.num_levels}, got "
                f"{level_pools.shape[1]}."
            )
        if level_pools.shape[2] != self.embed_dim:
            raise ValueError(
                f"level_pools must end with C={self.embed_dim}, got "
                f"{tuple(level_pools.shape)}."
            )
        if latent.ndim != 2:
            raise ValueError(f"latent must be [B, d_z], got {tuple(latent.shape)}.")
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have d_z={self.latent_dim}, got {tuple(latent.shape)}."
            )
        if latent.shape[0] != level_pools.shape[0]:
            raise ValueError(
                "level_pools and latent must share a batch size, got "
                f"{level_pools.shape[0]} and {latent.shape[0]}."
            )
        if expert_indices is not None:
            if expert_indices.ndim != 2:
                raise ValueError(
                    "expert_indices must be [B, k_e], got "
                    f"{tuple(expert_indices.shape)}."
            )
            if expert_indices.shape[0] != level_pools.shape[0]:
                raise ValueError(
                    "expert_indices must share the batch size, got "
                    f"{tuple(expert_indices.shape)}."
                )

        batch_size = level_pools.shape[0]

        if expert_indices is None:
            # [B, K, 1, C] + [B, 1, L, C] + [B, K, 1, d_z] -> [B, K, L, in]:
            # the concatenation layout matches the docstring order [v ; z ; e].
            v = level_pools.unsqueeze(1).expand(
                batch_size, self.num_experts, self.num_levels, self.embed_dim
            )
            z = latent.unsqueeze(1).unsqueeze(2).expand(
                batch_size, self.num_experts, self.num_levels, self.latent_dim
            )
            e = self.expert_embeddings.view(
                1, self.num_experts, 1, self.expert_embedding_dim
            ).expand(
                batch_size, self.num_experts, self.num_levels,
                self.expert_embedding_dim,
            )
            scores = self.scorer(torch.cat([v, z, e], dim=-1)).squeeze(-1)
        else:
            num_active = expert_indices.shape[1]
            # [B, k_e, L, in]: only the routed experts are scored.
            v = level_pools.unsqueeze(1).expand(
                batch_size, num_active, self.num_levels, self.embed_dim
            )
            z = latent.unsqueeze(1).unsqueeze(2).expand(
                batch_size, num_active, self.num_levels, self.latent_dim
            )
            e = self.expert_embeddings[expert_indices].unsqueeze(2).expand(
                batch_size, num_active, self.num_levels, self.expert_embedding_dim
            )
            scores = self.scorer(torch.cat([v, z, e], dim=-1)).squeeze(-1)

        return torch.softmax(scores, dim=-1)
