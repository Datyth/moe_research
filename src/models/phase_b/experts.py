"""Lightweight feed-forward experts for Phase B hierarchical enhancement.

Each expert ``E_k`` (proposal Eq. 45) is a pre-norm FFN over the ViT spatial
tokens ``X^(l) in R^{B x P x C}``:

    E_k(X) = W_{2,k} GELU(W_{1,k} LayerNorm(X))

with linear projections ``C -> m*C -> C`` (default ``768 -> 3072 -> 768``,
``m=4`` matching the MLP ratio of the ViT blocks the tokens come from). The
experts are deliberately token-position-wise — the same affine map applied to
every patch token — so an expert is a shape/texture *filter*, not a spatial
transformer. This mirrors the vendored MoE-FEB ``Expert``
(``esam/_vendor/moe.py``) in role but differs in activation and normalization:
pre-LayerNorm + GELU here, following the proposal, versus ReLU + dropout there.

Only the experts selected by the instance-level router (``k in K_b``, size
``k_e``) are evaluated per sample — see ``moe_enhancement`` — so the full
``nn.ModuleList`` of K experts is capacity, not per-pass cost.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


DEFAULT_EXPERT_HIDDEN_RATIO = 4


class FeedForwardExpert(nn.Module):
    """One shape expert: LayerNorm -> W1 -> GELU -> W2 (Eq. 45)."""

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        hidden_ratio: int = DEFAULT_EXPERT_HIDDEN_RATIO,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive.")
        self.embed_dim = embed_dim
        self.hidden_ratio = hidden_ratio
        hidden_dim = hidden_ratio * embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Map [B, P, C] (or [B, ..., C]) tokens through the expert."""

        if tokens.ndim < 2:
            raise ValueError(
                f"tokens must be at least [B, C], got {tuple(tokens.shape)}."
            )
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"tokens must end with C={self.embed_dim}, got {tuple(tokens.shape)}."
            )
        return self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(tokens))))


class ExpertBank(nn.Module):
    """The K per-shape experts ``{E_k}`` (proposal Section 1.3).

    A plain ``nn.ModuleList`` behind a bank name: forward-time selection and
    gathering is the enhancement module's job, because which experts run is a
    *per-sample* routing decision, not a module-level one. The learnable
    expert identities ``e_k`` that ``g_layer`` consumes live in the scorer
    (``LayerPreferenceScorer.expert_embeddings``), not here — the bank is
    purely the K expert FFNs.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        num_experts: int = 4,
        hidden_ratio: int = DEFAULT_EXPERT_HIDDEN_RATIO,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.hidden_ratio = hidden_ratio
        self.experts = nn.ModuleList(
            FeedForwardExpert(embed_dim=embed_dim, hidden_ratio=hidden_ratio)
            for _ in range(num_experts)
        )

    def forward(self, index: int, tokens: Tensor) -> Tensor:
        """Run the single expert `index` — convenience for tests/ablations."""

        if not 0 <= index < self.num_experts:
            raise ValueError(f"index must be in [0, {self.num_experts}), got {index}.")
        return self.experts[index](tokens)
