"""Shape-aware sparse Top-K router (proposal Phase B; ShapeMoE eqs. 3-4).

The router is **instance-level**: one routing decision per image, driven by the
routing latent ``z in R^{d_z}`` sampled from the posterior (training) or taken as
the prior mean (inference). This is a different mechanism from the token-level
``ExpertChoiceTokenSparseMoE`` inside the E-SAM encoder (MoE-FEB) — that one
routes patch tokens; this one selects which shape experts an object activates.

    r      = W_R z + b_R              in R^{B x K}      (logits over experts)
    pi_bar = Softmax(r)                                 (dense probabilities)
    K_b    = TopK(pi_bar_b, k_e)                        (active expert indices)
    pi     = renormalize(pi_bar over K_b)               (sparse, sums to 1)

Only the ``k_e`` selected experts carry non-zero probability; the rest are zero.
This module stops at producing ``pi`` and ``K_b``; the enhancement stage
(``moe_enhancement``) is what consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RoutingOutput:
    """One sparse routing decision per sample."""

    logits: Tensor
    """r, shape [B, K]."""

    dense_probs: Tensor
    """pi_bar = Softmax(r), shape [B, K]; the pre-Top-K distribution the
    load-balancing loss and (later) routing distillation read."""

    routing_probs: Tensor
    """pi, shape [B, K]; zero off the selected set, sums to 1 over it."""

    expert_indices: Tensor
    """K_b, shape [B, k_e] (long); the active experts per sample."""


class TopKRouter(nn.Module):
    """Linear expert scorer with Top-K sparsification and renormalization."""

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= active_experts <= num_experts:
            raise ValueError(
                f"active_experts must be in [1, num_experts={num_experts}], "
                f"got {active_experts}."
            )
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        # r = W_R z + b_R, exactly the proposal's router logits.
        self.route = nn.Linear(latent_dim, num_experts)

    def forward(self, z: Tensor) -> RoutingOutput:
        if z.ndim != 2:
            raise ValueError(f"z must be [B, d_z], got {tuple(z.shape)}.")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must have latent_dim={self.latent_dim}, got {z.shape[1]}."
            )

        logits = self.route(z)
        dense_probs = torch.softmax(logits, dim=1)

        # Select on the dense probabilities and renormalize over the chosen set,
        # rather than re-softmaxing the top-k logits: the proposal defines
        # pi_{b,k} = pi_bar_{b,k} / sum_{j in K_b} pi_bar_{b,j}.
        top_probs, expert_indices = dense_probs.topk(self.active_experts, dim=1)
        renormalized = top_probs / top_probs.sum(dim=1, keepdim=True)
        routing_probs = torch.zeros_like(dense_probs).scatter(
            1, expert_indices, renormalized
        )

        return RoutingOutput(
            logits=logits,
            dense_probs=dense_probs,
            routing_probs=routing_probs,
            expert_indices=expert_indices,
        )


def load_balance_loss(
    dense_probs: Tensor,
    expert_indices: Tensor,
    *,
    num_experts: int,
) -> Tensor:
    """Proposal's routing-balance loss ``L_bal = K * sum_k f_k * P_k`` (scalar).

    ``f_k`` is the fraction of active-expert *slots* assigned to expert k across
    the batch, and ``P_k`` its mean dense probability. The product is minimized
    when load is spread evenly, so this discourages routing collapse onto a few
    experts. This is the switch-transformer-style balance the proposal specifies —
    deliberately not ShapeMoE's CV^2 loss.
    """

    if dense_probs.ndim != 2:
        raise ValueError(
            f"dense_probs must be [B, K], got {tuple(dense_probs.shape)}."
        )
    if dense_probs.shape[1] != num_experts:
        raise ValueError(
            f"dense_probs must have K={num_experts}, got {dense_probs.shape[1]}."
        )
    if expert_indices.ndim != 2 or expert_indices.shape[0] != dense_probs.shape[0]:
        raise ValueError(
            "expert_indices must be [B, k_e] aligned with dense_probs, got "
            f"{tuple(expert_indices.shape)}."
        )

    batch_size, active_experts = expert_indices.shape
    # f_k: assignment count per expert, normalized by total slots B * k_e.
    one_hot = torch.zeros_like(dense_probs).scatter(
        1, expert_indices, torch.ones_like(dense_probs)
    )
    fraction = one_hot.sum(dim=0) / (batch_size * active_experts)
    mean_prob = dense_probs.mean(dim=0)
    return num_experts * torch.sum(fraction * mean_prob)
