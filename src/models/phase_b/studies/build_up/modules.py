"""Composable enhancement and conditioning modules for build-up B1-B6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...experts import ExpertBank
from ...posterior import DiagonalGaussian, GaussianParameterHead
from ...router import RoutingOutput, TopKRouter, load_balance_loss


@dataclass
class GlobalEnhancementOutput:
    """Token output shared by the global-alpha B1-B5 variants."""

    fused_tokens: Tensor
    enhanced_tokens: tuple[Tensor, ...]


@dataclass
class ConditioningOutput:
    """Routing state without any Phase-C prior or distillation state."""

    source: str
    latent: Tensor
    routing: RoutingOutput
    balance: Tensor
    posterior: DiagonalGaussian | None = None
    latent_kl: None = None


def validate_level_tokens(
    level_tokens: tuple[Tensor, ...],
    level_weights: Tensor,
    *,
    embed_dim: int,
    num_levels: int,
) -> tuple[int, int]:
    """Validate the common [B,L,P,C] contract and return B,P."""

    if len(level_tokens) != num_levels:
        raise ValueError(
            f"expected {num_levels} level token tensors, got {len(level_tokens)}."
        )
    if level_weights.ndim != 2 or level_weights.shape[1] != num_levels:
        raise ValueError(
            f"level_weights must be [B, {num_levels}], got "
            f"{tuple(level_weights.shape)}."
        )
    batch_size = level_weights.shape[0]
    token_count: int | None = None
    for level, tokens in enumerate(level_tokens):
        if tokens.ndim != 3 or tokens.shape[0] != batch_size:
            raise ValueError(
                f"level {level} tokens must be [B, P, C], got "
                f"{tuple(tokens.shape)}."
            )
        if tokens.shape[2] != embed_dim:
            raise ValueError(
                f"level {level} tokens must end with C={embed_dim}, got "
                f"{tuple(tokens.shape)}."
            )
        if token_count is None:
            token_count = tokens.shape[1]
        elif tokens.shape[1] != token_count:
            raise ValueError("All selected levels must share token count P.")
    return batch_size, int(token_count or 0)


def fuse_level_tokens(
    level_tokens: tuple[Tensor, ...],
    level_weights: Tensor,
) -> Tensor:
    """Compute the per-sample weighted sum over levels."""

    stacked = torch.stack(level_tokens, dim=1)
    return (level_weights[:, :, None, None] * stacked).sum(dim=1)


class IdentityEnhancer(nn.Module):
    """B1: no token transform; fuse raw levels with global alpha."""

    def __init__(self, *, embed_dim: int, num_levels: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels

    def forward(
        self,
        level_tokens: tuple[Tensor, ...],
        level_weights: Tensor,
    ) -> GlobalEnhancementOutput:
        validate_level_tokens(
            level_tokens,
            level_weights,
            embed_dim=self.embed_dim,
            num_levels=self.num_levels,
        )
        return GlobalEnhancementOutput(
            fused_tokens=fuse_level_tokens(level_tokens, level_weights),
            enhanced_tokens=level_tokens,
        )


class DenseTokenFFN(nn.Module):
    """Shared pre-norm dense FFN used by the parameter-matched B2 control."""

    def __init__(self, *, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive.")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    @staticmethod
    def parameter_count(*, embed_dim: int, hidden_dim: int) -> int:
        """LN + two biased linear layers."""

        return hidden_dim * (2 * embed_dim + 1) + 3 * embed_dim

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim < 2 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"tokens must end with C={self.embed_dim}, got {tuple(tokens.shape)}."
            )
        return self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(tokens))))


def nearest_dense_hidden_dim(*, embed_dim: int, target_parameters: int) -> int:
    """Return the integer hidden width closest to ``target_parameters``."""

    if embed_dim <= 0 or target_parameters <= 0:
        raise ValueError("embed_dim and target_parameters must be positive.")
    denominator = 2 * embed_dim + 1
    estimate = max(1, round((target_parameters - 3 * embed_dim) / denominator))
    candidates = {max(1, estimate - 1), estimate, estimate + 1}
    return min(
        candidates,
        key=lambda width: (
            abs(
                DenseTokenFFN.parameter_count(
                    embed_dim=embed_dim,
                    hidden_dim=width,
                )
                - target_parameters
            ),
            width,
        ),
    )


class DenseEnhancer(nn.Module):
    """B2: apply one shared dense residual FFN to every selected level."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_levels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.dense = DenseTokenFFN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        level_tokens: tuple[Tensor, ...],
        level_weights: Tensor,
    ) -> GlobalEnhancementOutput:
        validate_level_tokens(
            level_tokens,
            level_weights,
            embed_dim=self.embed_dim,
            num_levels=self.num_levels,
        )
        enhanced = tuple(tokens + self.dense(tokens) for tokens in level_tokens)
        return GlobalEnhancementOutput(
            fused_tokens=fuse_level_tokens(enhanced, level_weights),
            enhanced_tokens=enhanced,
        )


class SparseMoEEnhancer(nn.Module):
    """B3-B5: selected experts transform every level, then alpha fuses."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_levels: int,
        num_experts: int,
        expert_hidden_ratio: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_experts = num_experts
        self.experts = ExpertBank(
            embed_dim=embed_dim,
            num_experts=num_experts,
            hidden_ratio=expert_hidden_ratio,
        )

    def forward(
        self,
        level_tokens: tuple[Tensor, ...],
        level_weights: Tensor,
        routing_probs: Tensor,
        expert_indices: Tensor,
    ) -> GlobalEnhancementOutput:
        batch_size, _ = validate_level_tokens(
            level_tokens,
            level_weights,
            embed_dim=self.embed_dim,
            num_levels=self.num_levels,
        )
        if routing_probs.shape != (batch_size, self.num_experts):
            raise ValueError(
                f"routing_probs must be [B, {self.num_experts}], got "
                f"{tuple(routing_probs.shape)}."
            )
        if (
            expert_indices.ndim != 2
            or expert_indices.shape[0] != batch_size
            or expert_indices.shape[1] == 0
            or expert_indices.dtype != torch.long
        ):
            raise ValueError("expert_indices must be a non-empty long [B, k_e] tensor.")
        if bool(
            ((expert_indices < 0) | (expert_indices >= self.num_experts)).any()
        ):
            raise ValueError("expert_indices contains an out-of-range expert.")

        active_probs = routing_probs.gather(1, expert_indices)
        num_active = expert_indices.shape[1]
        flat_indices = expert_indices.reshape(-1)
        routed_experts = torch.unique(flat_indices)
        enhanced_levels: list[Tensor] = []
        for tokens in level_tokens:
            delta = torch.zeros_like(tokens)
            for expert_id in routed_experts.tolist():
                slot_positions = (flat_indices == expert_id).nonzero(
                    as_tuple=True
                )[0]
                sample_ids = slot_positions // num_active
                slot_ids = slot_positions % num_active
                expert_output = self.experts.experts[int(expert_id)](
                    tokens[sample_ids]
                )
                weights = active_probs[sample_ids, slot_ids]
                delta[sample_ids] += weights[:, None, None] * expert_output
            enhanced_levels.append(tokens + delta)

        enhanced = tuple(enhanced_levels)
        return GlobalEnhancementOutput(
            fused_tokens=fuse_level_tokens(enhanced, level_weights),
            enhanced_tokens=enhanced,
        )


class RandomTopKRouter(nn.Module):
    """Uniform random subset router with uniform selected-expert weights."""

    def __init__(
        self,
        *,
        latent_dim: int,
        num_experts: int,
        active_experts: int,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or num_experts <= 0:
            raise ValueError("latent_dim and num_experts must be positive.")
        if not 1 <= active_experts <= num_experts:
            raise ValueError("active_experts must be in [1, num_experts].")
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.active_experts = active_experts

    def forward(self, latent: Tensor) -> RoutingOutput:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must be [B, {self.latent_dim}], got {tuple(latent.shape)}."
            )
        batch_size = latent.shape[0]
        scores = torch.rand(
            batch_size,
            self.num_experts,
            device=latent.device,
            dtype=latent.dtype,
        )
        expert_indices = scores.topk(self.active_experts, dim=1).indices
        dense_probs = torch.full_like(scores, 1.0 / self.num_experts)
        routing_probs = torch.zeros_like(scores).scatter(
            1,
            expert_indices,
            torch.full_like(expert_indices, 1.0 / self.active_experts, dtype=latent.dtype),
        )
        return RoutingOutput(
            logits=torch.zeros_like(scores),
            dense_probs=dense_probs,
            routing_probs=routing_probs,
            expert_indices=expert_indices,
        )


class DirectConditioner(nn.Module):
    """Route a descriptor directly, without a Gaussian bottleneck."""

    def __init__(self, *, router: TopKRouter | RandomTopKRouter, source: str) -> None:
        super().__init__()
        self.router = router
        self.source = source

    def forward(self, descriptor: Tensor) -> ConditioningOutput:
        routing = self.router(descriptor)
        balance = load_balance_loss(
            routing.dense_probs,
            routing.expert_indices,
            num_experts=self.router.num_experts,
        )
        return ConditioningOutput(
            source=self.source,
            latent=descriptor,
            routing=routing,
            balance=balance,
        )


class GaussianConditioner(nn.Module):
    """Posterior-only Gaussian routing used by B5/B6."""

    def __init__(
        self,
        *,
        in_dim: int,
        latent_dim: int,
        num_experts: int,
        active_experts: int,
        std_floor: float,
        stochastic: bool,
    ) -> None:
        super().__init__()
        self.posterior_head = GaussianParameterHead(
            in_dim=in_dim,
            latent_dim=latent_dim,
            std_floor=std_floor,
        )
        self.router = TopKRouter(
            latent_dim=latent_dim,
            num_experts=num_experts,
            active_experts=active_experts,
        )
        self.stochastic = bool(stochastic)

    def forward(self, descriptor: Tensor) -> ConditioningOutput:
        posterior = self.posterior_head(descriptor)
        latent = (
            posterior.rsample()
            if self.training and self.stochastic
            else posterior.mean
        )
        routing = self.router(latent)
        balance = load_balance_loss(
            routing.dense_probs,
            routing.expert_indices,
            num_experts=self.router.num_experts,
        )
        return ConditioningOutput(
            source="posterior",
            latent=latent,
            routing=routing,
            balance=balance,
            posterior=posterior,
        )
