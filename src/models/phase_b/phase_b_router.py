"""Phase B up to the Top-K router (no experts yet).

Extends the fuse stage: the backbone still produces the segmentation logits and
``h_q = [h_I ; h_M]`` exactly as before, and this stage adds the privileged
posterior, the deployable prior, and the shape-aware Top-K router on top —
recording the routing decision and its KL / load-balancing terms in the
diagnostics. It deliberately does **not** feed the routing back into any expert
or the mask decoder, so the segmentation output is bit-identical to the fuse
model; the stage that wires the routing into experts and the decoder is
``phase_b_moe`` (``PhaseBMoEStage``).

Two paths, decided by whether the ground-truth mask is available:

- **training (mask present):** posterior ``q(z|I,M)`` from ``h_q``; sample
  ``z_q`` (reparameterization) and route with it; ``L_latent = KL(sg[q] || p)``
  pulls the prior toward the posterior, and ``L_bal`` balances expert load.
- **inference (no mask):** prior ``p(z|I)`` from ``h_I`` only; route with the
  deterministic mean ``z = mu_p`` — no posterior, no mask dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from ..base import SegmentationOutput
from ..registry import register_model
from .fuse_stage import PhaseBFuseStage
from .image_descriptor import DEFAULT_LEVELS
from .posterior import DiagonalGaussian, GaussianParameterHead, gaussian_kl
from .router import RoutingOutput, TopKRouter, load_balance_loss


@dataclass
class RouterStageOutput:
    """The routing decision and its losses for one forward pass."""

    source: str
    """'posterior' (training, mask seen) or 'prior' (inference, image only)."""

    prior: DiagonalGaussian
    """p(z|I), always present."""

    latent: Tensor
    """The routing latent z actually used, shape [B, d_z]."""

    routing: RoutingOutput
    """The Top-K router output computed from `latent`."""

    posterior: DiagonalGaussian | None = None
    """q(z|I,M); None on the prior (inference) path."""

    latent_kl: Tensor | None = None
    """Scalar KL(sg[q] || p); None on the prior path."""

    balance: Tensor | None = None
    """Scalar L_bal; None on the prior path."""


class PhaseBRouterHead:
    """Posterior + prior + Top-K router, independent of the SAM backbone.

    Kept as a plain composable unit (not a subclass of the backbone model) so it
    can be exercised on synthetic ``h_q`` / ``h_I`` without instantiating ViT-B.
    Instantiate via :class:`PhaseBRouterModule` which owns it as a submodule.
    """

    def __init__(
        self,
        posterior_head: GaussianParameterHead,
        prior_head: GaussianParameterHead,
        router: TopKRouter,
    ) -> None:
        self.posterior_head = posterior_head
        self.prior_head = prior_head
        self.router = router

    def __call__(
        self,
        image_descriptor: Tensor,
        fused: Tensor | None,
        *,
        sample: bool,
    ) -> RouterStageOutput:
        prior = self.prior_head(image_descriptor)

        if fused is None:
            # Inference: no mask, so no posterior. Route on the prior mean.
            z = prior.mean
            return RouterStageOutput(
                source="prior",
                prior=prior,
                latent=z,
                routing=self.router(z),
            )

        posterior = self.posterior_head(fused)
        z = posterior.rsample() if sample else posterior.mean
        routing = self.router(z)
        # sg[q]: the posterior is the teacher; only the prior is pulled.
        latent_kl = gaussian_kl(posterior.detach(), prior).mean()
        balance = load_balance_loss(
            routing.dense_probs,
            routing.expert_indices,
            num_experts=self.router.num_experts,
        )
        return RouterStageOutput(
            source="posterior",
            prior=prior,
            latent=z,
            routing=routing,
            posterior=posterior,
            latent_kl=latent_kl,
            balance=balance,
        )


@register_model("phase_b_router")
class PhaseBRouterStage(PhaseBFuseStage):
    """Fuse stage + privileged posterior/prior + shape-aware Top-K router.

    Segmentation logits pass through from the backbone untouched; the routing
    machinery only writes a ``phase_b_router`` diagnostic. ``stochastic`` toggles
    the reparameterized sample (training) versus the deterministic ``z_q = mu_q``
    ablation from the proposal.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        task: str = "binary",
        image_size: int = 256,
        checkpoint: str | None = None,
        use_moe: bool = True,
        use_lpeg: bool = True,
        moe_num_experts: int = 4,
        moe_top_k_ratio: float = 0.5,
        freeze_backbone: bool = True,
        descriptor_dim: int = 256,
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        scoring_hidden_dim: int = 64,
        shape_teacher_checkpoint: str | Path | None = None,
        shape_teacher: dict[str, Any] | None = None,
        freeze_shape_teacher: bool = True,
        latent_dim: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        std_floor: float = 1e-4,
        stochastic: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
            image_size=image_size,
            checkpoint=checkpoint,
            use_moe=use_moe,
            use_lpeg=use_lpeg,
            moe_num_experts=moe_num_experts,
            moe_top_k_ratio=moe_top_k_ratio,
            freeze_backbone=freeze_backbone,
            descriptor_dim=descriptor_dim,
            levels=levels,
            scoring_hidden_dim=scoring_hidden_dim,
            shape_teacher_checkpoint=shape_teacher_checkpoint,
            shape_teacher=shape_teacher,
            freeze_shape_teacher=freeze_shape_teacher,
        )

        self.stochastic = bool(stochastic)
        # Posterior reads h_q = [h_I ; h_M]; prior reads h_I alone.
        self.posterior_head = GaussianParameterHead(
            in_dim=self.fused_dim, latent_dim=latent_dim, std_floor=std_floor
        )
        self.prior_head = GaussianParameterHead(
            in_dim=self.fusion.descriptor_dim, latent_dim=latent_dim, std_floor=std_floor
        )
        self.router = TopKRouter(
            latent_dim=latent_dim,
            num_experts=num_experts,
            active_experts=active_experts,
        )
        self._head = PhaseBRouterHead(self.posterior_head, self.prior_head, self.router)

    @property
    def latent_dim(self) -> int:
        return self.router.latent_dim

    def forward(
        self,
        images: Tensor,
        *,
        masks: Tensor | None = None,
        **kwargs,
    ) -> SegmentationOutput:
        output = super().forward(images, masks=masks, **kwargs)
        diagnostics = output.diagnostics

        image_descriptor = diagnostics["image_descriptor"]
        fuse_stage = diagnostics.get("fuse_stage")
        fused = fuse_stage.fused if (masks is not None and fuse_stage is not None) else None

        diagnostics["phase_b_router"] = self._head(
            image_descriptor,
            fused,
            sample=self.training and self.stochastic,
        )
        return output
