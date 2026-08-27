"""ShapeMoE segmenter: shared trunk, shape posterior, sparse router, experts.

This assembles Sec. 3.3-3.5 of the paper on top of the UNet already in this
repository, standing in for the SAM stack the paper uses:

    images -> UNet encoder  -> bottleneck   -> E_S     -> mu_S, logvar_S
                            -> decoder F        |
                                                v
                                        router (Eq. 2-4) -> pi
                                                |
                                                v
                                    expert mask heads(F, pi) -> logits

Stage (1) of Sec. 3.2, the SAM image feature encoder plus the mask embedding
encoder E_M, has no counterpart here: the UNet trunk plays the role of the image
encoder, and there is no visible-mask prompt to embed. The student therefore
derives its shape posterior from image features alone, which is what keeps it
free of the ground-truth mask that the Phase 0 teacher is allowed to see.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ..base import BaseSegmentationModel, SegmentationOutput, TaskType
from ..registry import register_model
from ..unet import UNet
from .experts import ExpertMaskHeads
from .router import ShapeAwareSparseRouter
from .student import ShapeDistributionEncoder


@register_model("shapemoe_unet")
class ShapeMoESegmenter(BaseSegmentationModel):
    """Sparse shape-routed segmentation model with a UNet trunk."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        task: TaskType = "binary",
        base_channels: int = 32,
        latent_dim: int = 64,
        num_experts: int = 4,
        top_k: int = 1,
        shape_encoder_hidden_dim: int | None = None,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            task=task,
        )

        self.trunk = UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            base_channels=base_channels,
        )
        # The expert heads replace the trunk's single output projection, so drop
        # it rather than carry untrained parameters in every checkpoint.
        self.trunk.out_conv = nn.Identity()

        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.base_channels = base_channels

        self.shape_encoder = ShapeDistributionEncoder(
            in_channels=base_channels * 16,
            latent_dim=latent_dim,
            hidden_dim=shape_encoder_hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        self.router = ShapeAwareSparseRouter(
            latent_dim=latent_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.experts = ExpertMaskHeads(
            in_channels=base_channels,
            num_classes=num_classes,
            num_experts=num_experts,
        )

    def forward(
        self,
        images: Tensor,
        *,
        sample: bool | None = None,
        **kwargs: Any,
    ) -> SegmentationOutput:
        """Route each image by its predicted shape and segment it.

        ``sample`` controls Eq. (2). It defaults to the module's training flag:
        stochastic while training, deterministic at evaluation time so that a
        checkpoint gives the same mask twice. The paper does not discuss this.
        """

        bottleneck, features = self.trunk.forward_features(images)
        mu, logvar = self.shape_encoder(bottleneck)
        routing = self.router(
            mu,
            logvar,
            sample=self.training if sample is None else sample,
        )
        logits = self.experts(features, routing.pi)

        return SegmentationOutput(
            logits=logits,
            diagnostics={
                "mu": mu,
                "logvar": logvar,
                "latent": routing.latent,
                "pi": routing.pi,
                "router_probabilities": routing.probabilities,
                "expert_index": routing.indices,
            },
        )

    def model_config(self) -> dict[str, Any]:
        """Serializable constructor arguments, stored inside checkpoints."""

        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "task": self.task,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }
