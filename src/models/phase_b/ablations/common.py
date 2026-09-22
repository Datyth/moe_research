"""Shared, stateless building blocks for Phase B controlled ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...base import SegmentationOutput
from ...esam import EsamModel
from ...esam._vendor.common import LayerNorm2d


@dataclass
class AblationEnhancementOutput:
    """Variant-independent enhancement diagnostics, all per sample."""

    layer_weights: Tensor
    aux_norm_ratio: Tensor
    fused_token_norm: Tensor
    enhanced_token_norm: Tensor | None = None


def build_enhancement_neck(embed_dim: int, output_dim: int) -> nn.Sequential:
    """Build the exact 768→256 neck used by the A0 PhaseBMoEStage."""

    return nn.Sequential(
        nn.Conv2d(embed_dim, output_dim, kernel_size=1, bias=False),
        LayerNorm2d(output_dim),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(output_dim),
    )


def run_sam_encoder(backbone: EsamModel, images: Tensor) -> tuple[Tensor, list[Tensor]]:
    """Run SAM preprocessing and encoder without invoking its decoder/MoE-FEB."""

    network = backbone.network
    input_images = network.preprocess(images)
    return network.image_encoder(input_images)


def fuse_level_tokens(
    level_tokens: tuple[Tensor, ...],
    level_weights: Tensor,
) -> Tensor:
    """Return ``sum_l weight_l X^(l)`` with shape [B, P, C]."""

    if level_weights.ndim != 2:
        raise ValueError(
            "level_weights must be [B, L], got "
            f"{tuple(level_weights.shape)}."
        )
    batch_size, num_levels = level_weights.shape
    if num_levels == 0:
        raise ValueError("At least one level token is required.")
    if len(level_tokens) != num_levels:
        raise ValueError(
            f"level_weights has L={num_levels}, but received "
            f"{len(level_tokens)} token tensors."
        )

    reference_shape = None
    for level, tokens in enumerate(level_tokens):
        if tokens.ndim != 3 or tokens.shape[0] != batch_size:
            raise ValueError(
                f"level {level} tokens must be [B, P, C], got "
                f"{tuple(tokens.shape)}."
            )
        if reference_shape is None:
            reference_shape = tokens.shape[1:]
        elif tokens.shape[1:] != reference_shape:
            raise ValueError(
                "All level tokens must share [P, C]; got "
                f"{tuple(reference_shape)} and {tuple(tokens.shape[1:])}."
            )

    stacked_tokens = torch.stack(level_tokens, dim=1)
    return (level_weights[:, :, None, None] * stacked_tokens).sum(dim=1)


def tokens_to_spatial(
    tokens: Tensor,
    image_embeddings: Tensor,
    *,
    embed_dim: int,
) -> Tensor:
    """Reshape [B,P,C] tokens to the final SAM patch grid."""

    if tokens.ndim != 3 or tokens.shape[2] != embed_dim:
        raise ValueError(
            f"tokens must be [B, P, {embed_dim}], got {tuple(tokens.shape)}."
        )
    if image_embeddings.ndim != 4:
        raise ValueError(
            "image_embeddings must be [B, C, H, W], got "
            f"{tuple(image_embeddings.shape)}."
        )
    batch_size = image_embeddings.shape[0]
    if tokens.shape[0] != batch_size:
        raise ValueError("tokens and image_embeddings must share batch size.")
    height, width = image_embeddings.shape[-2:]
    if tokens.shape[1] != height * width:
        raise ValueError(
            "Token grid does not match final SAM embedding: "
            f"P={tokens.shape[1]} versus H*W={height * width}."
        )
    return tokens.transpose(1, 2).reshape(batch_size, embed_dim, height, width)


def inject_enhancement(
    image_embeddings: Tensor,
    tokens: Tensor,
    enhancement_neck: nn.Module,
    *,
    embed_dim: int,
) -> tuple[Tensor, Tensor]:
    """Apply the enhancement neck and residual injection."""

    spatial = tokens_to_spatial(tokens, image_embeddings, embed_dim=embed_dim)
    e_aux = enhancement_neck(spatial)
    if e_aux.shape != image_embeddings.shape:
        raise ValueError(
            "enhancement_neck output must match final SAM embedding, got "
            f"{tuple(e_aux.shape)} and {tuple(image_embeddings.shape)}."
        )
    return e_aux, image_embeddings + e_aux


def decode_sam(
    backbone: EsamModel,
    enhanced_embeddings: Tensor,
    *,
    image_size: int,
) -> tuple[Tensor, Tensor]:
    """Run the unchanged LPEG prompt encoder and SAM mask decoder."""

    network = backbone.network
    batch_size = enhanced_embeddings.shape[0]
    sparse_embeddings, dense_embeddings = network.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
        image_embedding=enhanced_embeddings if network.use_lpeg else None,
        batch_size=batch_size,
    )
    low_res_masks, iou_predictions = network.mask_decoder(
        image_embeddings=enhanced_embeddings,
        image_pe=network.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    masks = network.postprocess_masks(
        low_res_masks,
        input_size=(image_size, image_size),
        original_size=(image_size, image_size),
    )
    return masks, iou_predictions


def token_norm(tokens: Tensor) -> Tensor:
    """Per-sample flattened L2 token norm."""

    return tokens.flatten(1).norm(dim=1).detach()


def enhanced_level_norm(enhanced_tokens: tuple[Tensor, ...]) -> Tensor | None:
    """Mean per-level L2 norm, or None when no expert-enhanced levels exist."""

    if not enhanced_tokens:
        return None
    return torch.stack(
        [tokens.flatten(1).norm(dim=1) for tokens in enhanced_tokens], dim=1
    ).mean(dim=1).detach()


def make_segmentation_output(
    *,
    backbone: EsamModel,
    enhanced_embeddings: Tensor,
    image_size: int,
    diagnostics: dict,
) -> SegmentationOutput:
    """Decode and attach common IoU diagnostics."""

    logits, iou_predictions = decode_sam(
        backbone,
        enhanced_embeddings,
        image_size=image_size,
    )
    diagnostics["iou_predictions"] = iou_predictions
    return SegmentationOutput(logits=logits, diagnostics=diagnostics)
