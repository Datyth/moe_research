# Vendored from https://github.com/Asphyxiate-Rye/E-SAM
# segment_anything_ESAM/modeling/sam_my.py (class `Sam_my`). Dropped dead
# imports (SwinUNETR, unused MoE wildcard, two modules that don't exist
# upstream), unused `self.moe_encoder`/`self.norm`/`CrossAttention`. Reused
# `Attention` from transformer.py instead of a duplicate inline class.
# `top_k` -> `top_k_ratio`: see moe.py.
#
# Added `use_moe` (not upstream): when False, skips MoE routing/fusion and
# decodes the encoder's `image_embeddings` directly — an ablation baseline
# with everything else identical.

from typing import Any, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import Attention
from .moe import ExpertChoiceTokenSparseMoE
from .common import LayerNorm2d


class Sam_my(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        moe_top_k_ratio: float,
        moe_num_experts: int = 4,
        use_moe: bool = True,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.args = args
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.use_moe = use_moe
        if use_moe:
            self.ExpertChoiceTokenMoE = ExpertChoiceTokenSparseMoE(
                n_embed=self.image_encoder.embed_dim,
                num_experts=moe_num_experts,
                top_k_ratio=moe_top_k_ratio,
            )
            self.neck5 = nn.Sequential(
                nn.Conv2d(768, 256, kernel_size=1, bias=False),
                LayerNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(256),
            )
            self.attn = Attention(768, 8)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size, gt=None, mode='train'):
        input_images = self.preprocess(batched_input)
        image_embeddings, low_image_embeddings = self.image_encoder(input_images)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=None, masks=None, image_embedding=image_embeddings
        )

        indices = None
        if self.use_moe:
            embedding_moe = torch.cat(
                [embed.unsqueeze(1) for embed in low_image_embeddings], dim=1
            ).permute(0, 1, 4, 2, 3).contiguous()

            embedding_moe = embedding_moe.permute(0, 1, 3, 4, 2).contiguous()
            bs, num_features, h, w, dim = embedding_moe.shape
            embedding_moe, indices = self.ExpertChoiceTokenMoE(embedding_moe.reshape(bs * num_features, h * w, dim))
            embedding_moe = embedding_moe.reshape(bs, -1, dim)
            embedding_moe = self.attn(embedding_moe, embedding_moe, embedding_moe).reshape(
                bs, num_features, h, w, dim
            ).permute(0, 1, 4, 2, 3).contiguous()
            embedding_moe = embedding_moe.mean(1)
            image_embeddings = image_embeddings + self.neck5(embedding_moe)

        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        outputs = {
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
                "indices": indices}

        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
