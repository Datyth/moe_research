# Adapted from https://github.com/Asphyxiate-Rye/E-SAM
# segment_anything_ESAM/build_sam.py. Only the ViT-B builder is ported
# (matches this project's 256x256 pipeline); the top-level `LoRA_Sam` import
# was dropped (module doesn't exist upstream, never actually used).
#
# `num_multimask_outputs = num_classes - 1`, not `num_classes`: MaskDecoder
# always returns `num_multimask_outputs + 1` channels, and upstream treats
# `num_classes` as foreground-classes-plus-implicit-background. This
# project's BaseSegmentationModel instead requires exactly `num_classes`
# output channels (binary sigmoid or multiclass softmax) — see
# src/models/base.py.

from functools import partial

import torch
import torch.nn.functional as F

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer
from .sam_moe import Sam_my


def build_sam_vit_b(
    args,
    image_size: int,
    num_classes: int,
    checkpoint: str | None,
    moe_top_k_ratio: float,
    moe_num_experts: int,
    use_moe: bool = True,
    use_lpeg: bool = True,
) -> Sam_my:
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = (2, 5, 8, 11)
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam_my(
        args=args,
        image_encoder=ImageEncoderViT(
            args=args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            args=args,
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            use_lpeg=use_lpeg,
        ),
        mask_decoder=MaskDecoder(
            args=args,
            num_multimask_outputs=max(num_classes - 1, 0),
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        moe_top_k_ratio=moe_top_k_ratio,
        moe_num_experts=moe_num_experts,
        use_moe=use_moe,
        use_lpeg=use_lpeg,
        # Dataset pipeline already normalizes images; disable SAM's own
        # normalization to avoid doing it twice.
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    sam.train()

    if checkpoint is not None:
        with open(checkpoint, "rb") as file:
            state_dict = torch.load(file, map_location="cpu", weights_only=True)
        try:
            sam.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = _load_pretrained_backbone(
                sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes
            )
            sam.load_state_dict(new_state_dict, strict=False)

    return sam


def _load_pretrained_backbone(
    sam: Sam_my,
    state_dict: dict,
    image_size: int,
    vit_patch_size: int,
    encoder_global_attn_indexes,
) -> dict:
    """Load backbone/prompt-encoder/transformer weights; skip the decoder head
    (its shape depends on num_classes, so it stays randomly initialized)."""
    sam_dict = sam.state_dict()
    except_keys = ("mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head")
    new_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in sam_dict and not any(excluded in k for excluded in except_keys)
    }

    pos_embed = new_state_dict.get("image_encoder.pos_embed")
    token_size = image_size // vit_patch_size
    if pos_embed is not None and pos_embed.shape[1] != token_size:
        # Resize the absolute position embedding to the target resolution.
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict["image_encoder.pos_embed"] = pos_embed

        rel_pos_keys = [k for k in sam_dict if "rel_pos" in k]
        global_rel_pos_keys = [
            k for k in rel_pos_keys if int(k.split(".")[2]) in encoder_global_attn_indexes
        ]
        for key in global_rel_pos_keys:
            rel_pos_params = new_state_dict[key]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(
                rel_pos_params, (token_size * 2 - 1, w), mode="bilinear", align_corners=False
            )
            new_state_dict[key] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict
