"""Unit tests for stateless Phase B ablation helpers."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.models.phase_b.ablations.common import (
    decode_sam,
    fuse_level_tokens,
    inject_enhancement,
    tokens_to_spatial,
)


class _PromptEncoder(nn.Module):
    def forward(self, **kwargs):
        batch_size = kwargs["batch_size"]
        return torch.zeros(batch_size, 0, 4), torch.zeros(batch_size, 4, 2, 2)

    def get_dense_pe(self):
        return torch.zeros(1, 4, 2, 2)


class _MaskDecoder(nn.Module):
    def forward(self, *, image_embeddings, **kwargs):
        batch_size = image_embeddings.shape[0]
        return (
            torch.ones(batch_size, 1, 2, 2),
            torch.ones(batch_size, 1),
        )


class _Network:
    use_lpeg = True

    def __init__(self):
        self.prompt_encoder = _PromptEncoder()
        self.mask_decoder = _MaskDecoder()

    @staticmethod
    def postprocess_masks(masks, *, input_size, original_size):
        return torch.nn.functional.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )


class TestAblationCommonHelpers(unittest.TestCase):
    def test_weighted_fusion_and_validation(self):
        tokens = tuple(torch.randn(2, 4, 8) for _ in range(4))
        weights = torch.softmax(torch.randn(2, 4), dim=1)
        fused = fuse_level_tokens(tokens, weights)
        expected = sum(
            weights[:, index, None, None] * value
            for index, value in enumerate(tokens)
        )
        torch.testing.assert_close(fused, expected)
        with self.assertRaises(ValueError):
            fuse_level_tokens(tokens[:-1], weights)
        with self.assertRaises(ValueError):
            fuse_level_tokens(tokens, weights.unsqueeze(-1))

    def test_token_reshape_neck_and_residual_contract(self):
        tokens = torch.randn(2, 4, 8)
        embeddings = torch.randn(2, 3, 2, 2)
        spatial = tokens_to_spatial(tokens, embeddings, embed_dim=8)
        self.assertEqual(tuple(spatial.shape), (2, 8, 2, 2))

        neck = nn.Conv2d(8, 3, kernel_size=1)
        auxiliary, enhanced = inject_enhancement(
            embeddings,
            tokens,
            neck,
            embed_dim=8,
        )
        self.assertEqual(auxiliary.shape, embeddings.shape)
        torch.testing.assert_close(enhanced, embeddings + auxiliary)

        with self.assertRaises(ValueError):
            tokens_to_spatial(tokens[:, :-1], embeddings, embed_dim=8)

    def test_decoder_contract_keeps_segmentation_shape(self):
        backbone = SimpleNamespace(network=_Network())
        logits, iou = decode_sam(backbone, torch.randn(2, 4, 2, 2), image_size=8)
        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(iou.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
