"""Tests for the fixed Phase-A Small-CNN shape autoencoder."""

import unittest

import torch
from torch import nn

from src.losses import BCEDiceLoss
from src.models.shape import (
    ReconstructionDecoder,
    ShapeAutoencoder,
    SmallCNN,
    SpatialProjector,
)


def build_shape_model() -> ShapeAutoencoder:
    return ShapeAutoencoder(
        encoder=SmallCNN(),
        projector=SpatialProjector(),
        decoder=ReconstructionDecoder(),
    )


class TestShapeModels(unittest.TestCase):
    def test_forward_shapes_and_fixed_architecture(self):
        model = build_shape_model()
        output = model(torch.randn(2, 1, 256, 256))

        self.assertEqual(output.latent.shape, (2, 256))
        self.assertEqual(output.reconstruction_logits.shape, (2, 1, 256, 256))
        self.assertEqual(len(model.encoder.stages), 4)
        self.assertEqual(
            model.projector.spatial_pool.output_size,
            (4, 4),
        )
        self.assertEqual(
            model.decoder.initial_projection.out_features,
            128 * 8 * 8,
        )
        self.assertFalse(any(isinstance(module, nn.Sigmoid) for module in model.modules()))
        self.assertFalse(
            any(isinstance(module, nn.ConvTranspose2d) for module in model.modules())
        )

    def test_invalid_input_shapes_raise_value_error(self):
        model = build_shape_model()
        cases = (
            torch.randn(2, 256, 256),
            torch.randn(2, 3, 256, 256),
            torch.randn(2, 1, 128, 128),
        )
        for masks in cases:
            with self.subTest(shape=tuple(masks.shape)):
                with self.assertRaises(ValueError):
                    model(masks)

    def test_backward_reaches_every_parameter_and_module(self):
        torch.manual_seed(7)
        model = build_shape_model()
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
        output = model(targets)
        loss = BCEDiceLoss()(output.reconstruction_logits, targets)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module_name in ("encoder", "projector", "decoder"):
            module = getattr(model, module_name)
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and parameter.grad.abs().sum().item() > 0
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ),
                module_name,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
