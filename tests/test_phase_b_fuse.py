"""Phase B up to the fuse stage: h_I, the retained Shape Teacher, and h_q."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.phase_b import (
    DEFAULT_LEVELS,
    MultiLevelImageDescriptor,
    PrivilegedFusion,
    build_shape_teacher,
    load_shape_teacher,
)
from src.models.shape import ShapeAutoencoder, ReconstructionDecoder, SmallCNN, SpatialProjector


EMBED_DIM = 768
PATCH_GRID = 16


def block_outputs(batch_size: int = 2, depth: int = 12):
    """Stand in for SAM's per-block [B, H_p, W_p, C] outputs."""

    return [
        torch.randn(batch_size, PATCH_GRID, PATCH_GRID, EMBED_DIM)
        for _ in range(depth)
    ]


class TestMultiLevelImageDescriptor(unittest.TestCase):
    def test_descriptor_has_shape_batch_by_cs(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=256)
        output = descriptor(block_outputs())
        self.assertEqual(tuple(output.descriptor.shape), (2, 256))

    def test_level_weights_are_a_softmax_over_the_selected_levels(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=64)
        output = descriptor(block_outputs())
        self.assertEqual(tuple(output.level_weights.shape), (2, len(DEFAULT_LEVELS)))
        self.assertTrue(torch.all(output.level_weights > 0))
        torch.testing.assert_close(
            output.level_weights.sum(dim=1),
            torch.ones(2),
        )

    def test_descriptor_is_the_attention_weighted_sum_of_level_descriptors(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=32)
        output = descriptor(block_outputs())
        expected = (
            output.level_weights.unsqueeze(-1) * output.level_descriptors
        ).sum(dim=1)
        torch.testing.assert_close(output.descriptor, expected)

    def test_levels_are_one_indexed_over_transformer_blocks(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=16, levels=(1, 2))
        blocks = block_outputs(depth=12)
        # Layer 1 is block_outputs[0]; changing block 3 must not move h_I.
        first = descriptor(blocks).descriptor
        blocks[2] = torch.randn_like(blocks[2])
        torch.testing.assert_close(descriptor(blocks).descriptor, first)

    def test_retained_tokens_are_flattened_to_batch_tokens_channels(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=16)
        output = descriptor(block_outputs())
        self.assertEqual(len(output.level_tokens), len(DEFAULT_LEVELS))
        for tokens in output.level_tokens:
            self.assertEqual(
                tuple(tokens.shape),
                (2, PATCH_GRID * PATCH_GRID, EMBED_DIM),
            )

    def test_missing_requested_level_is_rejected(self):
        descriptor = MultiLevelImageDescriptor(descriptor_dim=16)
        with self.assertRaises(ValueError):
            descriptor(block_outputs(depth=6))

    def test_wrong_embedding_width_is_rejected(self):
        descriptor = MultiLevelImageDescriptor(embed_dim=384, descriptor_dim=16)
        with self.assertRaises(ValueError):
            descriptor(block_outputs())

    def test_duplicate_levels_are_rejected(self):
        with self.assertRaises(ValueError):
            MultiLevelImageDescriptor(levels=(3, 3, 6))


class TestShapeTeacher(unittest.TestCase):
    @staticmethod
    def write_phase_a_checkpoint(path: Path) -> ShapeAutoencoder:
        model = ShapeAutoencoder(
            encoder=SmallCNN(),
            projector=SpatialProjector(),
            decoder=ReconstructionDecoder(),
        )
        torch.save(
            {
                "format_version": 2,
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "model_config": {
                        "name": "shape_autoencoder",
                        "encoder": {"name": "small_cnn"},
                    }
                },
            },
            path,
        )
        return model

    def test_loading_phase_a_keeps_gm_and_reproduces_its_latent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            autoencoder = self.write_phase_a_checkpoint(path)
            teacher = load_shape_teacher(path)

            masks = torch.rand(2, 1, 256, 256)
            autoencoder.eval()
            with torch.no_grad():
                expected = autoencoder(masks).latent
            torch.testing.assert_close(teacher(masks), expected)

    def test_loaded_teacher_has_no_reconstruction_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            self.write_phase_a_checkpoint(path)
            teacher = load_shape_teacher(path)
            self.assertFalse(hasattr(teacher, "decoder"))

    def test_frozen_teacher_stays_in_eval_and_produces_no_gradients(self):
        teacher = build_shape_teacher({"encoder": {"name": "small_cnn"}})
        teacher.train()
        self.assertFalse(teacher.training)
        latent = teacher(torch.rand(2, 1, 256, 256))
        self.assertFalse(latent.requires_grad)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in teacher.parameters())
        )

    def test_a_non_phase_a_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"model_state_dict": {"unet.weight": torch.zeros(1)}}, path)
            with self.assertRaises(ValueError):
                load_shape_teacher(path)

    def test_a_missing_checkpoint_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_shape_teacher("/nonexistent/best.pt")

    def test_mask_must_be_single_channel_four_dimensional(self):
        teacher = build_shape_teacher({})
        with self.assertRaises(ValueError):
            teacher(torch.rand(2, 3, 256, 256))


class TestPrivilegedFusion(unittest.TestCase):
    def test_hq_is_the_concatenation_of_hi_and_hm(self):
        fusion = PrivilegedFusion(descriptor_dim=8, shape_latent_dim=4)
        image_descriptor = torch.randn(3, 8)
        shape_latent = torch.randn(3, 4)
        output = fusion(image_descriptor, shape_latent)
        self.assertEqual(tuple(output.fused.shape), (3, 12))
        self.assertEqual(fusion.output_dim, 12)
        torch.testing.assert_close(output.fused[:, :8], image_descriptor)
        torch.testing.assert_close(output.fused[:, 8:], shape_latent)

    def test_mismatched_widths_are_rejected(self):
        fusion = PrivilegedFusion(descriptor_dim=8, shape_latent_dim=4)
        with self.assertRaises(ValueError):
            fusion(torch.randn(3, 7), torch.randn(3, 4))
        with self.assertRaises(ValueError):
            fusion(torch.randn(3, 8), torch.randn(3, 5))

    def test_mismatched_batch_sizes_are_rejected(self):
        fusion = PrivilegedFusion(descriptor_dim=8, shape_latent_dim=4)
        with self.assertRaises(ValueError):
            fusion(torch.randn(3, 8), torch.randn(2, 4))


if __name__ == "__main__":
    unittest.main()


class TestPhaseBFuseTask(unittest.TestCase):
    """The task must actually hand the mask to the model as privileged input."""

    class RecordingModel(torch.nn.Module):
        def __init__(self, *, emit_fuse_stage: bool = True):
            super().__init__()
            self.projection = torch.nn.Conv2d(3, 1, kernel_size=1)
            self.emit_fuse_stage = emit_fuse_stage
            self.received_masks = None

        def forward(self, images, masks=None, **kwargs):
            from src.models.base import SegmentationOutput
            from src.models.phase_b import FuseStageOutput

            self.received_masks = masks
            batch_size = images.shape[0]
            diagnostics = {
                "level_weights": torch.full((batch_size, 4), 0.25),
            }
            if self.emit_fuse_stage and masks is not None:
                diagnostics["fuse_stage"] = FuseStageOutput(
                    fused=torch.zeros(batch_size, 512),
                    image_descriptor=torch.zeros(batch_size, 256),
                    shape_latent=torch.zeros(batch_size, 256),
                    level_weights=diagnostics["level_weights"],
                    level_tokens=(),
                )
            return SegmentationOutput(
                logits=self.projection(images),
                diagnostics=diagnostics,
            )

    @staticmethod
    def batch(batch_size: int = 2):
        return {
            "image": torch.rand(batch_size, 3, 16, 16),
            "mask": torch.randint(0, 2, (batch_size, 1, 16, 16)).float(),
        }

    def make_task(self, **kwargs):
        from src.tasks import PhaseBFuseTask

        return PhaseBFuseTask(criterion=torch.nn.BCEWithLogitsLoss(), **kwargs)

    def test_training_step_passes_the_ground_truth_mask_to_the_model(self):
        model = self.RecordingModel()
        batch = self.batch()
        self.make_task().training_step(model, batch, torch.device("cpu"))
        self.assertIsNotNone(model.received_masks)
        torch.testing.assert_close(model.received_masks, batch["mask"])

    def test_evaluation_reports_segmentation_and_level_attention_metrics(self):
        output = self.make_task().evaluation_step(
            self.RecordingModel(), self.batch(), torch.device("cpu")
        )
        self.assertEqual(
            set(output.metrics),
            {
                "dice",
                "iou",
                "hd",
                "hd95",
                "assd",
                "boundary_f1",
                "level_weight_entropy",
                "level_weight_max",
            },
        )
        # Four equal weights: entropy is log(4), the maximum possible.
        self.assertAlmostEqual(
            float(output.metrics["level_weight_entropy"]),
            float(torch.tensor(4.0).log()),
            places=5,
        )

    def test_a_model_that_produces_no_hq_fails_loudly(self):
        model = self.RecordingModel(emit_fuse_stage=False)
        with self.assertRaises(ValueError):
            self.make_task().training_step(model, self.batch(), torch.device("cpu"))

    def test_strict_fuse_can_be_disabled(self):
        model = self.RecordingModel(emit_fuse_stage=False)
        task = self.make_task(strict_fuse=False)
        task.training_step(model, self.batch(), torch.device("cpu"))

    def test_multiclass_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_task(task="multiclass")
