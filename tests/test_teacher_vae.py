"""CPU tests for the Phase 0 mask-VAE shape teacher."""

import math
import unittest
from pathlib import Path

import torch

from src.losses import MaskVAELoss, gaussian_kl_to_standard_normal
from scripts.training.pretrain_teacher_vae import run_epoch as run_teacher_epoch
from src.models import MaskVAETeacher, load_mask_embedding_encoder


def build_small_teacher(latent_dim: int = 8) -> MaskVAETeacher:
    """A 32x32 teacher small enough to train inside a unit test."""

    return MaskVAETeacher(
        latent_dim=latent_dim,
        image_size=(32, 32),
        encoder_channels=(8, 16),
        decoder_channels=(16, 8, 8),
        decoder_seed_size=8,
    )


def random_masks(batch_size: int = 4, size: int = 32) -> torch.Tensor:
    return (torch.rand(batch_size, 1, size, size) > 0.5).float()


class TestMaskVAETeacher(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_small_teacher()
        self.masks = random_masks()

    def test_forward_shapes(self):
        output = self.model(self.masks)

        self.assertEqual(output.recon_logits.shape, self.masks.shape)
        self.assertEqual(output.mu.shape, (4, 8))
        self.assertEqual(output.logvar.shape, (4, 8))
        self.assertEqual(output.z.shape, (4, 8))

    def test_decoder_returns_logits_not_probabilities(self):
        output = self.model(self.masks)
        logits = output.recon_logits

        self.assertTrue(
            (logits < 0.0).any() or (logits > 1.0).any(),
            "decoder must emit raw logits, not sigmoid outputs.",
        )

    def test_sampling_is_stochastic_but_mu_is_not(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(self.masks).z
            second = self.model(self.masks).z
            deterministic = self.model(self.masks, sample=False).z

        self.assertFalse(torch.allclose(first, second))
        self.assertTrue(
            torch.allclose(deterministic, self.model.encode(self.masks)[0])
        )

    def test_reparameterization_passes_gradient_to_both_heads(self):
        output = self.model(self.masks)
        output.z.sum().backward()

        self.assertIsNotNone(self.model.fc_mu.weight.grad)
        self.assertIsNotNone(self.model.fc_logvar.weight.grad)
        self.assertGreater(self.model.fc_mu.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            self.model.fc_logvar.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_logvar_is_clamped(self):
        model = build_small_teacher()
        with torch.no_grad():
            model.fc_logvar.bias.fill_(500.0)
            model.fc_logvar.weight.fill_(0.0)
        _, logvar = model.encode(self.masks)

        self.assertTrue(torch.all(logvar <= model.logvar_max))

    def test_rejects_mismatched_mask_resolution(self):
        with self.assertRaises(ValueError):
            self.model(random_masks(size=64))

    def test_rejects_incompatible_decoder_geometry(self):
        with self.assertRaises(ValueError):
            MaskVAETeacher(
                latent_dim=8,
                image_size=(32, 32),
                encoder_channels=(8, 16),
                decoder_channels=(16, 8),
                decoder_seed_size=8,
            )

    def test_evaluation_epoch_reports_all_reconstruction_metrics(self):
        metrics = run_teacher_epoch(
            model=self.model,
            loader=[{"mask": self.masks}],
            criterion=MaskVAELoss(),
            device=torch.device("cpu"),
            include_surface_metrics=True,
        )

        expected = {
            "loss",
            "reconstruction",
            "kl",
            "dice",
            "iou",
            "hd95",
            "assd",
            "boundary_f1",
            "active_units",
        }
        self.assertTrue(expected.issubset(metrics))
        self.assertTrue(all(math.isfinite(metrics[key]) for key in expected))

    def test_model_config_round_trips(self):
        rebuilt = MaskVAETeacher(**self.model.model_config())

        self.assertEqual(rebuilt.latent_dim, self.model.latent_dim)
        self.assertEqual(rebuilt.image_size, self.model.image_size)


class TestMaskEmbeddingEncoder(unittest.TestCase):
    """E_M is the Phase 0 deliverable: loadable and freezable on its own."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = build_small_teacher()
        self.masks = random_masks()

    def test_embedding_is_a_feature_map_not_a_vector(self):
        embedding = self.model.embed(self.masks)

        # Two stride-2 blocks on a 32x32 mask leave an 8x8 map.
        self.assertEqual(embedding.shape, (4, 16, 8, 8))

    def test_embedding_geometry_matches_the_actual_embedding(self):
        geometry = self.model.embedding_geometry()
        embedding = self.model.embed(self.masks)

        self.assertEqual(geometry["embedding_channels"], embedding.shape[1])
        self.assertEqual(geometry["embedding_size"], list(embedding.shape[2:]))
        self.assertEqual(geometry["embedding_stride"], 4)

    def test_encode_runs_through_the_same_embedding(self):
        pooled = self.model.pool(self.model.embed(self.masks)).flatten(1)
        expected_mu = self.model.fc_mu(pooled)

        self.assertTrue(torch.allclose(self.model.encode(self.masks)[0], expected_mu))

    def test_freeze_stops_gradients_only_in_the_embedding_encoder(self):
        self.model.freeze_mask_embedding_encoder()
        self.model(self.masks).recon_logits.sum().backward()

        for parameter in self.model.mask_embedding_encoder.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(self.model.fc_mu.weight.grad)

    def test_state_dict_covers_the_embedding_encoder_only(self):
        keys = set(self.model.mask_embedding_state_dict())
        expected = set(self.model.mask_embedding_encoder.state_dict())

        self.assertEqual(keys, expected)
        self.assertFalse(any(key.startswith("decoder") for key in keys))

    def test_loader_rebuilds_a_frozen_encoder_from_a_checkpoint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(
                {
                    "model_config": self.model.model_config(),
                    "model_state_dict": self.model.state_dict(),
                },
                path,
            )
            encoder, geometry = load_mask_embedding_encoder(path)

        # The loaded encoder is frozen, so compare against the teacher in the
        # same mode: BatchNorm uses batch statistics while training and running
        # statistics while evaluating.
        self.model.eval()

        self.assertEqual(geometry["embedding_channels"], 16)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        self.assertTrue(
            torch.allclose(encoder(self.masks), self.model.embed(self.masks))
        )

    def test_loaded_encoder_stays_frozen_when_a_parent_calls_train(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(
                {
                    "model_config": self.model.model_config(),
                    "model_state_dict": self.model.state_dict(),
                },
                path,
            )
            encoder, _ = load_mask_embedding_encoder(path)

        parent = torch.nn.Sequential(encoder)
        parent.train()

        self.assertFalse(encoder.training)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))

    def test_freeze_survives_a_later_train_call(self):
        self.model.freeze_mask_embedding_encoder()
        self.model.train()

        self.assertFalse(self.model.mask_embedding_encoder.training)
        self.assertTrue(self.model.fc_mu.training)

    def test_frozen_batchnorm_statistics_stop_moving(self):
        self.model.freeze_mask_embedding_encoder()
        self.model.train()
        first_block = self.model.mask_embedding_encoder[0][1]
        before = first_block.running_mean.clone()

        for _ in range(3):
            self.model(random_masks())

        self.assertTrue(torch.equal(before, first_block.running_mean))

    def test_loader_rejects_a_foreign_checkpoint(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.pt"
            torch.save({"something_else": 1}, path)
            with self.assertRaises(ValueError):
                load_mask_embedding_encoder(path)


class TestGaussianKL(unittest.TestCase):
    def test_zero_for_standard_normal_posterior(self):
        mu = torch.zeros(3, 8)
        logvar = torch.zeros(3, 8)

        self.assertTrue(
            torch.allclose(
                gaussian_kl_to_standard_normal(mu, logvar),
                torch.zeros(3),
                atol=1e-6,
            )
        )

    def test_non_negative_for_random_posteriors(self):
        torch.manual_seed(1)
        mu = torch.randn(16, 8)
        logvar = torch.randn(16, 8)

        self.assertTrue(torch.all(gaussian_kl_to_standard_normal(mu, logvar) >= 0.0))

    def test_matches_manual_scalar_case(self):
        # One dimension, mu = 1, sigma^2 = e => KL = 0.5 * (1 + e - 1 - 1).
        mu = torch.tensor([[1.0]])
        logvar = torch.tensor([[1.0]])
        expected = 0.5 * (1.0 + math.e - 1.0 - 1.0)

        self.assertAlmostEqual(
            gaussian_kl_to_standard_normal(mu, logvar).item(),
            expected,
            places=5,
        )

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            gaussian_kl_to_standard_normal(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))


class TestMaskVAELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_small_teacher()
        self.masks = random_masks()
        self.output = self.model(self.masks)

    def test_total_is_reconstruction_plus_beta_times_kl(self):
        criterion = MaskVAELoss(beta=0.3)
        losses = criterion(self.output, self.masks)
        expected = losses.reconstruction + 0.3 * losses.kl

        self.assertTrue(torch.allclose(losses.total, expected))

    def test_beta_zero_drops_the_prior_term(self):
        losses = MaskVAELoss(beta=0.0)(self.output, self.masks)

        self.assertTrue(torch.allclose(losses.total, losses.reconstruction))

    def test_sum_reduction_scales_with_pixel_count(self):
        summed = MaskVAELoss(recon_reduction="sum")(self.output, self.masks)
        averaged = MaskVAELoss(recon_reduction="mean")(self.output, self.masks)

        self.assertTrue(
            torch.allclose(
                summed.reconstruction,
                averaged.reconstruction * 32 * 32,
                rtol=1e-4,
            )
        )

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            MaskVAELoss()(self.output, random_masks(batch_size=2))

    def test_rejects_negative_beta(self):
        with self.assertRaises(ValueError):
            MaskVAELoss(beta=-1.0)

    def test_overfits_a_single_batch(self):
        torch.manual_seed(0)
        model = build_small_teacher()
        criterion = MaskVAELoss(beta=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        masks = random_masks(batch_size=2)

        first = criterion(model(masks), masks).total.item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(masks), masks).total
            loss.backward()
            optimizer.step()
        last = criterion(model(masks), masks).total.item()

        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
