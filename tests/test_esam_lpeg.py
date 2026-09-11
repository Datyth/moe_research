"""Tests for the MoE-SAM LPEG, prompt-encoder contract, ablation switches and
deterministic MoE routing at evaluation time."""

import unittest
from types import SimpleNamespace

import torch

from src.models import EsamModel, build_model
from src.models.esam.lpeg import LPEG
from src.models.esam._vendor.prompt_encoder import PromptEncoder


def _build_esam(**model_overrides):
    config = {
        "name": "esam",
        "in_channels": 3,
        "num_classes": 1,
        "task": "binary",
        "image_size": 64,
        "checkpoint": None,
        "moe_num_experts": 2,
        "moe_top_k_ratio": 0.5,
    }
    config.update(model_overrides)
    return build_model(config)


class TestLPEGModule(unittest.TestCase):
    def test_output_is_single_sparse_token(self):
        lpeg = LPEG(embed_dim=256)
        embedding = torch.randn(2, 256, 4, 4)
        prompt = lpeg(embedding)
        self.assertEqual(prompt.shape, (2, 1, 256))

    def test_inner_activation_is_gelu(self):
        self.assertIsInstance(LPEG().act2, torch.nn.GELU)


class TestPromptEncoderContract(unittest.TestCase):
    def _encoder(self, use_lpeg):
        args = SimpleNamespace(batch_size=1)
        return PromptEncoder(
            args=args,
            embed_dim=256,
            image_embedding_size=(4, 4),
            input_image_size=(64, 64),
            mask_in_chans=16,
            use_lpeg=use_lpeg,
        )

    def test_lpeg_enabled_returns_one_sparse_token_and_no_mask_dense(self):
        encoder = self._encoder(use_lpeg=True)
        embedding = torch.randn(3, 256, 4, 4)
        sparse, dense = encoder(
            points=None, boxes=None, masks=None,
            image_embedding=embedding, batch_size=3,
        )
        self.assertEqual(sparse.shape, (3, 1, 256))
        expected_dense = encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(3, -1, 4, 4)
        self.assertEqual(dense.shape, (3, 256, 4, 4))
        torch.testing.assert_close(dense, expected_dense)

    def test_lpeg_disabled_rejects_image_embedding(self):
        encoder = self._encoder(use_lpeg=False)
        with self.assertRaises(ValueError):
            encoder(
                points=None, boxes=None, masks=None,
                image_embedding=torch.randn(2, 256, 4, 4),
            )

    def test_lpeg_disabled_empty_sparse_uses_explicit_batch_size(self):
        encoder = self._encoder(use_lpeg=False)
        sparse, dense = encoder(
            points=None, boxes=None, masks=None,
            image_embedding=None, batch_size=5,
        )
        self.assertEqual(sparse.shape, (5, 0, 256))
        self.assertEqual(dense.shape, (5, 256, 4, 4))


class TestEsamAblationMatrix(unittest.TestCase):
    """All four use_moe x use_lpeg combinations must forward and backward."""

    def test_all_four_combinations(self):
        for use_moe in (False, True):
            for use_lpeg in (False, True):
                with self.subTest(use_moe=use_moe, use_lpeg=use_lpeg):
                    model = _build_esam(use_moe=use_moe, use_lpeg=use_lpeg)
                    model.train()
                    if use_moe:
                        self.assertTrue(hasattr(model.network, "ExpertChoiceTokenMoE"))
                    else:
                        self.assertFalse(
                            hasattr(model.network, "ExpertChoiceTokenMoE")
                        )
                    self.assertIs(
                        model.network.prompt_encoder.lpeg is not None, use_lpeg
                    )
                    images = torch.randn(2, 3, 64, 64)
                    output = model(images)
                    self.assertEqual(output.logits.shape, (2, 1, 64, 64))
                    output.logits.mean().backward()

    def test_full_model_backward_reaches_lpeg(self):
        model = _build_esam(use_moe=True, use_lpeg=True)
        model.train()
        output = model(torch.randn(2, 3, 64, 64))
        output.logits.mean().backward()
        for name, parameter in model.network.prompt_encoder.named_parameters():
            if "lpeg" in name:
                self.assertIsNotNone(parameter.grad, msg=name)
                self.assertTrue(parameter.requires_grad, msg=name)
            else:
                self.assertFalse(parameter.requires_grad, msg=name)


class TestPromptAfterMoeOrdering(unittest.TestCase):
    def test_prompt_encoder_runs_after_moe_neck(self):
        model = _build_esam(use_moe=True, use_lpeg=True)
        model.train()
        order = []
        neck_hook = model.network.neck5.register_forward_hook(
            lambda module, inputs, output: order.append("neck5")
        )
        prompt_hook = model.network.prompt_encoder.register_forward_hook(
            lambda module, inputs, output: order.append("prompt_encoder")
        )
        try:
            with torch.no_grad():
                model(torch.randn(2, 3, 64, 64))
        finally:
            neck_hook.remove()
            prompt_hook.remove()
        self.assertEqual(order, ["neck5", "prompt_encoder"])


class TestDeterministicMoERouting(unittest.TestCase):
    def test_eval_mode_routing_is_deterministic(self):
        model = _build_esam(use_moe=True, use_lpeg=True)
        images = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            first = model(images)
            second = model(images)
        torch.testing.assert_close(first.logits, second.logits)
        torch.testing.assert_close(
            first.diagnostics["moe_expert_indices"],
            second.diagnostics["moe_expert_indices"],
        )

    def test_train_mode_router_applies_noise(self):
        model = _build_esam(use_moe=True, use_lpeg=True)
        model.train()
        torch.manual_seed(0)
        first = model(torch.randn(2, 3, 64, 64))
        torch.manual_seed(1)
        second = model(torch.randn(2, 3, 64, 64))
        self.assertFalse(torch.allclose(first.logits, second.logits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
