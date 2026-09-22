"""Phase B enhancement stage: experts, g_layer, fusion, decoder injection.

The module-level tests run on synthetic tensors (no ViT-B); the full-model
smoke test that builds the backbone is gated behind RUN_BACKBONE_TESTS=1,
matching the router-stage tests.
"""

import os
import unittest

import torch

from src.models.phase_b import (
    EnhancementStageOutput,
    ExpertBank,
    FeedForwardExpert,
    HierarchicalMoEEnhancement,
    LayerPreferenceScorer,
    ShapeConditionedFusionMoEEnhancement,
)


EMBED_DIM = 64  # stand-in for 768; exercises the same code paths cheaply
LATENT_DIM = 16
NUM_EXPERTS = 4
NUM_LEVELS = 4
ACTIVE = 2
PATCHES = 16


def level_tokens(batch_size: int = 3):
    return tuple(
        torch.randn(batch_size, PATCHES, EMBED_DIM, requires_grad=True)
        for _ in range(NUM_LEVELS)
    )


class TestFeedForwardExpert(unittest.TestCase):
    def test_preserves_shape_and_expands_4x(self):
        expert = FeedForwardExpert(embed_dim=EMBED_DIM)
        self.assertEqual(expert.fc1.out_features, 4 * EMBED_DIM)
        self.assertEqual(expert.fc2.out_features, EMBED_DIM)
        out = expert(torch.randn(2, PATCHES, EMBED_DIM))
        self.assertEqual(tuple(out.shape), (2, PATCHES, EMBED_DIM))

    def test_gradients_flow_through_both_affine_layers(self):
        expert = FeedForwardExpert(embed_dim=EMBED_DIM)
        expert(torch.randn(2, 5, EMBED_DIM)).sum().backward()
        self.assertIsNotNone(expert.fc1.weight.grad)
        self.assertIsNotNone(expert.fc2.weight.grad)

    def test_rejects_wrong_channel_width(self):
        expert = FeedForwardExpert(embed_dim=EMBED_DIM)
        with self.assertRaises(ValueError):
            expert(torch.randn(2, 5, EMBED_DIM - 1))


class TestExpertBank(unittest.TestCase):
    def test_holds_k_experts_with_distinct_parameters(self):
        bank = ExpertBank(embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS)
        self.assertEqual(len(bank.experts), NUM_EXPERTS)
        weights = [e.fc1.weight for e in bank.experts]
        for i in range(NUM_EXPERTS):
            for j in range(i + 1, NUM_EXPERTS):
                self.assertFalse(torch.allclose(weights[i], weights[j]))

    def test_experts_receive_gradient_when_run(self):
        bank = ExpertBank(embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS)
        loss = 0
        for expert in bank.experts:
            loss = loss + expert(torch.randn(2, 3, EMBED_DIM)).sum()
        loss.backward()
        for expert in bank.experts:
            self.assertIsNotNone(expert.fc1.weight.grad)


class TestLayerPreferenceScorer(unittest.TestCase):
    def test_beta_is_softmax_over_levels_for_all_experts(self):
        scorer = LayerPreferenceScorer(
            embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            num_experts=NUM_EXPERTS, num_levels=NUM_LEVELS,
        )
        beta = scorer(
            torch.randn(3, NUM_LEVELS, EMBED_DIM), torch.randn(3, LATENT_DIM)
        )
        self.assertEqual(tuple(beta.shape), (3, NUM_EXPERTS, NUM_LEVELS))
        torch.testing.assert_close(beta.sum(dim=-1), torch.ones(3, NUM_EXPERTS))

    def test_beta_for_active_experts_only(self):
        scorer = LayerPreferenceScorer(
            embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            num_experts=NUM_EXPERTS, num_levels=NUM_LEVELS,
        )
        indices = torch.tensor([[0, 2], [1, 3], [3, 0]])
        beta = scorer(
            torch.randn(3, NUM_LEVELS, EMBED_DIM),
            torch.randn(3, LATENT_DIM),
            indices,
        )
        self.assertEqual(tuple(beta.shape), (3, ACTIVE, NUM_LEVELS))
        torch.testing.assert_close(beta.sum(dim=-1), torch.ones(3, ACTIVE))

    def test_preferences_differ_across_experts(self):
        scorer = LayerPreferenceScorer(
            embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            num_experts=NUM_EXPERTS, num_levels=NUM_LEVELS,
        )
        torch.manual_seed(0)
        beta = scorer(
            torch.randn(2, NUM_LEVELS, EMBED_DIM), torch.randn(2, LATENT_DIM)
        )
        # Rows are per-expert preferences for the same sample.
        self.assertFalse(torch.allclose(beta[0, 0], beta[0, 1]))

    def test_invalid_shapes_are_rejected(self):
        scorer = LayerPreferenceScorer(
            embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            num_experts=NUM_EXPERTS, num_levels=NUM_LEVELS,
        )
        with self.assertRaises(ValueError):
            scorer(torch.randn(2, NUM_LEVELS, EMBED_DIM - 1), torch.randn(2, LATENT_DIM))
        with self.assertRaises(ValueError):
            scorer(torch.randn(2, NUM_LEVELS + 1, EMBED_DIM), torch.randn(2, LATENT_DIM))
        with self.assertRaises(ValueError):
            scorer(torch.randn(2, NUM_LEVELS, EMBED_DIM), torch.randn(3, LATENT_DIM))


class TestHierarchicalMoEEnhancement(unittest.TestCase):
    def _routing(self, batch_size: int = 3):
        torch.manual_seed(11)
        logits = torch.randn(batch_size, NUM_EXPERTS)
        dense = torch.softmax(logits, dim=1)
        probs, indices = dense.topk(ACTIVE, dim=1)
        renorm = probs / probs.sum(dim=1, keepdim=True)
        sparse = torch.zeros_like(dense).scatter(1, indices, renorm)
        return sparse, indices

    def _run(self, batch_size: int = 3):
        torch.manual_seed(0)
        module = HierarchicalMoEEnhancement(
            embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS, latent_dim=LATENT_DIM,
        )
        tokens = level_tokens(batch_size)
        pools = torch.randn(batch_size, NUM_LEVELS, EMBED_DIM)
        z = torch.randn(batch_size, LATENT_DIM)
        pi, K_b = self._routing(batch_size)
        return module(tokens, pools, z, pi, K_b)

    def test_output_shapes(self):
        out = self._run()
        self.assertEqual(tuple(out.fused_tokens.shape), (3, PATCHES, EMBED_DIM))
        self.assertEqual(len(out.enhanced_tokens), NUM_LEVELS)
        self.assertEqual(tuple(out.layer_weights.shape), (3, NUM_LEVELS))
        self.assertEqual(tuple(out.expert_layer_weights.shape), (3, ACTIVE, NUM_LEVELS))

    def test_gamma_sums_to_one(self):
        # pi sums to 1 over K_b and beta sums to 1 over levels, so
        # gamma_l = sum_k pi_k beta_{k,l} sums to 1 over levels.
        out = self._run()
        torch.testing.assert_close(out.layer_weights.sum(dim=1), torch.ones(3))

    def test_residual_identity_when_experts_output_zero(self):
        # If every expert maps its input to zero, X_hat == X and Z_fused is
        # the gamma-weighted sum of the raw tokens.
        module = HierarchicalMoEEnhancement(
            embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS, latent_dim=LATENT_DIM,
        )
        for p in module.experts.parameters():
            torch.nn.init.zeros_(p)
        tokens = level_tokens(2)
        pools = torch.randn(2, NUM_LEVELS, EMBED_DIM)
        z = torch.randn(2, LATENT_DIM)
        pi, K_b = self._routing(2)
        out = module(tokens, pools, z, pi, K_b)
        expected = sum(
            out.layer_weights[:, l].view(2, 1, 1) * tokens[l]
            for l in range(NUM_LEVELS)
        )
        torch.testing.assert_close(out.fused_tokens, expected)
        for l in range(NUM_LEVELS):
            torch.testing.assert_close(out.enhanced_tokens[l], tokens[l])

    def test_gradient_reaches_experts_and_scorer(self):
        module = HierarchicalMoEEnhancement(
            embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS, latent_dim=LATENT_DIM,
        )
        tokens = level_tokens(2)
        pools = torch.randn(2, NUM_LEVELS, EMBED_DIM)
        z = torch.randn(2, LATENT_DIM)
        pi, K_b = self._routing(2)
        out = module(tokens, pools, z, pi, K_b)
        out.fused_tokens.sum().backward()
        got_expert_grad = any(
            e.fc1.weight.grad is not None for e in module.experts.experts
        )
        self.assertTrue(got_expert_grad)
        self.assertIsNotNone(module.layer_scorer.scorer[0].weight.grad)
        self.assertIsNotNone(module.layer_scorer.expert_embeddings.grad)

    def test_unrouted_experts_get_no_gradient(self):
        # Only experts in K_b may receive gradient; with the routing fixed
        # and enough samples that not every expert is picked, the untouched
        # experts' parameters must stay grad-free.
        torch.manual_seed(3)
        module = HierarchicalMoEEnhancement(
            embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS, latent_dim=LATENT_DIM,
        )
        tokens = level_tokens(2)
        pools = torch.randn(2, NUM_LEVELS, EMBED_DIM)
        z = torch.randn(2, LATENT_DIM)
        # Route both samples to experts {0, 2} only.
        K_b = torch.tensor([[0, 2], [0, 2]])
        pi = torch.zeros(2, NUM_EXPERTS)
        pi.scatter_(1, K_b, 0.5)
        out = module(tokens, pools, z, pi, K_b)
        out.fused_tokens.sum().backward()
        for k, expert in enumerate(module.experts.experts):
            if k in (0, 2):
                self.assertIsNotNone(expert.fc1.weight.grad)
                self.assertNotEqual(float(expert.fc1.weight.grad.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    expert.fc1.weight.grad is None
                    or float(expert.fc1.weight.grad.abs().sum()) == 0.0
                )

    def test_mismatched_level_token_counts_rejected(self):
        module = HierarchicalMoEEnhancement(
            embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS, latent_dim=LATENT_DIM,
        )
        tokens = level_tokens(2) + (torch.randn(2, 8, EMBED_DIM),)
        pools = torch.randn(2, NUM_LEVELS, EMBED_DIM)
        z = torch.randn(2, LATENT_DIM)
        pi, K_b = self._routing(2)
        with self.assertRaises(ValueError):
            module(tokens, pools, z, pi, K_b)


class TestShapeConditionedFusionMoEEnhancement(unittest.TestCase):
    def _routing(self, batch_size: int = 3):
        indices = torch.tensor([[0, 2]] * batch_size)
        probs = torch.zeros(batch_size, NUM_EXPERTS)
        probs.scatter_(1, indices, 0.5)
        return probs, indices

    def _module(self):
        return ShapeConditionedFusionMoEEnhancement(
            embed_dim=EMBED_DIM,
            num_experts=NUM_EXPERTS,
            num_levels=NUM_LEVELS,
            latent_dim=LATENT_DIM,
        )

    def _run(self, batch_size: int = 3):
        torch.manual_seed(17)
        module = self._module()
        tokens = level_tokens(batch_size)
        latent = torch.randn(batch_size, LATENT_DIM)
        probs, indices = self._routing(batch_size)
        return module, tokens, module(tokens, latent, probs, indices)

    def test_alpha_and_token_shapes(self):
        module, tokens, out = self._run()
        self.assertEqual(module.g_fuse[0].in_features, EMBED_DIM + LATENT_DIM)
        self.assertEqual(module.g_fuse[0].out_features, 64)
        self.assertEqual(tuple(out.layer_weights.shape), (3, NUM_LEVELS))
        self.assertTrue(torch.isfinite(out.layer_weights).all())
        torch.testing.assert_close(
            out.layer_weights.sum(dim=1), torch.ones(3)
        )
        self.assertEqual(
            tuple(out.fused_tokens.shape), (3, PATCHES, EMBED_DIM)
        )
        self.assertEqual(
            tuple(out.enhanced_fused_tokens.shape),
            (3, PATCHES, EMBED_DIM),
        )
        expected = sum(
            out.layer_weights[:, level].view(3, 1, 1) * tokens[level]
            for level in range(NUM_LEVELS)
        )
        torch.testing.assert_close(out.fused_tokens, expected)

    def test_zero_expert_outputs_leave_fused_tokens_unchanged(self):
        module = self._module()
        for parameter in module.experts.parameters():
            torch.nn.init.zeros_(parameter)
        tokens = level_tokens(2)
        latent = torch.randn(2, LATENT_DIM)
        probs, indices = self._routing(2)
        out = module(tokens, latent, probs, indices)
        torch.testing.assert_close(
            out.enhanced_fused_tokens, out.fused_tokens
        )

    def test_only_selected_experts_run_and_receive_gradient(self):
        module = self._module()
        call_counts = [0] * NUM_EXPERTS

        def counting_hook(expert_id):
            def hook(_module, _inputs, _output):
                call_counts[expert_id] += 1

            return hook

        handles = [
            expert.register_forward_hook(counting_hook(expert_id))
            for expert_id, expert in enumerate(module.experts.experts)
        ]
        tokens = level_tokens(3)
        latent = torch.randn(3, LATENT_DIM)
        probs, indices = self._routing(3)
        try:
            out = module(tokens, latent, probs, indices)
        finally:
            for handle in handles:
                handle.remove()

        out.enhanced_fused_tokens.sum().backward()
        self.assertEqual(call_counts, [1, 0, 1, 0])
        for expert_id, expert in enumerate(module.experts.experts):
            gradient = expert.fc1.weight.grad
            if expert_id in (0, 2):
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    gradient is None or float(gradient.abs().sum()) == 0.0
                )

    def test_fusion_scorer_receives_gradient(self):
        module, _, out = self._run(batch_size=2)
        out.enhanced_fused_tokens.square().mean().backward()
        for layer_index in (0, 2):
            gradient = module.g_fuse[layer_index].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_invalid_input_shapes_are_rejected(self):
        module = self._module()
        tokens = level_tokens(2)
        latent = torch.randn(2, LATENT_DIM)
        probs, indices = self._routing(2)

        with self.assertRaises(ValueError):
            module(tokens[:-1], latent, probs, indices)
        wrong_channels = tokens[:-1] + (
            torch.randn(2, PATCHES, EMBED_DIM - 1),
        )
        with self.assertRaises(ValueError):
            module(wrong_channels, latent, probs, indices)
        with self.assertRaises(ValueError):
            module(tokens, torch.randn(2, LATENT_DIM - 1), probs, indices)
        with self.assertRaises(ValueError):
            module(tokens, latent, probs[:, :-1], indices)
        with self.assertRaises(ValueError):
            module(tokens, latent, probs, indices[:1])
        with self.assertRaises(ValueError):
            module(tokens, latent, probs, indices.float())
        with self.assertRaises(ValueError):
            module(
                tokens,
                latent,
                probs,
                torch.zeros(2, NUM_EXPERTS + 1, dtype=torch.long),
            )

    def test_task_logs_shape_fusion_entropy_separately(self):
        from src.tasks.phase_b_moe import PhaseBMoETask

        alpha = torch.full((2, NUM_LEVELS), 1.0 / NUM_LEVELS)
        stage = EnhancementStageOutput(
            layer_weights=alpha,
            expert_layer_weights=None,
            shape_fusion_layer_weights=alpha,
            aux_norm_ratio=torch.ones(2),
        )
        metrics = PhaseBMoETask._enhancement_metrics(stage)
        self.assertIn("shape_fusion_layer_weight_entropy", metrics)
        self.assertNotIn("fused_level_weight_entropy", metrics)

        hierarchical_stage = EnhancementStageOutput(
            layer_weights=alpha,
            expert_layer_weights=torch.ones(2, ACTIVE, NUM_LEVELS),
            shape_fusion_layer_weights=None,
            aux_norm_ratio=torch.ones(2),
        )
        hierarchical_metrics = PhaseBMoETask._enhancement_metrics(
            hierarchical_stage
        )
        self.assertIn("fused_level_weight_entropy", hierarchical_metrics)
        self.assertNotIn(
            "shape_fusion_layer_weight_entropy", hierarchical_metrics
        )

    def test_invalid_enhancement_mode_is_rejected_before_model_build(self):
        from src.models.phase_b import PhaseBMoEStage

        with self.assertRaisesRegex(ValueError, "enhancement_mode"):
            PhaseBMoEStage(enhancement_mode="unknown")


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone in this test.",
)
class TestPhaseBMoEStageModel(unittest.TestCase):
    def test_forward_enhances_and_changes_logits(self):
        from src.models.phase_b import PhaseBMoEStage

        torch.manual_seed(0)
        model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.eval()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()

        with torch.no_grad():
            out = model(images, masks=masks)

        diagnostics = out.diagnostics
        network = model.backbone.network
        self.assertFalse(network.use_moe)
        self.assertFalse(hasattr(network, "ExpertChoiceTokenMoE"))
        self.assertIsNone(diagnostics["moe_expert_indices"])
        self.assertIn("phase_b_router", diagnostics)
        self.assertIn("phase_b_moe", diagnostics)
        stage = diagnostics["phase_b_moe"]
        self.assertEqual(model.enhancement_mode, "hierarchical")
        self.assertIsInstance(model.enhancement, HierarchicalMoEEnhancement)
        self.assertIsNotNone(stage.expert_layer_weights)
        self.assertIsNone(stage.shape_fusion_layer_weights)
        self.assertGreater(float(stage.aux_norm_ratio.mean()), 0.0)
        # Full model outputs [B, num_classes, H, W] logits.
        self.assertEqual(out.logits.shape[-2:], (256, 256))

    def test_gradients_reach_enhancement_but_not_frozen_backbone(self):
        from src.models.phase_b import PhaseBMoEStage

        torch.manual_seed(0)
        model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.train()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()

        model(images, masks=masks).logits.sum().backward()
        # Enhancement received gradient...
        self.assertTrue(
            any(
                e.fc1.weight.grad is not None
                and float(e.fc1.weight.grad.abs().sum()) > 0.0
                for e in model.enhancement.experts.experts
            ),
            "Segmentation output must backpropagate into a routed Shape expert.",
        )
        # ...but the frozen ViT blocks did not.
        for name, param in model.backbone.network.image_encoder.named_parameters():
            if "Adapter" not in name:
                self.assertFalse(param.requires_grad)
                break

    def test_inference_path_routes_from_prior(self):
        from src.models.phase_b import PhaseBMoEStage

        torch.manual_seed(0)
        model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.eval()
        images = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            out = model(images)

        self.assertEqual(out.diagnostics["phase_b_router"].source, "prior")
        self.assertIn("phase_b_moe", out.diagnostics)

    def test_shape_conditioned_forward_backward_contract(self):
        from src.models.phase_b import PhaseBMoEStage

        torch.manual_seed(0)
        model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
            enhancement_mode="shape_conditioned",
        )
        model.train()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()

        out = model(images, masks=masks)
        stage = out.diagnostics["phase_b_moe"]
        router_stage = out.diagnostics["phase_b_router"]
        alpha = stage.shape_fusion_layer_weights

        self.assertEqual(model.enhancement_mode, "shape_conditioned")
        self.assertIsInstance(
            model.enhancement, ShapeConditionedFusionMoEEnhancement
        )
        self.assertFalse(model.backbone.network.use_moe)
        self.assertFalse(
            hasattr(model.backbone.network, "ExpertChoiceTokenMoE")
        )
        self.assertIsNone(out.diagnostics["moe_expert_indices"])
        self.assertEqual(router_stage.source, "posterior")
        self.assertIsNone(stage.expert_layer_weights)
        self.assertIs(stage.layer_weights, alpha)
        self.assertEqual(tuple(alpha.shape), (1, NUM_LEVELS))
        torch.testing.assert_close(alpha.sum(dim=1), torch.ones(1))
        self.assertGreater(float(stage.aux_norm_ratio.mean()), 0.0)
        self.assertEqual(tuple(out.logits.shape), (1, 1, 256, 256))

        out.logits.sum().backward()
        for layer_index in (0, 2):
            gradient = model.enhancement.g_fuse[layer_index].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

        selected = set(router_stage.routing.expert_indices.flatten().tolist())
        for expert_id, expert in enumerate(model.enhancement.experts.experts):
            gradient = expert.fc1.weight.grad
            if expert_id in selected:
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    gradient is None or float(gradient.abs().sum()) == 0.0
                )

    def test_shape_conditioned_inference_routes_from_prior(self):
        from src.models.phase_b import PhaseBMoEStage

        torch.manual_seed(0)
        model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
            enhancement_mode="shape_conditioned",
        )
        model.eval()

        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256))

        stage = out.diagnostics["phase_b_moe"]
        self.assertEqual(out.diagnostics["phase_b_router"].source, "prior")
        self.assertEqual(
            tuple(stage.shape_fusion_layer_weights.shape), (1, NUM_LEVELS)
        )
        self.assertEqual(tuple(out.logits.shape), (1, 1, 256, 256))

    def test_hierarchical_default_and_explicit_modes_are_strictly_compatible(self):
        from src.models.phase_b import PhaseBMoEStage

        default_model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        explicit_model = PhaseBMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
            enhancement_mode="hierarchical",
        )

        self.assertEqual(default_model.enhancement_mode, "hierarchical")
        self.assertIsInstance(
            default_model.enhancement, HierarchicalMoEEnhancement
        )
        self.assertEqual(
            tuple(default_model.state_dict()),
            tuple(explicit_model.state_dict()),
        )
        explicit_model.load_state_dict(default_model.state_dict(), strict=True)

if __name__ == "__main__":
    unittest.main()
