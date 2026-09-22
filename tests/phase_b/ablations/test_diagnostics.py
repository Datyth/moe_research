"""Formula tests for shared Phase B diagnostics."""

import math
import unittest
from types import SimpleNamespace

import torch

from src.models.phase_b import EnhancementStageOutput, RoutingOutput
from src.tasks.phase_b_diagnostics import (
    enhancement_metrics,
    layer_fusion_metrics,
    routing_metrics,
)


class TestPhaseBDiagnosticFormulas(unittest.TestCase):
    def test_usage_entropy_uses_active_slot_fractions(self):
        dense = torch.tensor(
            [
                [0.7, 0.2, 0.05, 0.05],
                [0.6, 0.05, 0.3, 0.05],
            ]
        )
        indices = torch.tensor([[0, 1], [0, 2]])
        active = dense.gather(1, indices)
        active = active / active.sum(dim=1, keepdim=True)
        sparse = torch.zeros_like(dense).scatter(1, indices, active)
        stage = SimpleNamespace(
            routing=RoutingOutput(
                logits=dense.log(),
                dense_probs=dense,
                routing_probs=sparse,
                expert_indices=indices,
            ),
            latent_kl=None,
            balance=torch.tensor(1.25),
        )

        metrics = routing_metrics(stage)
        fractions = torch.tensor([0.5, 0.25, 0.25, 0.0])
        expected_entropy = -(
            fractions.clamp_min(torch.finfo(fractions.dtype).tiny).log()
            * fractions
        ).sum()
        torch.testing.assert_close(
            metrics["expert_usage_entropy"],
            expected_entropy,
        )
        for expert_id, expected in enumerate(fractions):
            torch.testing.assert_close(
                metrics[f"expert_usage_fraction_{expert_id}"],
                expected,
            )
        torch.testing.assert_close(
            metrics["routing_max_probability"],
            torch.tensor(0.65),
        )
        torch.testing.assert_close(
            metrics["routing_top1_top2_margin"],
            torch.tensor(0.4),
        )
        torch.testing.assert_close(
            metrics["load_balance_loss"],
            metrics["load_balance"],
        )

    def test_layer_metrics_use_actual_fusion_not_descriptor_alpha(self):
        descriptor_alpha = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
        gamma = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        stage = EnhancementStageOutput(
            layer_weights=gamma,
            expert_layer_weights=torch.ones(1, 2, 4),
            shape_fusion_layer_weights=None,
            aux_norm_ratio=torch.ones(1),
        )
        metrics = layer_fusion_metrics(
            {
                "level_weights": descriptor_alpha,
                "level_ids": (3, 6, 9, 12),
                "phase_b_moe": stage,
            }
        )
        self.assertAlmostEqual(float(metrics["layer_weight_3"]), 0.1)
        self.assertAlmostEqual(float(metrics["layer_weight_12"]), 0.4)
        torch.testing.assert_close(
            metrics["level_weight_max"],
            torch.tensor(0.4),
        )

    def test_enhancement_norms_and_compatibility_alias(self):
        alpha = torch.full((2, 4), 0.25)
        stage = EnhancementStageOutput(
            layer_weights=alpha,
            expert_layer_weights=None,
            shape_fusion_layer_weights=alpha,
            aux_norm_ratio=torch.tensor([1.0, 3.0]),
            fused_token_norm=torch.tensor([2.0, 4.0]),
            enhanced_token_norm=torch.tensor([5.0, 7.0]),
        )
        metrics = enhancement_metrics(stage)
        self.assertEqual(float(metrics["enhancement_aux_ratio"]), 2.0)
        self.assertEqual(float(metrics["fused_token_norm"]), 3.0)
        self.assertEqual(float(metrics["enhanced_token_norm"]), 6.0)
        self.assertAlmostEqual(
            float(metrics["shape_fusion_layer_weight_entropy"]),
            math.log(4),
            places=6,
        )
        self.assertNotIn("fused_level_weight_entropy", metrics)


if __name__ == "__main__":
    unittest.main()
