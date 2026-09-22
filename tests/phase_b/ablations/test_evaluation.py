"""Metrics, transfer comparison, CSV and benchmark tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluation.evaluate_phase_b_ablation import (
    benchmark_forward_paths,
    evaluate_paths,
    mask_geometry,
    model_efficiency_metadata,
    resolve_evaluation_paths,
    routing_path_comparison,
    write_per_sample_csv,
)
from src.models import SegmentationOutput
from src.models.phase_b import EnhancementStageOutput, RoutingOutput


def _routing(probabilities, indices):
    dense = torch.tensor(probabilities, dtype=torch.float32)
    selected = torch.tensor(indices, dtype=torch.long)
    active = dense.gather(1, selected)
    active = active / active.sum(dim=1, keepdim=True)
    sparse = torch.zeros_like(dense).scatter(1, selected, active)
    return RoutingOutput(
        logits=dense.log(),
        dense_probs=dense,
        routing_probs=sparse,
        expert_indices=selected,
    )


class _TransferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.router = SimpleNamespace(active_experts=2)
        self.image_descriptor = SimpleNamespace(levels=(3, 6, 9, 12))
        self.enhancement_mode = "hierarchical"

    def forward(self, images, masks=None):
        batch_size, _, height, width = images.shape
        posterior = masks is not None
        value = 1.0 if posterior else -1.0
        logits = torch.full(
            (batch_size, 1, height, width),
            value,
            device=images.device,
        ) * self.scale
        if posterior:
            probabilities = [[0.6, 0.3, 0.05, 0.05]] * batch_size
            indices = [[0, 1]] * batch_size
            source = "posterior"
        else:
            probabilities = [[0.6, 0.05, 0.3, 0.05]] * batch_size
            indices = [[0, 2]] * batch_size
            source = "prior"
        routing = _routing(probabilities, indices)
        router_stage = SimpleNamespace(
            source=source,
            routing=routing,
            latent_kl=None,
            balance=torch.tensor(1.0),
        )
        alpha = torch.full((batch_size, 4), 0.25, device=images.device)
        enhancement = EnhancementStageOutput(
            layer_weights=alpha,
            expert_layer_weights=torch.ones(batch_size, 2, 4),
            shape_fusion_layer_weights=None,
            aux_norm_ratio=torch.ones(batch_size),
            fused_token_norm=torch.full((batch_size,), 2.0),
            enhanced_token_norm=torch.full((batch_size,), 3.0),
        )
        return SegmentationOutput(
            logits=logits,
            diagnostics={
                "phase_b_router": router_stage,
                "phase_b_moe": enhancement,
                "level_ids": (3, 6, 9, 12),
            },
        )


class TestAblationEvaluation(unittest.TestCase):
    def test_path_resolution(self):
        self.assertEqual(
            resolve_evaluation_paths("phase_b_moe", "auto"),
            ("posterior", "prior"),
        )
        self.assertEqual(
            resolve_evaluation_paths("phase_b_a2_direct_fuse", "auto"),
            ("posterior",),
        )
        self.assertEqual(
            resolve_evaluation_paths("phase_b_a1_no_expert", "auto"),
            ("image",),
        )
        with self.assertRaises(ValueError):
            resolve_evaluation_paths("phase_b_a2_direct_fuse", "prior")

    def test_geometry_is_finite_for_empty_and_square_masks(self):
        empty = mask_geometry(np.zeros((8, 8), dtype=np.uint8))
        self.assertTrue(all(np.isfinite(value) for value in empty.values()))
        self.assertTrue(all(value == 0.0 for value in empty.values()))

        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 1
        geometry = mask_geometry(mask)
        self.assertEqual(geometry["lesion_area"], 4.0)
        self.assertEqual(geometry["lesion_perimeter"], 8.0)
        self.assertAlmostEqual(geometry["solidity"], 1.0)
        self.assertAlmostEqual(geometry["eccentricity"], 0.0)
        self.assertAlmostEqual(
            geometry["boundary_complexity"],
            1.0 / geometry["circularity"],
        )

    def test_routing_comparison_uses_categorical_kl_and_set_agreement(self):
        posterior = {
            "phase_b_router": SimpleNamespace(
                routing=_routing([[0.6, 0.3, 0.05, 0.05]], [[0, 1]])
            )
        }
        prior = {
            "phase_b_router": SimpleNamespace(
                routing=_routing([[0.6, 0.05, 0.3, 0.05]], [[0, 2]])
            )
        }
        metrics = routing_path_comparison(posterior, prior)
        self.assertGreater(float(metrics["categorical_routing_kl"][0]), 0.0)
        self.assertEqual(float(metrics["topk_exact_match"][0]), 0.0)
        self.assertAlmostEqual(float(metrics["topk_jaccard"][0]), 1.0 / 3.0)

    def test_evaluate_both_paths_reports_gap_alias_and_csv_fields(self):
        batch = {
            "image": torch.zeros(2, 3, 8, 8),
            "mask": torch.ones(2, 1, 8, 8),
            "sample_id": ["a", "b"],
        }
        aggregates, rows, benchmark_batch = evaluate_paths(
            model=_TransferModel(),
            loader=[batch],
            device=torch.device("cpu"),
            paths=("posterior", "prior"),
            criterion=torch.nn.BCEWithLogitsLoss(),
            threshold=0.5,
            boundary_tolerance=2.0,
        )
        comparison = aggregates["posterior_prior"]
        self.assertEqual(
            comparison["posterior_prior_dice_gap"],
            comparison["transfer_gap_dice"],
        )
        self.assertIn("categorical_routing_kl", comparison)
        self.assertIn("posterior_expert_indices", rows[0])
        self.assertIn("prior_routing_dense_probs", rows[0])
        self.assertIn("posterior_layer_weights", rows[0])
        self.assertEqual(tuple(benchmark_batch["images"].shape), (2, 3, 8, 8))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_sample.csv"
            write_per_sample_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as file:
                saved = list(csv.DictReader(file))
            self.assertEqual([row["sample_id"] for row in saved], ["a", "b"])
            self.assertIn("solidity", saved[0])

    def test_efficiency_metadata_and_cpu_benchmark(self):
        model = _TransferModel()
        metadata = model_efficiency_metadata(model, "phase_b_moe")
        self.assertEqual(metadata["active_experts"], 2)
        self.assertEqual(metadata["expert_calls_per_sample"], 8)

        batch = {
            "images": torch.zeros(1, 3, 8, 8),
            "targets": torch.ones(1, 1, 8, 8),
        }
        result = benchmark_forward_paths(
            model=model,
            batch=batch,
            paths=("posterior",),
            warmups=0,
            iterations=2,
        )["posterior"]
        self.assertGreater(result["latency_ms"], 0.0)
        self.assertGreater(result["throughput_samples_per_second"], 0.0)
        self.assertIsNone(result["peak_allocated_cuda_bytes"])


if __name__ == "__main__":
    unittest.main()
