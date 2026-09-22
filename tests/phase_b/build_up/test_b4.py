"""B4 privileged direct-routing tests."""

import unittest
from pathlib import Path

import torch

from src.configs import load_experiment_config
from src.models.phase_b import PrivilegedFusion, load_shape_teacher
from src.models.phase_b.studies.build_up import PhaseBB4ShapeDirect


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestB4(unittest.TestCase):
    def test_hq_has_512_dimensions_and_teacher_is_frozen(self):
        config = load_experiment_config(
            PROJECT_ROOT / "configs/phase_b/build_up/b4_shape_direct.yaml",
            project_root=PROJECT_ROOT,
        )
        teacher = load_shape_teacher(config["model"]["shape_teacher_checkpoint"])
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))
        fusion = PrivilegedFusion(descriptor_dim=256, shape_latent_dim=256)
        output = fusion(torch.randn(2, 256), torch.randn(2, 256))
        self.assertEqual(tuple(output.fused.shape), (2, 512))

    def test_missing_mask_fails_before_encoder_access(self):
        model = PhaseBB4ShapeDirect.__new__(PhaseBB4ShapeDirect)
        with self.assertRaisesRegex(ValueError, "requires masks"):
            model.forward(torch.randn(1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
