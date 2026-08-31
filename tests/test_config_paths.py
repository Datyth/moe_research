"""Tests for project-root and environment-variable path resolution."""

import os
import tempfile
import unittest
from pathlib import Path

from src.configs.experiment import _resolve_path


class TestResolvePath(unittest.TestCase):
    def test_relative_path_resolves_against_project_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resolved = _resolve_path("dataset/acdc", root, "dataset.root")
            self.assertEqual(Path(resolved), (root / "dataset" / "acdc").resolve())

    def test_absolute_path_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            resolved = _resolve_path(temporary_directory, Path("/anywhere"), "dataset.root")
            self.assertEqual(Path(resolved), Path(temporary_directory).resolve())

    def test_env_var_is_expanded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.environ["TEST_ACDC_DATA_ROOT"] = temporary_directory
            try:
                resolved = _resolve_path(
                    "${TEST_ACDC_DATA_ROOT}", Path("/anywhere"), "dataset.root"
                )
            finally:
                os.environ.pop("TEST_ACDC_DATA_ROOT")
            self.assertEqual(Path(resolved), Path(temporary_directory).resolve())

    def test_undefined_env_var_raises_clear_error(self):
        with self.assertRaises(ValueError) as context:
            _resolve_path(
                "${TEST_ACDC_MISSING_VAR}", Path("/anywhere"), "dataset.root"
            )
        self.assertIn("TEST_ACDC_MISSING_VAR", str(context.exception))

    def test_tilde_is_expanded(self):
        resolved = _resolve_path("~/somewhere", Path("/anywhere"), "dataset.root")
        self.assertTrue(Path(resolved).is_absolute())


if __name__ == "__main__":
    unittest.main()
