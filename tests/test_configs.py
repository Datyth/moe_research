"""Static validation of every experiment YAML in configs/.

These checks run on CPU without the datasets present, and exist to catch config
mistakes long before a multi-hour training run starts:

* every runnable config loads through the real loader (`extends` chain,
  required sections, supported scheduler names, path resolution);
* `experiment.name` never drifts from the file name (run folders depend on it);
* the E-SAM ablation matrix (E0..E3) is complete for all four prepared
  datasets, and each row states `use_moe`/`use_lpeg` explicitly instead of
  relying on EsamModel's defaults (both default to True);
* E-SAM and promptless-SAM `model.image_size` match `dataset.image_size`;
* every prepared dataset has a frozen-encoder promptless-SAM baseline;
* `dataset.version` matches the version recorded inside the tracked manifest;
* `warmup_poly` gets enough iterations for its warmup (it raises otherwise).
"""

from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path
from typing import Any

from src.configs.experiment import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# `*_common.yaml` files are hyperparameter bases on purpose: they omit
# `experiment` and `model`, so they are not runnable by themselves.
BASE_SUFFIX = "_common.yaml"

# Datasets with a prepared, integrity-audited split manifest.
DATASETS = ("acdc", "amos22", "isic2018", "synapse")

# Paper Table 2 ablation rows -> (use_moe, use_lpeg).
ABLATION_FLAGS: dict[str, tuple[bool, bool]] = {
    "e0": (False, False),
    "e1": (True, False),
    "e2": (False, True),
    "e3": (True, True),
}

# Legacy UNet baselines predate the "experiment.name == file stem" convention
# and are referenced by their run-folder names in README/docs, so they are
# exempt. Every E-SAM config must follow the convention.
LEGACY_EXPERIMENT_NAMES = {
    "unet": "unet_isic2018",
    "acdc_unet": "unet_acdc",
}

SUPPORTED_SCHEDULERS = {"none", "cosine", "reduce_on_plateau", "warmup_poly"}


def _config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def _runnable_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in _config_paths():
        if path.name.endswith(BASE_SUFFIX):
            continue
        configs[path.stem] = load_experiment_config(path, project_root=REPO_ROOT)
    return configs


class TestConfigLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = _runnable_configs()

    def test_configs_are_present(self) -> None:
        self.assertGreaterEqual(len(self.configs), len(DATASETS) * 5)

    def test_every_config_loads(self) -> None:
        # `load_experiment_config` raises for unknown schedulers, missing
        # sections, bad paths and circular `extends`; an empty failure list here
        # means every config is loadable as written.
        failures: dict[str, str] = {}
        for path in _config_paths():
            if path.name.endswith(BASE_SUFFIX):
                continue
            try:
                load_experiment_config(path, project_root=REPO_ROOT)
            except Exception as error:  # noqa: BLE001 - reported, not raised
                failures[path.name] = f"{type(error).__name__}: {error}"
        self.assertEqual(failures, {})

    def test_experiment_name_matches_file_stem(self) -> None:
        for stem, config in self.configs.items():
            expected = LEGACY_EXPERIMENT_NAMES.get(stem, stem)
            self.assertEqual(config["experiment"]["name"], expected, stem)

    def test_common_bases_are_intentionally_incomplete(self) -> None:
        for path in _config_paths():
            if not path.name.endswith(BASE_SUFFIX):
                continue
            with self.assertRaises(ValueError) as context:
                load_experiment_config(path, project_root=REPO_ROOT)
            message = str(context.exception)
            self.assertTrue(
                "experiment" in message or "model" in message,
                f"{path.name}: {message}",
            )

    def test_scheduler_names_are_supported(self) -> None:
        for stem, config in self.configs.items():
            self.assertIn(config["scheduler"]["name"], SUPPORTED_SCHEDULERS, stem)

    def test_dataset_paths_are_resolved_and_manifests_tracked(self) -> None:
        for stem, config in self.configs.items():
            dataset = config["dataset"]
            self.assertTrue(Path(dataset["root"]).is_absolute(), stem)
            manifest = Path(dataset["manifest"])
            self.assertTrue(manifest.is_file(), f"{stem}: missing {manifest}")

    def test_manifest_version_matches_config(self) -> None:
        for stem, config in self.configs.items():
            dataset = config["dataset"]
            with Path(dataset["manifest"]).open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(
                manifest.get("version"),
                dataset["version"],
                f"{stem}: dataset.version disagrees with the manifest",
            )
            for split in ("training", "validation", "test"):
                self.assertGreater(len(manifest[split]), 0, f"{stem}: {split}")

    def test_warmup_poly_has_enough_steps(self) -> None:
        for stem, config in self.configs.items():
            scheduler = config["scheduler"]
            if scheduler["name"] != "warmup_poly":
                continue
            with Path(config["dataset"]["manifest"]).open(
                "r", encoding="utf-8"
            ) as file:
                train_count = len(json.load(file)["training"])
            steps_per_epoch = math.ceil(train_count / config["training"]["batch_size"])
            total_steps = steps_per_epoch * config["training"]["epochs"]
            self.assertGreater(
                total_steps,
                scheduler["warmup_steps"],
                f"{stem}: warmup_poly needs warmup_steps < total_steps",
            )


class TestPromptlessSamConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = _runnable_configs()

    def test_baseline_exists_for_every_dataset(self) -> None:
        for dataset in DATASETS:
            stem = f"{dataset}_sam"
            self.assertIn(stem, self.configs, stem)
            model = self.configs[stem]["model"]
            self.assertEqual(model["name"], "sam", stem)
            self.assertTrue(model["freeze_image_encoder"], stem)
            self.assertTrue(model["freeze_prompt_encoder"], stem)
            self.assertNotIn("use_moe", model, stem)
            self.assertNotIn("use_lpeg", model, stem)

    def test_synapse_ct_native_protocol_is_explicit(self) -> None:
        config = self.configs["synapse_ct_sam"]
        dataset = config["dataset"]
        self.assertEqual(dataset["name"], "synapse_ct")
        self.assertEqual(dataset["num_classes"], 9)
        self.assertEqual(dataset["image_size"], [224, 224])
        self.assertEqual(config["training"]["epochs"], 400)

        with Path(dataset["manifest"]).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        self.assertTrue(manifest["split_metadata"]["validation_is_test"])
        self.assertEqual(manifest["validation"], manifest["test"])
        self.assertEqual(
            manifest["training"][0]["expected_samples"], 1280
        )
        self.assertEqual(manifest["training"][0]["expected_cases"], 18)
        self.assertEqual(manifest["test"][0]["expected_cases"], 12)

    def test_image_size_matches_dataset(self) -> None:
        for stem, config in self.configs.items():
            if config["model"]["name"] != "sam":
                continue
            height, width = config["dataset"]["image_size"]
            self.assertEqual(height, width, stem)
            self.assertEqual(config["model"]["image_size"], height, stem)

    def test_baseline_uses_same_recipe_as_e0(self) -> None:
        for dataset in DATASETS:
            baseline = self.configs[f"{dataset}_sam"]
            e0 = self.configs[f"{dataset}_e0"]
            for section in ("dataset", "loss", "optimizer", "scheduler", "training"):
                self.assertEqual(baseline[section], e0[section], f"{dataset}:{section}")
            self.assertEqual(baseline["seed"], e0["seed"], dataset)


class TestEsamAblationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = _runnable_configs()

    def test_matrix_is_complete_for_every_dataset(self) -> None:
        for dataset in DATASETS:
            for row in ABLATION_FLAGS:
                self.assertIn(f"{dataset}_{row}", self.configs, f"{dataset}_{row}")

    def test_esam_configs_state_ablation_flags_explicitly(self) -> None:
        for stem, config in self.configs.items():
            if config["model"]["name"] != "esam":
                continue
            for flag in ("use_moe", "use_lpeg"):
                self.assertIn(flag, config["model"], f"{stem}: {flag} is implicit")
                self.assertIsInstance(config["model"][flag], bool, stem)

    def test_ablation_rows_match_the_paper_flags(self) -> None:
        for dataset in DATASETS:
            observed = {}
            for row, expected in ABLATION_FLAGS.items():
                model = self.configs[f"{dataset}_{row}"]["model"]
                observed[row] = (model["use_moe"], model["use_lpeg"])
                self.assertEqual(observed[row], expected, f"{dataset}_{row}")
            # The four rows must be distinguishable, otherwise the ablation
            # measures nothing.
            self.assertEqual(len(set(observed.values())), 4, dataset)

    def test_esam_image_size_matches_dataset(self) -> None:
        for stem, config in self.configs.items():
            if config["model"]["name"] != "esam":
                continue
            height, width = config["dataset"]["image_size"]
            self.assertEqual(height, width, stem)
            self.assertEqual(config["model"]["image_size"], height, stem)

    def test_esam_rows_share_one_recipe_per_dataset(self) -> None:
        # Only `model` may differ between ablation rows of the same dataset.
        for dataset in DATASETS:
            reference = self.configs[f"{dataset}_e0"]
            for row in ("e1", "e2", "e3"):
                other = self.configs[f"{dataset}_{row}"]
                for section in ("dataset", "loss", "optimizer", "scheduler"):
                    self.assertEqual(
                        reference[section], other[section], f"{dataset}_{row}:{section}"
                    )
                self.assertEqual(
                    {k: v for k, v in reference["training"].items() if k != "device"},
                    {k: v for k, v in other["training"].items() if k != "device"},
                    f"{dataset}_{row}:training",
                )
                self.assertEqual(reference["seed"], other["seed"], f"{dataset}_{row}")

    def test_multiclass_rows_use_ce_dice(self) -> None:
        for stem, config in self.configs.items():
            if config["model"]["name"] != "esam":
                continue
            expected = "ce_dice" if config["dataset"]["task"] == "multiclass" else "bce_dice"
            self.assertEqual(config["loss"]["name"], expected, stem)

    def test_smoke_configs_exist_and_are_short(self) -> None:
        for dataset in DATASETS:
            smoke = self.configs[f"{dataset}_smoke"]
            self.assertEqual(smoke["training"]["epochs"], 1, dataset)
            self.assertEqual(smoke["scheduler"]["name"], "none", dataset)

    def test_registries_and_signatures_accept_every_config(self) -> None:
        """Catch typos that only surface after hours of data loading.

        `build_loss`/`build_model` forward every remaining YAML key as a
        constructor keyword, so an unknown key (or an unregistered name) fails
        at run time, not at config load time.
        """
        from src.data import DATASET_REGISTRY
        from src.losses import LOSS_REGISTRY
        from src.models.registry import MODEL_REGISTRY

        # execute_experiment injects these dataset fields into the model block.
        injected = {"task", "in_channels", "num_classes"}
        registries = {
            "dataset": dict(DATASET_REGISTRY),
            "model": dict(MODEL_REGISTRY),
            "loss": dict(LOSS_REGISTRY),
        }

        for stem, config in self.configs.items():
            for section, registry in registries.items():
                block = dict(config[section])
                name = block.pop("name")
                self.assertIn(name, registry, f"{stem}: {section} '{name}'")
                if section == "dataset":
                    continue
                parameters = inspect.signature(registry[name].__init__).parameters
                if any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                ):
                    continue
                accepted = set(parameters) - {"self"} | (
                    injected if section == "model" else set()
                )
                for key in block:
                    self.assertIn(key, accepted, f"{stem}: {section}.{key}")


if __name__ == "__main__":
    unittest.main()
