"""Shared manifest and run discovery for the Phase-C D3-D13 sweep."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.configs import load_experiment_config
from src.experiment import config_fingerprint


@dataclass(frozen=True)
class PhaseCComprehensiveSpec:
    ablation_id: str
    experiment_name: str
    config_filename: str

    def config_path(self, project_root: Path) -> Path:
        return (
            project_root
            / "configs"
            / "phase_c"
            / "comprehensive"
            / self.config_filename
        )


@dataclass(frozen=True)
class PhaseCComprehensiveRun:
    path: Path
    metadata: dict[str, Any]

    @property
    def is_completed(self) -> bool:
        return (
            self.metadata.get("status") == "completed"
            and (self.path / "best.pt").is_file()
        )


PHASE_C_COMPREHENSIVE_SPECS = (
    PhaseCComprehensiveSpec(
        "D3", "phase_c_d3_router_seg", "d3_router_seg.yaml"
    ),
    PhaseCComprehensiveSpec(
        "D4", "phase_c_d4_layer_router_seg", "d4_layer_router_seg.yaml"
    ),
    PhaseCComprehensiveSpec(
        "D6", "phase_c_d6_routing_route", "d6_routing_route.yaml"
    ),
    PhaseCComprehensiveSpec(
        "D7",
        "phase_c_d7_routing_route_latent",
        "d7_routing_route_latent.yaml",
    ),
    PhaseCComprehensiveSpec(
        "D8", "phase_c_d8_expert_coadapt_seg", "d8_expert_coadapt_seg.yaml"
    ),
    PhaseCComprehensiveSpec(
        "D9",
        "phase_c_d9_expert_coadapt_route",
        "d9_expert_coadapt_route.yaml",
    ),
    PhaseCComprehensiveSpec(
        "D10",
        "phase_c_d10_expert_coadapt_route_latent",
        "d10_expert_coadapt_route_latent.yaml",
    ),
    PhaseCComprehensiveSpec(
        "D11",
        "phase_c_d11_representation_moe_seg",
        "d11_representation_moe_seg.yaml",
    ),
    PhaseCComprehensiveSpec(
        "D12",
        "phase_c_d12_representation_moe_route",
        "d12_representation_moe_route.yaml",
    ),
    PhaseCComprehensiveSpec(
        "D13",
        "phase_c_d13_representation_moe_route_latent",
        "d13_representation_moe_route_latent.yaml",
    ),
)


def load_spec_config(
    spec: PhaseCComprehensiveSpec,
    project_root: Path,
) -> dict[str, Any]:
    config = load_experiment_config(
        spec.config_path(project_root),
        project_root=project_root,
    )
    if config["experiment"].get("ablation_id") != spec.ablation_id:
        raise ValueError(f"{spec.config_filename} has the wrong ablation ID.")
    if config["experiment"]["name"] != spec.experiment_name:
        raise ValueError(f"{spec.config_filename} has the wrong experiment name.")
    if config["seed"] != 42 or config["training"]["epochs"] != 100:
        raise ValueError(
            f"{spec.ablation_id} must use seed 42 and exactly 100 epochs."
        )
    return config


def matching_runs(
    spec: PhaseCComprehensiveSpec,
    project_root: Path,
) -> list[PhaseCComprehensiveRun]:
    """Return runs matching the resolved config, seed, name, and D ID."""

    config = load_spec_config(spec, project_root)
    expected_fingerprint = config_fingerprint(config)
    parent = (
        Path(config["experiment"]["output_root"])
        / config["experiment"]["name"]
    )
    matches: list[PhaseCComprehensiveRun] = []
    if not parent.is_dir():
        return matches
    for run_dir in sorted(path for path in parent.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("experiment_name") == spec.experiment_name
            and metadata.get("ablation_id") == spec.ablation_id
            and metadata.get("seed") == 42
            and metadata.get("config_fingerprint") == expected_fingerprint
        ):
            matches.append(PhaseCComprehensiveRun(run_dir, metadata))
    return matches


def newest_completed_run(
    runs: list[PhaseCComprehensiveRun],
) -> PhaseCComprehensiveRun | None:
    completed = [run for run in runs if run.is_completed]
    if not completed:
        return None
    return max(
        completed,
        key=lambda run: (str(run.metadata.get("ended_at", "")), run.path.name),
    )


__all__ = [
    "PHASE_C_COMPREHENSIVE_SPECS",
    "PhaseCComprehensiveRun",
    "PhaseCComprehensiveSpec",
    "load_spec_config",
    "matching_runs",
    "newest_completed_run",
]
