#!/usr/bin/env python3
"""Evaluate completed D3-D13 runs and write a combined scientific summary."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.phase_c_comprehensive import (
    PHASE_C_COMPREHENSIVE_SPECS,
    matching_runs,
    newest_completed_run,
)


SUMMARY_FIELDS = (
    "ID",
    "experiment_name",
    "run_id",
    "seed",
    "val_dice",
    "test_dice",
    "test_iou",
    "test_hd",
    "test_hd95",
    "test_assd",
    "test_boundary_f1",
    "posterior_dice",
    "latent_kl",
    "mean_distance",
    "std_distance",
    "wasserstein2_squared",
    "route_kl",
    "exact_topk_set_agreement",
    "topk_overlap",
    "routing_js",
    "gamma_distance",
    "transfer_gap_dice",
    "total_parameters",
    "trainable_parameters",
    "checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "runs"
        / "phase_c_comprehensive"
        / "comprehensive_summary.csv",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def build_summary_row(
    *,
    ablation_id: str,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluation.get("metrics")
    phase_c = evaluation.get("phase_c")
    if not isinstance(metrics, dict) or not isinstance(phase_c, dict):
        raise ValueError("Phase-C evaluation payload is incomplete.")
    counts = phase_c.get("parameter_counts")
    if not isinstance(counts, dict):
        raise ValueError("Phase-C evaluation lacks parameter counts.")

    metric_mapping = {
        "test_dice": "dice",
        "test_iou": "iou",
        "test_hd": "hd",
        "test_hd95": "hd95",
        "test_assd": "assd",
        "test_boundary_f1": "boundary_f1",
        "posterior_dice": "posterior_dice",
        "latent_kl": "latent_kl",
        "mean_distance": "mean_distance",
        "std_distance": "std_distance",
        "wasserstein2_squared": "wasserstein2_squared",
        "route_kl": "route_kl",
        "exact_topk_set_agreement": "exact_topk_set_agreement",
        "topk_overlap": "topk_overlap",
        "routing_js": "routing_js",
        "gamma_distance": "gamma_distance",
        "transfer_gap_dice": "transfer_gap_dice",
    }
    missing = [name for name in metric_mapping.values() if name not in metrics]
    if missing:
        raise ValueError(
            "Phase-C evaluation lacks required metrics: " + ", ".join(missing)
        )
    if metadata.get("best_val_dice") is None:
        raise ValueError("Completed run metadata lacks best_val_dice.")

    row: dict[str, Any] = {
        "ID": ablation_id,
        "experiment_name": metadata["experiment_name"],
        "run_id": metadata["run_id"],
        "seed": metadata["seed"],
        "val_dice": metadata["best_val_dice"],
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "checkpoint": evaluation["checkpoint"],
    }
    row.update(
        {
            output_name: metrics[metric_name]
            for output_name, metric_name in metric_mapping.items()
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")

    evaluator = (
        PROJECT_ROOT
        / "scripts"
        / "evaluation"
        / "evaluate_phase_c_distill.py"
    )
    rows: list[dict[str, Any]] = []
    for spec in PHASE_C_COMPREHENSIVE_SPECS:
        run = newest_completed_run(matching_runs(spec, PROJECT_ROOT))
        if run is None:
            print(f"Skipping {spec.ablation_id}: no completed matching run.")
            continue

        checkpoint = run.path / "best.pt"
        output_dir = run.path / "evaluation" / "comprehensive_test"
        command = [
            sys.executable,
            str(evaluator),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--split",
            "test",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.data_root is not None:
            command.extend(["--data-root", str(args.data_root)])
        print(
            f"\n=== Evaluating {spec.ablation_id}: {run.path.name} ===",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        evaluation = _read_json(output_dir / "metrics.json")
        row = build_summary_row(
            ablation_id=spec.ablation_id,
            metadata=run.metadata,
            evaluation=evaluation,
        )
        _write_json(output_dir.parent / "comprehensive_summary.json", row)
        rows.append(row)

    if not rows:
        raise SystemExit("No completed comprehensive Phase-C runs were found.")
    output = args.output.expanduser().resolve()
    _write_csv(output, rows)
    print(f"\nCombined summary: {output}")


if __name__ == "__main__":
    main()
