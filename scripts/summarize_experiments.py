#!/usr/bin/env python3
"""Summarize standardized experiment runs and aggregate repeated seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics

import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_NAMES = ("loss", "dice", "iou")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def collect_runs(runs_root: Path, experiment: str | None = None) -> list[dict[str, Any]]:
    pattern = f"{experiment}/*/test_metrics.json" if experiment else "*/*/test_metrics.json"
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob(pattern)):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        config_path = run_dir / "config.yaml"
        if not metadata_path.is_file() or not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as file:
            saved_config = yaml.safe_load(file)
        if not isinstance(saved_config, dict):
            raise ValueError(f"Expected a YAML mapping: {config_path}")

        metadata = _read_json(metadata_path)
        metrics = _read_json(metrics_path)
        if metadata.get("status") != "completed":
            continue
        row = {
            "experiment": str(metadata["experiment_name"]),
            "config_fingerprint": str(metadata["config_fingerprint"]),
            "run_id": str(metadata["run_id"]),
            "seed": int(metadata["seed"]),
        }
        for metric_name in METRIC_NAMES:
            row[metric_name] = float(metrics[metric_name])
        rows.append(row)
    return rows


def aggregate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["config_fingerprint"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for (experiment, fingerprint), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "experiment": experiment,
            "config_fingerprint": fingerprint,
            "n": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: item["seed"])),
        }
        for metric_name in METRIC_NAMES:
            values = [float(row[metric_name]) for row in group]
            aggregate[f"{metric_name}_mean"] = statistics.mean(values)
            aggregate[f"{metric_name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        aggregates.append(aggregate)
    return aggregates


def write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "row_type",
        "experiment",
        "config_fingerprint",
        "run_id",
        "seed",
        "n",
        "seeds",
        "loss",
        "dice",
        "iou",
        "loss_mean",
        "loss_std",
        "dice_mean",
        "dice_std",
        "iou_mean",
        "iou_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_type": "run", **row})
        for aggregate in aggregates:
            writer.writerow({"row_type": "aggregate", **aggregate})
    temporary_path.replace(path)


def _mean_std(mean: float, std: float | None) -> str:
    return f"{mean:.6f} ± {std:.6f}" if std is not None else f"{mean:.6f} ± N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = collect_runs(args.runs_root.expanduser().resolve(), args.experiment)
    if not rows:
        raise SystemExit("No completed runs with test_metrics.json were found.")
    aggregates = aggregate_runs(rows)
    write_summary_csv(args.output.expanduser().resolve(), rows, aggregates)

    print("Per-run test metrics")
    print("experiment\trun_id\tseed\tloss\tdice\tiou")
    for row in rows:
        print(
            f"{row['experiment']}\t{row['run_id']}\t{row['seed']}\t"
            f"{row['loss']:.6f}\t{row['dice']:.6f}\t{row['iou']:.6f}"
        )
    print("\nAggregate by experiment and config fingerprint")
    print("experiment\tn\tloss mean ± std\tdice mean ± std\tiou mean ± std")
    for aggregate in aggregates:
        print(
            f"{aggregate['experiment']}\t{aggregate['n']}\t"
            f"{_mean_std(aggregate['loss_mean'], aggregate['loss_std'])}\t"
            f"{_mean_std(aggregate['dice_mean'], aggregate['dice_std'])}\t"
            f"{_mean_std(aggregate['iou_mean'], aggregate['iou_std'])}"
        )
    print(f"\nSummary CSV: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
