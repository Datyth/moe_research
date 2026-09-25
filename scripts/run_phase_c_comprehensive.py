#!/usr/bin/env python3
"""Launch the ten new Phase-C comprehensive ablations in fixed order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.phase_c_comprehensive import (
    PHASE_C_COMPREHENSIVE_SPECS,
    matching_runs,
    newest_completed_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restart-uncheckpointed",
        action="store_true",
        help=(
            "Leave matching interrupted runs without last.pt untouched and "
            "start a new timestamped run."
        ),
    )
    return parser.parse_args()


def main(*, restart_uncheckpointed: bool = False) -> None:
    training_cli = PROJECT_ROOT / "scripts" / "run_experiment.py"
    for spec in PHASE_C_COMPREHENSIVE_SPECS:
        print(
            f"\n=== {spec.ablation_id}: {spec.experiment_name} ===",
            flush=True,
        )
        matches = matching_runs(spec, PROJECT_ROOT)
        completed = newest_completed_run(matches)
        if completed is not None:
            print(f"Skipping completed run: {completed.path}", flush=True)
            continue

        unfinished = [run for run in matches if not run.is_completed]
        if unfinished:
            latest = unfinished[-1]
            resumable = [
                run for run in unfinished if (run.path / "last.pt").is_file()
            ]
            if resumable:
                latest = resumable[-1]
                raise SystemExit(
                    "Matching checkpointed run found; refusing to create a "
                    "duplicate. Resume it with:\n"
                    f"  {sys.executable} {training_cli} --resume {latest.path}"
                )
            if not restart_uncheckpointed:
                raise SystemExit(
                    "Matching interrupted run has no last.pt and cannot be "
                    "resumed. It was left untouched. Restart the sweep with:\n"
                    f"  {sys.executable} {Path(__file__).resolve()} "
                    "--restart-uncheckpointed"
                )
            print(
                "Leaving uncheckpointed interrupted run untouched and "
                f"starting a fresh run: {latest.path}",
                flush=True,
            )

        subprocess.run(
            [
                sys.executable,
                str(training_cli),
                "--config",
                str(spec.config_path(PROJECT_ROOT)),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    arguments = parse_args()
    main(restart_uncheckpointed=arguments.restart_uncheckpointed)
