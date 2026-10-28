"""Reproducibility check: run the full pipeline twice from scratch and compare.

    python -m src.repro_check --train-path data/train.csv

Writes results/reproducibility_check.md with: date, platform, library
versions, whether both runs finished, and the largest difference between the
two runs' metrics / predictions. Run it in a clean virtual environment
(see scripts/clean_env_check.sh) on the real data before submitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import run

PACKAGES = ["pandas", "numpy", "scikit-learn", "catboost", "matplotlib"]


def _versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def compare_runs(dir_a: Path, dir_b: Path) -> dict:
    a = pd.read_csv(dir_a / "metrics.csv").set_index(["model", "version", "validation_scheme", "fold"])
    b = pd.read_csv(dir_b / "metrics.csv").set_index(["model", "version", "validation_scheme", "fold"])
    if not a.index.equals(b.index):
        return {"same_rows": False, "max_metric_diff": float("nan"), "max_prediction_diff": float("nan")}
    metric_diff = float((a["mape"] - b["mape"]).abs().max())

    pred_diff = 0.0
    for path in sorted((dir_a / "predictions").glob("*.csv.gz")):
        pa = pd.read_csv(path)
        pb = pd.read_csv(dir_b / "predictions" / path.name)
        if len(pa) != len(pb):
            return {"same_rows": False, "max_metric_diff": metric_diff, "max_prediction_diff": float("nan")}
        pred_diff = max(pred_diff, float(np.abs(pa["y_pred"] - pb["y_pred"]).max()))
    return {"same_rows": True, "max_metric_diff": metric_diff, "max_prediction_diff": pred_diff}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--report", default="results/reproducibility_check.md")
    parser.add_argument("--quick", action="store_true", help="smoke test only")
    parser.add_argument("--final-results-dir", default=None,
                        help="copy the first run's results here (e.g. results)")
    args = parser.parse_args()

    started = dt.datetime.now()
    status, details, cmp = "SUCCESS", "", {}
    with tempfile.TemporaryDirectory() as tmp:
        run1, run2 = Path(tmp) / "run1", Path(tmp) / "run2"
        try:
            run(args.train_path, run1, quick=args.quick, verbose=False)
            run(args.train_path, run2, quick=args.quick, verbose=False)
            cmp = compare_runs(run1, run2)
            if not cmp["same_rows"] or cmp["max_metric_diff"] != 0.0 or cmp["max_prediction_diff"] > 1e-6:
                status = "METRICS DIFFER"
            if args.final_results_dir and status == "SUCCESS":
                shutil.copytree(run1, args.final_results_dir, dirs_exist_ok=True)
        except Exception as exc:  # recorded in the report, not hidden
            status, details = "FAILED", f"{type(exc).__name__}: {exc}"

    lines = [
        "# Reproducibility check",
        "",
        f"- Date: {started:%Y-%m-%d %H:%M}",
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- Python: {platform.python_version()}",
        *(f"- {k}: {v}" for k, v in _versions().items()),
        f"- Command: `python -m src.repro_check --train-path {Path(args.train_path).name}"
        f"{' --quick' if args.quick else ''}`",
        f"- Run time: {(dt.datetime.now() - started).total_seconds() / 60:.1f} min",
        f"- Pipeline finished without errors (two independent runs): "
        f"{'yes' if status in ('SUCCESS', 'METRICS DIFFER') else 'NO'}",
    ]
    if cmp:
        lines += [
            f"- Same rows in both runs: {cmp['same_rows']}",
            f"- Max |MAPE difference| between runs: {cmp['max_metric_diff']:.6g}",
            f"- Max |prediction difference| between runs: {cmp['max_prediction_diff']:.6g}",
        ]
    lines += [f"- **Result: {status}**"]
    if details:
        lines += ["", f"Error: `{details}`"]
    if args.quick:
        lines += ["", "> Smoke test (--quick) - not a valid reproducibility confirmation."]

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
