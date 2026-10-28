"""Run the whole project in order and write everything to results/.

    python -m src.pipeline --train-path data/train.csv

Steps: baselines -> CatBoost (raw/log, default/tuned) -> error analysis ->
summary table. All decisions use the validation folds; the 2016 test is
scored once per final model.

--quick uses a tiny grid for smoke tests; its numbers are NOT results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import evaluate_baselines
from src.error_analysis import run_error_analysis
from src.experiments import (
    prepare_data, run_catboost_experiments, select_final_model, tuning_effect,
)
from src.report_tables import summary_table, to_markdown, update_readme
from src.results import check_same_evaluation_rows
from src.model import MODEL_REGISTRY


def run(train_path, results_dir="results", quick=False, readme=None, verbose=True) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(train_path)

    base_metrics, coverage = evaluate_baselines(df, results_dir)
    cb = run_catboost_experiments(df, results_dir, quick=quick, verbose=verbose)

    metrics = pd.read_csv(results_dir / "metrics.csv")
    check_same_evaluation_rows(
        results_dir, ["seasonal_naive_last_year", "group_median",
                      "ridge_calendar_one_hot", *MODEL_REGISTRY],
    )

    final_model = select_final_model(metrics)  # chosen on validation only
    tuning_effect(metrics).to_csv(results_dir / "tuning_effect.csv", index=False)
    analysis = run_error_analysis(
        results_dir, final_model, train_df=df, focus_country="Norway"
    )

    table = summary_table(metrics)
    table_md = to_markdown(table, final_model)
    (results_dir / "summary_table.md").write_text(
        table_md + f"\n\nFinal model (lowest validation MAPE): {final_model}\n",
        encoding="utf-8",
    )
    if readme is not None:
        update_readme(readme, table_md)

    if verbose:
        print("\n" + table_md)
        print(f"\nFinal model (chosen on validation): {final_model}")
    return {"metrics": metrics, "final_model": final_model, "coverage": coverage,
            "analysis": analysis, "catboost": cb}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--readme", default=None, help="update results table in this README")
    parser.add_argument("--quick", action="store_true", help="smoke test only")
    args = parser.parse_args()
    run(args.train_path, args.results_dir, args.quick, args.readme)


if __name__ == "__main__":
    main()
