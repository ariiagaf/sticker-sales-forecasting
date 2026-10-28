"""Build summary tables from results/metrics.csv (no hand-copied numbers)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.results import SCHEME_TEST, SCHEME_VALIDATION

MODEL_ORDER = [
    "group_median", "seasonal_naive_last_year", "ridge_calendar_one_hot",
    "catboost_raw", "catboost_log", "catboost_raw_tuned", "catboost_log_tuned",
]
START, END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    val = (metrics[metrics["validation_scheme"] == SCHEME_VALIDATION]
           .groupby(["model", "version"])["mape"].mean().rename("val"))
    test = (metrics[metrics["validation_scheme"] == SCHEME_TEST]
            .groupby(["model", "version"])["mape"].mean().rename("test"))
    out = pd.concat([val, test], axis=1).reset_index()
    out["order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def to_markdown(table: pd.DataFrame, final_model: str | None = None) -> str:
    lines = [
        "| Model | Validation MAPE, % (mean of folds 2013-2015) | Test MAPE, % (2016) |",
        "|---|---|---|",
    ]
    for _, r in table.iterrows():
        name = f"{r['model']} ({r['version']})"
        if final_model and r["model"] == final_model:
            name = f"**{name}**"
        lines.append(f"| {name} | {r['val']:.2f} | {r['test']:.2f} |")
    return "\n".join(lines)


def update_readme(readme_path: str | Path, table_md: str) -> bool:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    new = re.sub(
        f"{re.escape(START)}.*?{re.escape(END)}",
        lambda _: f"{START}\n{table_md}\n{END}", text, flags=re.S,
    )
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    import argparse

    from src.experiments import select_final_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    metrics = pd.read_csv(Path(args.results_dir) / "metrics.csv")
    table_md = to_markdown(summary_table(metrics), select_final_model(metrics))
    print(table_md)
    print("README updated." if update_readme(args.readme, table_md) else "README markers not found.")


if __name__ == "__main__":
    main()
