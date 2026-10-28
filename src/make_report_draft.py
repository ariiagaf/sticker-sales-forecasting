"""Generate results/report_draft.md from the saved results.

Every number in the draft is read from results/ (metrics.csv, tuning tables,
error-analysis tables, reproducibility_check.md). Convert to PDF with e.g.
    pandoc results/report_draft.md -o project.pdf
Edit the wording as needed, but keep the numbers coming from the files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.experiments import select_final_model, tuning_effect
from src.report_tables import summary_table, to_markdown
from src.validation import VALIDATION_YEARS, TEST_YEAR, describe_scheme


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(results_dir: str | Path) -> str:
    rd = Path(results_dir)
    metrics = pd.read_csv(rd / "metrics.csv")
    final = select_final_model(metrics)
    table = summary_table(metrics)
    cov = pd.read_csv(rd / "seasonal_naive_coverage.csv")
    effect = tuning_effect(metrics)
    tuning = pd.read_csv(rd / "tuning_results.csv")
    by_country = pd.read_csv(rd / "error_analysis_by_country.csv")
    by_product = pd.read_csv(rd / "error_analysis_by_product.csv")
    by_store = pd.read_csv(rd / "error_analysis_by_store.csv")
    by_year = pd.read_csv(rd / "error_analysis_by_year.csv")
    summary = (rd / "error_analysis_summary.md").read_text(encoding="utf-8").strip()
    repro_path = rd / "reproducibility_check.md"
    repro = repro_path.read_text(encoding="utf-8").strip() if repro_path.exists() else (
        "_Run `bash scripts/clean_env_check.sh` and paste results/reproducibility_check.md here._"
    )

    mean_val = metrics[metrics["validation_scheme"] == "expanding_window"].groupby("model")["mape"].mean()
    test = metrics[metrics["validation_scheme"] == "final_test"].set_index("model")["mape"]
    base = "seasonal_naive_last_year"

    tuned_rows = []
    for name in ("catboost_raw_tuned", "catboost_log_tuned"):
        p = json.loads(metrics.loc[metrics["model"] == name, "params"].iloc[0])
        short = {k: p[k] for k in ("iterations", "depth", "learning_rate", "l2_leaf_reg")}
        tuned_rows.append(f"- `{name}`: {short}")

    delta = test[base] - test[final]
    verdict = (f"{final} beats {base} on the independent test by {delta:.2f} p.p."
               if delta > 0 else
               f"{final} does NOT beat {base} on the independent test ({delta:+.2f} p.p.).")

    return f"""# Forecasting Sticker Sales - Project Report

Kaggle Playground Series S5E1 (`playground-series-s5e1`). All numbers in this document are generated from `results/`.

## 1. Problem statement and validation scheme

Task: forecast daily `num_sold` for 90 (country, store, product) series. Metric: MAPE (`src/metrics.py`), identical for all models.
Rows with a missing target are dropped, never imputed.

Time periods (train data covers 2010-2016):

```
{describe_scheme()}
```

- Hyperparameters and model choice use only the validation folds ({", ".join(map(str, VALIDATION_YEARS))}).
- The test year {TEST_YEAR} is used only for the final evaluation: every final model is fitted once on 2010-{TEST_YEAR - 1} and scored once on {TEST_YEAR}. Test results never influenced tuning or model choice (enforced in code: `src/validation.py`, `src/tuning.py`).
- All models, baselines and CatBoost, are evaluated on identical rows (checked in code) with the same MAPE.

## 2. Baselines

- `group_median`: median `num_sold` of each (country, store, product).
- `seasonal_naive_last_year`: actual sales of the same date one year earlier, taken from the training data only; group median if that value is unavailable.
- `ridge_calendar_one_hot`: Ridge on calendar features and one-hot country/store/product (log1p target).

Because each evaluation window is one year long, `seasonal_naive_last_year` uses the previous year's **actual** sales for each date.
Share of forecasts that come from real last-year sales (the rest is the group-median fallback, e.g. 29 Feb or a missing last-year target):

{_md(cov, "{:.4f}")}

## 3. Models

CatBoost with 15 calendar features and categorical country/store/product, seed 42.
Two target modes with one interface (`src/model.py`): `catboost_raw` trains on `num_sold`; `catboost_log` trains on `log1p(num_sold)` and maps predictions back with `expm1`.
Each mode is evaluated with default parameters and with parameters tuned on the validation folds only (`catboost_raw_tuned`, `catboost_log_tuned`).

Selected hyperparameters:

{chr(10).join(tuned_rows)}

Full search results: `results/tuning_results.csv`. Top candidates:

{_md(tuning.groupby("target_transform").head(3).drop(columns=[c for c in tuning.columns if c.startswith("mape_fold")]), "{:.4f}")}

## 4. Results

{to_markdown(table, final)}

Final CatBoost model, chosen by validation MAPE: **{final}**. {verdict}

Effect of tuning (percentage points, default minus tuned; positive = tuning helped):

{_md(effect, "{:.3f}")}

Raw vs log target (mean validation MAPE / test MAPE): raw {mean_val["catboost_raw"]:.2f} / {test["catboost_raw"]:.2f}, log {mean_val["catboost_log"]:.2f} / {test["catboost_log"]:.2f}.

## 5. Error analysis (independent test, {TEST_YEAR})

{summary}

By country:

{_md(by_country, "{:.3f}")}

By store:

{_md(by_store, "{:.3f}")}

By product:

{_md(by_product, "{:.3f}")}

By year (validation years were used for tuning and are optimistic; only `test_{TEST_YEAR}` is independent):

{_md(by_year, "{:.3f}")}

Norway level check: see `results/error_analysis_norway_levels.csv` and the Norway section of `notebooks/04_evaluation.ipynb`.
Describe the Norway behaviour only as far as that table supports it.

## 6. Reproducibility

Install: `pip install -r requirements.txt` (pinned versions). Data: place `train.csv` in `data/`. Run: `python -m src.pipeline --train-path data/train.csv` (or notebooks 02 -> 03 -> 04).

{repro}

## 7. Conclusion and limitations

- {verdict}
- The raw-vs-log and tuning conclusions are those in section 4: base them on both the validation mean and the independent test.
- Only calendar and categorical features are used; no lag/rolling features.
- The test period is one year, so long-horizon behaviour (the real Kaggle test covers 2017-2019) is not evaluated; a multi-year window would require recursive forecasts for the seasonal baseline.
- Missing-target rows are dropped, not modelled.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    out = Path(args.results_dir) / "report_draft.md"
    out.write_text(build(args.results_dir), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
