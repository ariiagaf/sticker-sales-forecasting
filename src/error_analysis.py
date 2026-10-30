"""Error analysis from SAVED predictions (no model is retrained here).

Reads results/predictions/<model>.csv.gz written by the baseline and CatBoost
runs, so the analysis uses exactly the numbers in results/metrics.csv and the
CatBoost / baseline comparison is on identical rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.results import load_predictions
from src.validation import test_name

BASELINE = "seasonal_naive_last_year"
GROUPS = ("country", "store", "product")


def _pct_err(df: pd.DataFrame, col: str) -> pd.Series:
    return (df[col] - df["y_true"]).abs() / df["y_true"] * 100


def load_pair(results_dir, model: str, baseline: str = BASELINE) -> pd.DataFrame:
    """Rows where both `model` and the baseline have predictions."""
    keys = ["split", "date", "country", "store", "product", "y_true"]
    m = load_predictions(results_dir, model).rename(columns={"y_pred": "pred_model"})
    b = load_predictions(results_dir, baseline).rename(columns={"y_pred": "pred_baseline"})
    pair = m.merge(b[[*keys, "pred_baseline"]], on=keys, how="inner", validate="one_to_one")
    if len(pair) != len(m) or len(pair) != len(b):
        raise AssertionError("Model and baseline were evaluated on different rows.")
    pair["ape_model"] = _pct_err(pair, "pred_model")
    pair["ape_baseline"] = _pct_err(pair, "pred_baseline")
    pair["signed_err_model"] = (pair["pred_model"] - pair["y_true"]) / pair["y_true"] * 100
    pair["signed_err_baseline"] = (pair["pred_baseline"] - pair["y_true"]) / pair["y_true"] * 100
    pair["year"] = pair["date"].dt.year
    return pair


def breakdown(pair: pd.DataFrame, by, model_label: str) -> pd.DataFrame:
    """MAPE of model and baseline by group; gap > 0 means model is better."""
    by = [by] if isinstance(by, str) else list(by)
    out = (
        pair.groupby(by)[["ape_model", "ape_baseline"]].mean().reset_index()
        .rename(columns={"ape_model": f"mape_{model_label}", "ape_baseline": f"mape_{BASELINE}"})
    )
    out["gap_baseline_minus_model"] = out[f"mape_{BASELINE}"] - out[f"mape_{model_label}"]
    out["model_better"] = out["gap_baseline_minus_model"] > 0
    return out.sort_values(f"mape_{model_label}", ascending=False).reset_index(drop=True)


def worst_series(pair: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return (
        pair.groupby(list(GROUPS))["ape_model"].mean()
        .sort_values(ascending=False).head(n).reset_index()
    )


def country_level_table(
    train_df: pd.DataFrame, pair: pd.DataFrame, country: str
) -> pd.DataFrame:
    """Per product and year: actual level in train data, and signed bias of
    model vs baseline on evaluated years. Lets one check a level-shift claim."""
    actual = (
        train_df[train_df["country"] == country]
        .assign(year=lambda d: d["date"].dt.year)
        .groupby(["product", "year"])["num_sold"].mean().rename("mean_actual").reset_index()
    )
    bias = (
        pair[pair["country"] == country]
        .groupby(["product", "year"])[["signed_err_model", "signed_err_baseline"]]
        .mean().reset_index()
    )
    return actual.merge(bias, on=["product", "year"], how="left")


def summarize(
    model_label: str,
    by_country: pd.DataFrame,
    by_product: pd.DataFrame,
    worst: pd.DataFrame,
    overall_model: float,
    overall_baseline: float,
    split: str,
) -> str:
    """Data-driven conclusion text (no hand-written numbers)."""
    mcol, bcol = f"mape_{model_label}", f"mape_{BASELINE}"
    verdict = "better" if overall_model < overall_baseline else "NOT better"
    lines = [
        f"On {split}, {model_label} has MAPE {overall_model:.2f}% vs "
        f"{overall_baseline:.2f}% for {BASELINE}: the model is {verdict} overall "
        f"({overall_baseline - overall_model:+.2f} p.p.)."
    ]
    worse_c = by_country[~by_country["model_better"]]
    if len(worse_c):
        parts = ", ".join(
            f"{r['country']} ({r[mcol]:.2f}% vs {r[bcol]:.2f}%)" for _, r in worse_c.iterrows()
        )
        lines.append(f"Countries where the model is worse than the baseline: {parts}.")
    else:
        lines.append("The model is better than the baseline in every country.")
    hardest_c = by_country.iloc[0]
    lines.append(f"Hardest country for the model: {hardest_c['country']} ({hardest_c[mcol]:.2f}%).")
    worse_p = by_product[~by_product["model_better"]]
    if len(worse_p):
        lines.append(
            "Products where the model is worse than the baseline: "
            + ", ".join(worse_p["product"]) + "."
        )
    hardest_p = by_product.iloc[0]
    lines.append(f"Hardest product: {hardest_p['product']} ({hardest_p[mcol]:.2f}%).")
    series = "; ".join(
        f"{r['country']} / {r['store']} / {r['product']} ({r['ape_model']:.2f}%)"
        for _, r in worst.iterrows()
    )
    lines.append(f"Worst series: {series}.")
    return "\n\n".join(lines)


def run_error_analysis(
    results_dir: str | Path,
    model: str,
    train_df: pd.DataFrame | None = None,
    focus_country: str | None = None,
    make_plot: bool = True,
) -> dict:
    """Compute and save all error-analysis tables for the final model.

    Tables use the independent test split. `by_year` also lists the validation
    years (marked by `split`), which were used for tuning and are optimistic.
    """
    results_dir = Path(results_dir)
    pair_all = load_pair(results_dir, model)
    pair = pair_all[pair_all["split"] == test_name()].copy()
    if pair.empty:
        raise ValueError("No test predictions found; run the model evaluation first.")

    tables = {
        "country": breakdown(pair, "country", model),
        "store": breakdown(pair, "store", model),
        "product": breakdown(pair, "product", model),
        "by_year": breakdown(pair_all, ["split", "year"], model).sort_values(["split", "year"]),
        "country_year": breakdown(pair_all, ["country", "year"], model).sort_values(["country", "year"]),
    }
    for name, table in tables.items():
        key = "year" if name == "by_year" else name
        table.to_csv(results_dir / f"error_analysis_by_{key}.csv", index=False)

    worst = worst_series(pair)
    worst.to_csv(results_dir / "error_analysis_worst_series.csv", index=False)

    overall_model = pair["ape_model"].mean()
    overall_baseline = pair["ape_baseline"].mean()
    text = summarize(
        model, tables["country"], tables["product"], worst,
        overall_model, overall_baseline, test_name(),
    )
    (results_dir / "error_analysis_summary.md").write_text(text + "\n", encoding="utf-8")

    out = {"pair": pair, "pair_all": pair_all, "tables": tables, "worst": worst, "summary": text}

    if train_df is not None:
        focus = focus_country or tables["country"].iloc[0]["country"]
        lvl = country_level_table(train_df, pair_all, focus)
        lvl.to_csv(results_dir / f"error_analysis_{focus.lower()}_levels.csv", index=False)
        out["focus_country"], out["focus_table"] = focus, lvl

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(worst), 1, figsize=(10, 3.6 * len(worst)), squeeze=False)
        for ax, (_, r) in zip(axes[:, 0], worst.iterrows()):
            sub = pair[
                (pair["country"] == r["country"]) & (pair["store"] == r["store"])
                & (pair["product"] == r["product"])
            ].sort_values("date")
            ax.plot(sub["date"], sub["y_true"], label="actual")
            ax.plot(sub["date"], sub["pred_model"], label=model)
            ax.plot(sub["date"], sub["pred_baseline"], label=BASELINE, alpha=0.6)
            ax.set_title(f"{r['country']} / {r['store']} / {r['product']} ({test_name()})")
            ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "error_analysis_worst_series.png", dpi=150)
        plt.close(fig)
    return out
