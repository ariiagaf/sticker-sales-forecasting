"""Reading / writing experiment results (metrics and predictions).

Every model (baselines and CatBoost) writes through these helpers, so
`results/metrics.csv` and `results/predictions/*.csv.gz` all have the same
format and re-running an experiment REPLACES its own rows instead of
appending duplicates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = [
    "model",
    "version",
    "validation_scheme",
    "fold",
    "train_period",
    "validation_period",
    "mape",
    "params",
]
KEY_COLUMNS = ["model", "version", "validation_scheme", "fold"]

# validation_scheme values
SCHEME_VALIDATION = "expanding_window"
SCHEME_TEST = "final_test"

PREDICTION_COLUMNS = ["split", "date", "country", "store", "product", "y_true", "y_pred"]


def period_text(df: pd.DataFrame) -> str:
    return f"{df['date'].min().date()} to {df['date'].max().date()}"


def make_record(
    model: str,
    version: str,
    scheme: str,
    fold: str,
    train_part: pd.DataFrame,
    eval_part: pd.DataFrame,
    score: float,
    params: dict,
) -> dict:
    return {
        "model": model,
        "version": version,
        "validation_scheme": scheme,
        "fold": fold,
        "train_period": period_text(train_part),
        "validation_period": period_text(eval_part),
        "mape": round(float(score), 6),
        "params": json.dumps(params, sort_keys=True, default=str),
    }


def upsert_metrics(records: list[dict] | pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Insert rows into metrics.csv, replacing rows with the same key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new = pd.DataFrame(records)[METRIC_COLUMNS]
    if path.exists() and path.stat().st_size > 1:
        old = pd.read_csv(path)
        merged = old.merge(
            new[KEY_COLUMNS].drop_duplicates(),
            on=KEY_COLUMNS,
            how="left",
            indicator=True,
        )
        old = old[merged["_merge"].eq("left_only").to_numpy()]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out = out.sort_values(KEY_COLUMNS).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def prediction_path(results_dir: str | Path, model: str) -> Path:
    return Path(results_dir) / "predictions" / f"{model}.csv.gz"


def save_predictions(
    results_dir: str | Path,
    model: str,
    split: str,
    eval_part: pd.DataFrame,
    y_pred,
    target_col: str = "num_sold",
) -> None:
    """Store predictions of `model` for one split (fold / test)."""
    path = prediction_path(results_dir, model)
    path.parent.mkdir(parents=True, exist_ok=True)

    new = eval_part[["date", "country", "store", "product"]].copy().reset_index(drop=True)
    new.insert(0, "split", split)
    new["y_true"] = eval_part[target_col].to_numpy()
    new["y_pred"] = pd.Series(y_pred).to_numpy(dtype=float)
    new = new[PREDICTION_COLUMNS]

    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        old = old[old["split"] != split]
        new = pd.concat([old, new], ignore_index=True)

    new.sort_values(["split", "date", "country", "store", "product"]).to_csv(
        path, index=False, float_format="%.6f"
    )


def load_predictions(results_dir: str | Path, model: str) -> pd.DataFrame:
    path = prediction_path(results_dir, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved predictions for {model!r} at {path}. "
            "Run the baseline / model notebook (or src.pipeline) first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def check_same_evaluation_rows(results_dir: str | Path, models: list[str]) -> None:
    """Assert that all models were evaluated on identical rows in every split.

    Guarantees that baselines and CatBoost are compared on the same periods
    and the same (date, country, store, product) observations.
    """
    keys = ["split", "date", "country", "store", "product"]
    reference_name = models[0]
    reference = (
        load_predictions(results_dir, reference_name)[keys]
        .sort_values(keys)
        .reset_index(drop=True)
    )
    for name in models[1:]:
        other = (
            load_predictions(results_dir, name)[keys]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        if not reference.equals(other):
            raise AssertionError(
                f"{name} and {reference_name} were not evaluated on identical rows."
            )
