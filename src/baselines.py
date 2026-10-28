import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_date_features
from src.metrics import mape
from src.preprocessing import drop_missing_target, preprocess_data
from src.results import (
    SCHEME_TEST,
    SCHEME_VALIDATION,
    make_record,
    save_predictions,
    upsert_metrics,
)
from src.validation import (
    expanding_window_splits,
    fold_name,
    split_development_test,
    test_name,
)


TARGET_COL = "num_sold"
GROUP_COLS = ["country", "store", "product"]


def group_median_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    stats = (
        train_df.groupby(GROUP_COLS, as_index=False)[TARGET_COL]
        .median()
        .rename(columns={TARGET_COL: "prediction"})
    )
    fallback = train_df[TARGET_COL].median()

    pred = val_df[GROUP_COLS].merge(stats, on=GROUP_COLS, how="left")
    return pred["prediction"].fillna(fallback).to_numpy()


def _last_year_history(train_df: pd.DataFrame) -> pd.DataFrame:
    """Training sales shifted forward by one calendar year.

    A row dated D in train becomes a candidate forecast for D + 1 year.
    Only the dates that exist in train are shifted, so a forecast for a date
    is only available if that date minus one year is inside the training data.
    (29 Feb shifts onto 28 Feb; the real 28 Feb of the previous year is kept.)
    """
    history = (
        train_df[["date", *GROUP_COLS, TARGET_COL]]
        .sort_values("date")
        .copy()
    )
    history["date"] = history["date"] + pd.DateOffset(years=1)
    history = history.rename(columns={TARGET_COL: "prediction"})
    return history.drop_duplicates(subset=["date", *GROUP_COLS], keep="first")


def seasonal_naive_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    """Value of the same date one year earlier, group median if unavailable.

    The only data used is `train_df` (everything strictly before val_df), so
    no future information is used. For a one-year evaluation window, the value
    for each date comes from the actual sales of the previous year. If the
    evaluation window is longer than one year, dates beyond the first year have
    no previous-year value in train_df and silently use the group median
    (see `seasonal_naive_coverage`).
    """
    history = _last_year_history(train_df)

    pred = val_df[["date", *GROUP_COLS]].merge(
        history,
        on=["date", *GROUP_COLS],
        how="left",
    )
    if len(pred) != len(val_df):
        raise AssertionError("Seasonal naive merge changed the number of rows.")

    missing_mask = pred["prediction"].isna()
    if missing_mask.any():
        fallback = group_median_predict(
            train_df, val_df.loc[missing_mask.to_numpy()]
        )
        pred.loc[missing_mask, "prediction"] = fallback

    return pred["prediction"].to_numpy()


def seasonal_naive_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """How many forecasts come from real last-year sales vs. the fallback."""
    history = _last_year_history(train_df)
    pred = val_df[["date", *GROUP_COLS]].merge(
        history, on=["date", *GROUP_COLS], how="left"
    )
    from_last_year = int(pred["prediction"].notna().sum())
    total = int(len(pred))
    return {
        "rows": total,
        "from_last_year": from_last_year,
        "fallback_group_median": total - from_last_year,
        "share_from_last_year": round(from_last_year / total, 6),
    }


def ridge_calendar_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alpha: float = 1.0,
) -> np.ndarray:
    train_features = add_date_features(train_df)
    val_features = add_date_features(val_df)

    numeric_features = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "is_weekend",
        "time_idx",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    categorical_features = GROUP_COLS

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", encoder, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    feature_cols = numeric_features + categorical_features
    y_train = np.log1p(train_features[TARGET_COL])

    model.fit(train_features[feature_cols], y_train)
    prediction = np.expm1(model.predict(val_features[feature_cols]))

    return np.clip(prediction, 1e-6, None)


BASELINES = [
    {
        "model": "group_median",
        "version": "v1",
        "params": {"group_cols": GROUP_COLS},
        "predict": group_median_predict,
    },
    {
        "model": "seasonal_naive_last_year",
        "version": "v1",
        "params": {"lag": "1 year", "fallback": "group_median"},
        "predict": seasonal_naive_predict,
    },
    {
        "model": "ridge_calendar_one_hot",
        "version": "v1",
        "params": {"alpha": 1.0, "target_transform": "log1p"},
        "predict": ridge_calendar_predict,
    },
]


def evaluate_baselines(
    df: pd.DataFrame,
    results_dir: str | Path | None = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all baselines with the shared temporal scheme.

    Returns (metrics, seasonal_coverage). With `results_dir` set, metrics rows
    and the predictions of every split are written there.

    Every baseline is trained / evaluated on exactly the same splits as the
    CatBoost models: expanding-window validation folds, and the independent
    test year (fit on all development data, scored once).
    """
    dev_df, test_df = split_development_test(df)

    splits = []
    for number, (train_part, val_part) in enumerate(
        expanding_window_splits(dev_df), start=1
    ):
        splits.append(
            (SCHEME_VALIDATION, fold_name(number, val_part), train_part, val_part)
        )
    splits.append((SCHEME_TEST, test_name(), dev_df, test_df))

    records = []
    coverage_rows = []

    for baseline in BASELINES:
        for scheme, split_label, train_part, eval_part in splits:
            prediction = baseline["predict"](train_part, eval_part)
            score = mape(eval_part[TARGET_COL], prediction)
            records.append(
                make_record(
                    baseline["model"],
                    baseline["version"],
                    scheme,
                    split_label,
                    train_part,
                    eval_part,
                    score,
                    baseline["params"],
                )
            )
            if results_dir is not None:
                save_predictions(
                    results_dir,
                    baseline["model"],
                    split_label,
                    eval_part,
                    prediction,
                )
            if baseline["model"] == "seasonal_naive_last_year":
                coverage_rows.append(
                    {
                        "split": split_label,
                        **seasonal_naive_coverage(train_part, eval_part),
                    }
                )

    metrics = pd.DataFrame(records)
    coverage = pd.DataFrame(coverage_rows)

    if results_dir is not None:
        upsert_metrics(metrics, Path(results_dir) / "metrics.csv")
        coverage.to_csv(
            Path(results_dir) / "seasonal_naive_coverage.csv", index=False
        )

    return metrics, coverage


def load_training_data(train_path: str | Path) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    train = preprocess_data(train)
    train = drop_missing_target(train)
    return train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    train_df = load_training_data(args.train_path)
    metrics_df, coverage = evaluate_baselines(train_df, args.results_dir)

    summary = (
        metrics_df.groupby(["model", "validation_scheme"])["mape"]
        .mean()
        .unstack("validation_scheme")
    )
    print(summary.round(4).to_string())
    print("\nSeasonal naive: share of forecasts from real last-year sales")
    print(coverage.to_string(index=False))
    print(f"\nSaved metrics and predictions to {args.results_dir}/")


if __name__ == "__main__":
    main()
