import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_date_features
from src.metrics import mape
from src.preprocessing import drop_missing_target, load_data, preprocess_data
from src.validation import expanding_window_splits, temporal_train_val_split


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


def seasonal_naive_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> np.ndarray:
    history = train_df[["date", *GROUP_COLS, TARGET_COL]].copy()
    history["date"] = history["date"] + pd.DateOffset(years=1)
    history = history.rename(columns={TARGET_COL: "prediction"})

    pred = val_df[["date", *GROUP_COLS]].merge(
        history,
        on=["date", *GROUP_COLS],
        how="left",
    )

    missing_mask = pred["prediction"].isna()
    if missing_mask.any():
        fallback = group_median_predict(train_df, val_df.loc[missing_mask])
        pred.loc[missing_mask, "prediction"] = fallback

    return pred["prediction"].to_numpy()


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


def _period_text(df: pd.DataFrame) -> str:
    return f"{df['date'].min().date()} to {df['date'].max().date()}"


def _evaluate_split(
    model_name: str,
    version: str,
    params: dict,
    predict_func,
    train_part: pd.DataFrame,
    val_part: pd.DataFrame,
    validation_scheme: str,
    fold: str,
) -> dict:
    prediction = predict_func(train_part, val_part)
    score = mape(val_part[TARGET_COL], prediction)

    return {
        "model": model_name,
        "version": version,
        "validation_scheme": validation_scheme,
        "fold": fold,
        "train_period": _period_text(train_part),
        "validation_period": _period_text(val_part),
        "mape": round(score, 6),
        "params": json.dumps(params, sort_keys=True),
    }


def evaluate_baselines(
    train_df: pd.DataFrame,
    include_long_horizon: bool = True,
) -> pd.DataFrame:
    records = []

    for baseline in BASELINES:
        for fold_number, (train_part, val_part) in enumerate(
            expanding_window_splits(train_df),
            start=1,
        ):
            validation_year = val_part["date"].dt.year.iloc[0]
            records.append(
                _evaluate_split(
                    model_name=baseline["model"],
                    version=baseline["version"],
                    params=baseline["params"],
                    predict_func=baseline["predict"],
                    train_part=train_part,
                    val_part=val_part,
                    validation_scheme="expanding_window",
                    fold=f"fold_{fold_number}_{validation_year}",
                )
            )

        if include_long_horizon:
            train_part, val_part = temporal_train_val_split(
                train_df,
                val_start_date="2014-01-01",
            )
            records.append(
                _evaluate_split(
                    model_name=baseline["model"],
                    version=baseline["version"],
                    params=baseline["params"],
                    predict_func=baseline["predict"],
                    train_part=train_part,
                    val_part=val_part,
                    validation_scheme="long_horizon",
                    fold="2014_2016",
                )
            )

    return pd.DataFrame(records)


def save_metrics(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 1:
        old_metrics = pd.read_csv(output_path)
        baseline_names = set(metrics_df["model"])
        old_metrics = old_metrics[~old_metrics["model"].isin(baseline_names)]
        metrics_df = pd.concat([old_metrics, metrics_df], ignore_index=True)

    metrics_df.to_csv(output_path, index=False)


def load_training_data(train_path: str | Path) -> pd.DataFrame:
    train, _ = load_data(train_path, train_path)
    train = preprocess_data(train)
    train = drop_missing_target(train)
    return train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--output-path", default="results/metrics.csv")
    parser.add_argument("--no-long-horizon", action="store_true")
    args = parser.parse_args()

    train_df = load_training_data(args.train_path)
    metrics_df = evaluate_baselines(
        train_df,
        include_long_horizon=not args.no_long_horizon,
    )
    save_metrics(metrics_df, args.output_path)

    summary = (
        metrics_df.groupby(["model", "validation_scheme"])["mape"]
        .mean()
        .reset_index()
        .sort_values(["validation_scheme", "mape"])
    )
    print(summary.to_string(index=False))
    print(f"\nSaved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
