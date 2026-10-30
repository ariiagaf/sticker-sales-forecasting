"""CatBoost model with two interchangeable target modes.

    catboost_raw : CatBoost trained on the original `num_sold`
    catboost_log : CatBoost trained on log1p(num_sold), predictions mapped back
                   to the original scale with expm1

Both modes share one interface:

    model = fit_model(train_df, params=None, target_transform="none" | "log1p")
    pred  = predict_model(model, features_df)   # always in original units

The fitted model remembers its own target transform, so `predict_model` always
returns predictions on the original `num_sold` scale (clipped at 0).
"""

from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor

TARGET_COL = "num_sold"
TARGET_TRANSFORMS = ("none", "log1p")

# model name -> (version, target transform). Names/versions are the ones used
# in results/metrics.csv.
MODEL_REGISTRY = {
    "catboost_raw": {"version": "v1", "target_transform": "none"},
    "catboost_log": {"version": "v2", "target_transform": "log1p"},
    "catboost_raw_tuned": {"version": "v3", "target_transform": "none"},
    "catboost_log_tuned": {"version": "v3", "target_transform": "log1p"},
}

# Kept for backwards compatibility with older notebooks.
MODEL_NAME = "catboost_log"
MODEL_VERSION = "v2"
TARGET_TRANSFORM = "log1p"

CATEGORICAL_FEATURES = [
    "country",
    "store",
    "product",
]

NUMERIC_FEATURES = [
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

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": 42,
    "task_type": "CPU",
    "thread_count": 4,
    "has_time": True,
    "use_best_model": False,
    "allow_writing_files": False,
    "verbose": False,
}


def _check_transform(target_transform: str) -> None:
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(
            f"Unknown target_transform {target_transform!r}; "
            f"expected one of {TARGET_TRANSFORMS}."
        )


def transform_target(y, target_transform: str) -> np.ndarray:
    """Original scale -> model scale."""
    _check_transform(target_transform)
    y = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.log1p(y)
    return y


def inverse_transform_target(y_model_scale, target_transform: str) -> np.ndarray:
    """Model scale -> original scale (no clipping)."""
    _check_transform(target_transform)
    y = np.asarray(y_model_scale, dtype=float)
    if target_transform == "log1p":
        return np.expm1(y)
    return y


def create_model(params: dict | None = None) -> CatBoostRegressor:
    model_params = DEFAULT_PARAMS.copy()
    if params is not None:
        model_params.update(params)

    return CatBoostRegressor(
        cat_features=CATEGORICAL_FEATURES,
        **model_params,
    )


def fit_model(
    train_features,
    params: dict | None = None,
    target_transform: str = "log1p",
) -> CatBoostRegressor:
    """Fit CatBoost on `target_transform(num_sold)`.

    target_transform: "none" (raw target) or "log1p".
    """
    _check_transform(target_transform)

    required_columns = FEATURE_COLS + [TARGET_COL]
    missing_columns = [
        column for column in required_columns if column not in train_features.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if train_features.empty:
        raise ValueError("Training data is empty.")

    if train_features[required_columns].isna().any().any():
        raise ValueError("Training features or target contain missing values.")

    if target_transform == "log1p" and (train_features[TARGET_COL] < 0).any():
        raise ValueError("log1p target requires non-negative num_sold.")

    target = transform_target(train_features[TARGET_COL], target_transform)

    model = create_model(params)
    model.fit(train_features[FEATURE_COLS], target)
    # remembered so that predict_model maps predictions back correctly
    model.target_transform_ = target_transform
    return model


def predict_model(
    model: CatBoostRegressor,
    features,
    target_transform: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Predict in ORIGINAL units of num_sold (negative values clipped to 0)."""
    if target_transform is None:
        target_transform = getattr(model, "target_transform_", None)
        if target_transform is None:
            raise ValueError(
                "Model has no target_transform_; pass target_transform explicitly."
            )
    _check_transform(target_transform)

    missing_columns = [
        column for column in FEATURE_COLS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if features.empty:
        raise ValueError("Prediction data is empty.")

    if features[FEATURE_COLS].isna().any().any():
        raise ValueError("Prediction features contain missing values.")

    raw_output = np.asarray(model.predict(features[FEATURE_COLS]), dtype=float)

    if raw_output.shape != (len(features),):
        raise ValueError("Expected one prediction per input row.")
    if not np.isfinite(raw_output).all():
        raise ValueError("Model output contains NaN or infinite values.")

    predictions = inverse_transform_target(raw_output, target_transform)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain NaN or infinite values after inverse transform."
        )

    negative_count = int((predictions < 0).sum())
    if negative_count and verbose:
        print(
            f"Clipping {negative_count} negative predictions to zero "
            f"(raw minimum: {predictions.min():.6f})."
        )

    return np.maximum(predictions, 0.0)
