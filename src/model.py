import numpy as np
from catboost import CatBoostRegressor

MODEL_NAME = "catboost_log"
MODEL_VERSION = "v2"
TARGET_COL = "num_sold"
TARGET_TRANSFORM = "log1p"
SUPPORTED_TARGET_TRANSFORMS = ("none", "log1p")

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


def create_model(params: dict | None = None) -> CatBoostRegressor:
    model_params = DEFAULT_PARAMS.copy()
    if params is not None:
        model_params.update(params)

    return CatBoostRegressor(
        cat_features=CATEGORICAL_FEATURES,
        **model_params,
    )


def _validate_target_transform(target_transform: str) -> None:
    if target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        supported = ", ".join(SUPPORTED_TARGET_TRANSFORMS)
        raise ValueError(
            f"Unsupported target_transform: {target_transform!r}. "
            f"Expected one of: {supported}."
        )


def _transform_target(target, target_transform: str) -> np.ndarray:
    _validate_target_transform(target_transform)

    transformed = np.asarray(target, dtype=float)
    if target_transform == "log1p":
        if np.any(transformed <= -1):
            raise ValueError(
                "log1p target transform requires target values greater than -1."
            )
        transformed = np.log1p(transformed)

    if not np.isfinite(transformed).all():
        raise ValueError("Transformed target contains NaN or infinite values.")

    return transformed


def _inverse_transform_predictions(
    predictions,
    target_transform: str,
) -> np.ndarray:
    _validate_target_transform(target_transform)

    transformed = np.asarray(predictions, dtype=float)
    if target_transform == "log1p":
        transformed = np.expm1(transformed)

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Predictions contain NaN or infinite values after inverse transform."
        )

    return transformed


def fit_model(
    train_features,
    params: dict | None = None,
    *,
    target_transform: str = TARGET_TRANSFORM,
) -> CatBoostRegressor:
    _validate_target_transform(target_transform)

    required_columns = FEATURE_COLS + [TARGET_COL]
    missing_columns = [
        column
        for column in required_columns
        if column not in train_features.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if train_features.empty:
        raise ValueError("Training data is empty.")

    if train_features[required_columns].isna().any().any():
        raise ValueError("Training features or target contain missing values.")

    target = _transform_target(
        train_features[TARGET_COL],
        target_transform,
    )

    model = create_model(params)
    model.fit(
        train_features[FEATURE_COLS],
        target,
    )
    return model


def predict_model(
    model: CatBoostRegressor,
    features,
    *,
    target_transform: str = TARGET_TRANSFORM,
) -> np.ndarray:
    _validate_target_transform(target_transform)

    missing_columns = [
        column
        for column in FEATURE_COLS
        if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if features.empty:
        raise ValueError("Prediction data is empty.")

    if features[FEATURE_COLS].isna().any().any():
        raise ValueError("Prediction features contain missing values.")

    model_predictions = np.asarray(
        model.predict(features[FEATURE_COLS]),
        dtype=float,
    )

    if model_predictions.shape != (len(features),):
        raise ValueError("Expected one prediction per input row.")

    if not np.isfinite(model_predictions).all():
        raise ValueError("Model predictions contain NaN or infinite values.")

    predictions = _inverse_transform_predictions(
        model_predictions,
        target_transform,
    )

    negative_count = int((predictions < 0).sum())

    if negative_count:
        print(
            f"Clipping {negative_count} negative predictions to zero "
            f"(raw minimum: {predictions.min():.6f})."
        )

    return np.maximum(predictions, 0.0)
