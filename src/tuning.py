"""Hyperparameter search for CatBoost on the validation folds ONLY.

`tune_catboost` accepts development data only (dates before the test year)
and raises otherwise, so the independent test period can not influence the
choice of hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import mape
from src.model import TARGET_COL, fit_model, predict_model
from src.validation import (
    assert_no_test_data,
    expanding_window_splits,
    fold_name,
)

TUNING_GRID = [
    {"name": "reference", "iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"name": "shallower", "iterations": 500, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"name": "deeper", "iterations": 500, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3},
    {"name": "slow_lr_800", "iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"name": "slow_lr_1000", "iterations": 1000, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3},
    {"name": "fast_lr", "iterations": 500, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3},
    {"name": "more_regularization", "iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"name": "medium_regularization", "iterations": 800, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"name": "shallow_slow", "iterations": 1000, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
    {"name": "deep_slow", "iterations": 800, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5},
]

# Tiny grid for smoke tests (`--quick`); never used for reported results.
QUICK_GRID = [
    {"name": "quick_a", "iterations": 30, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3},
    {"name": "quick_b", "iterations": 30, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3},
]

PARAM_KEYS = ("iterations", "depth", "learning_rate", "l2_leaf_reg")


def candidate_params(candidate: dict) -> dict:
    return {key: candidate[key] for key in PARAM_KEYS}


def tune_catboost(
    dev_df: pd.DataFrame,
    target_transform: str,
    grid: list[dict] | None = None,
    verbose: bool = True,
):
    """Evaluate every candidate on the expanding-window validation folds.

    Parameters
    ----------
    dev_df : development data with date features (NO test-year rows).
    target_transform : "none" or "log1p".

    Returns
    -------
    results : DataFrame, one row per candidate, with per-fold and mean MAPE,
        sorted by mean MAPE (best first).
    predictions : dict {(candidate_name, fold_label): (val_df, y_pred)} so the
        chosen configuration needs no refit.
    """
    assert_no_test_data(dev_df)
    grid = TUNING_GRID if grid is None else grid

    folds = list(expanding_window_splits(dev_df))
    rows = []
    predictions = {}

    for candidate in grid:
        params = candidate_params(candidate)
        fold_scores = {}
        for number, (train_part, val_part) in enumerate(folds, start=1):
            label = fold_name(number, val_part)
            model = fit_model(train_part, params=params, target_transform=target_transform)
            y_pred = predict_model(model, val_part)
            fold_scores[label] = mape(val_part[TARGET_COL].to_numpy(), y_pred)
            predictions[(candidate["name"], label)] = (val_part, y_pred)

        row = {
            "target_transform": target_transform,
            "name": candidate["name"],
            **params,
            **{f"mape_{label}": score for label, score in fold_scores.items()},
            "mean_validation_mape": float(np.mean(list(fold_scores.values()))),
        }
        rows.append(row)
        if verbose:
            print(
                f"[{target_transform}] {candidate['name']:<22} "
                f"mean validation MAPE = {row['mean_validation_mape']:.4f}"
            )

    results = (
        pd.DataFrame(rows)
        .sort_values("mean_validation_mape")
        .reset_index(drop=True)
    )
    return results, predictions


def best_params(results: pd.DataFrame) -> tuple[str, dict]:
    """(candidate name, params) with the lowest mean validation MAPE."""
    best = results.iloc[0]
    params = {
        "iterations": int(best["iterations"]),
        "depth": int(best["depth"]),
        "learning_rate": float(best["learning_rate"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
    }
    return str(best["name"]), params
