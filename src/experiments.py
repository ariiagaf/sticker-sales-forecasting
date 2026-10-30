"""CatBoost experiments: raw vs log1p target, default vs tuned.

Protocol (the same for every model, see src/validation.py):

1. All decisions (hyperparameters, raw-vs-log, final model) use ONLY the
   validation folds inside the development data (< 2016).
2. Each model is then fitted once on all development data and scored once on
   the independent test year (2016). Test numbers are never fed back.
"""

from __future__ import annotations

from pathlib import Path

import catboost
import pandas as pd

from src.features import add_date_features
from src.metrics import mape
from src.model import (
    DEFAULT_PARAMS,
    FEATURE_COLS,
    MODEL_REGISTRY,
    TARGET_COL,
    fit_model,
    predict_model,
)
from src.preprocessing import drop_missing_target, preprocess_data
from src.results import (
    SCHEME_TEST,
    SCHEME_VALIDATION,
    make_record,
    save_predictions,
    upsert_metrics,
)
from src.tuning import QUICK_GRID, TUNING_GRID, best_params, tune_catboost
from src.validation import (
    expanding_window_splits,
    fold_name,
    split_development_test,
    test_name,
)

QUICK_DEFAULT_PARAMS = {"iterations": 30, "depth": 4, "learning_rate": 0.1}


def prepare_data(train_path: str | Path) -> pd.DataFrame:
    """Load train.csv, sort, drop rows with unknown target, add date features."""
    train = pd.read_csv(train_path)
    train = preprocess_data(train)
    train = drop_missing_target(train)
    return add_date_features(train)


def _record_params(params: dict, target_transform: str) -> dict:
    """Record every explicitly configured training option, including defaults."""
    return {
        **DEFAULT_PARAMS,
        **params,
        "target_transform": target_transform,
        "features": FEATURE_COLS,
        "catboost_version": catboost.__version__,
    }


def evaluate_default(
    dev_df: pd.DataFrame,
    model_name: str,
    params: dict | None = None,
    results_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit a CatBoost config on each validation fold and record MAPE."""
    info = MODEL_REGISTRY[model_name]
    transform = info["target_transform"]
    params = params or {}
    records = []

    for number, (train_part, val_part) in enumerate(expanding_window_splits(dev_df), start=1):
        label = fold_name(number, val_part)
        model = fit_model(train_part, params=params, target_transform=transform)
        # Keep the target scale explicit at prediction time too.  This makes
        # the raw/log experiment definition self-contained rather than
        # relying only on the fitted model attribute.
        y_pred = predict_model(model, val_part, target_transform=transform)
        score = mape(val_part[TARGET_COL].to_numpy(), y_pred)
        records.append(
            make_record(
                model_name, info["version"], SCHEME_VALIDATION, label,
                train_part, val_part, score, _record_params(params, transform),
            )
        )
        if results_dir is not None:
            save_predictions(results_dir, model_name, label, val_part, y_pred)
        if verbose:
            print(f"{model_name} {label}: MAPE = {score:.4f}%")
    return pd.DataFrame(records)


def evaluate_on_test(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    params: dict | None = None,
    results_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Final evaluation: fit on ALL development data, score once on test."""
    info = MODEL_REGISTRY[model_name]
    transform = info["target_transform"]
    params = params or {}

    model = fit_model(dev_df, params=params, target_transform=transform)
    # Use the same explicit transform as at fit time for the independent test.
    y_pred = predict_model(model, test_df, target_transform=transform)
    score = mape(test_df[TARGET_COL].to_numpy(), y_pred)

    record = make_record(
        model_name, info["version"], SCHEME_TEST, test_name(),
        dev_df, test_df, score, _record_params(params, transform),
    )
    if results_dir is not None:
        save_predictions(results_dir, model_name, test_name(), test_df, y_pred)
    if verbose:
        print(f"{model_name} {test_name()}: MAPE = {score:.4f}%")
    return record


def run_catboost_experiments(
    df: pd.DataFrame,
    results_dir: str | Path = "results",
    quick: bool = False,
    verbose: bool = True,
) -> dict:
    """Run raw/log x default/tuned. Returns a dict with metrics and tuning tables.

    `df` must contain date features (see `prepare_data`).
    quick=True uses a tiny grid and few iterations: for smoke tests only.
    """
    results_dir = Path(results_dir)
    dev_df, test_df = split_development_test(df)
    grid = QUICK_GRID if quick else TUNING_GRID
    default_params = QUICK_DEFAULT_PARAMS if quick else {}

    all_records = []
    tuning_tables = []
    chosen = {}

    for transform, default_name, tuned_name in (
        ("none", "catboost_raw", "catboost_raw_tuned"),
        ("log1p", "catboost_log", "catboost_log_tuned"),
    ):
        # default configuration: validation folds
        all_records.append(
            evaluate_default(dev_df, default_name, default_params, results_dir, verbose)
        )

        # tuning: validation folds only
        tuning, preds = tune_catboost(dev_df, transform, grid, verbose)
        tuning_tables.append(tuning)
        best_name, best = best_params(tuning)
        chosen[tuned_name] = {"candidate": best_name, "params": best}

        # record validation results of the chosen configuration (no refit)
        info = MODEL_REGISTRY[tuned_name]
        tuned_records = []
        for number, (train_part, val_part) in enumerate(
            expanding_window_splits(dev_df), start=1
        ):
            label = fold_name(number, val_part)
            _, y_pred = preds[(best_name, label)]
            score = mape(val_part[TARGET_COL].to_numpy(), y_pred)
            tuned_records.append(
                make_record(
                    tuned_name, info["version"], SCHEME_VALIDATION, label,
                    train_part, val_part, score, _record_params(best, transform),
                )
            )
            save_predictions(results_dir, tuned_name, label, val_part, y_pred)
        all_records.append(pd.DataFrame(tuned_records))

    # independent test: every final model is fitted once and scored once
    test_records = [
        evaluate_on_test(dev_df, test_df, "catboost_raw", default_params, results_dir, verbose),
        evaluate_on_test(dev_df, test_df, "catboost_log", default_params, results_dir, verbose),
        evaluate_on_test(
            dev_df, test_df, "catboost_raw_tuned",
            chosen["catboost_raw_tuned"]["params"], results_dir, verbose,
        ),
        evaluate_on_test(
            dev_df, test_df, "catboost_log_tuned",
            chosen["catboost_log_tuned"]["params"], results_dir, verbose,
        ),
    ]

    metrics = pd.concat([*all_records, pd.DataFrame(test_records)], ignore_index=True)
    upsert_metrics(metrics, results_dir / "metrics.csv")

    tuning_all = pd.concat(tuning_tables, ignore_index=True)
    tuning_all.to_csv(results_dir / "tuning_results.csv", index=False)

    return {"metrics": metrics, "tuning": tuning_all, "chosen": chosen}


def select_final_model(metrics: pd.DataFrame) -> str:
    """CatBoost model with the lowest mean VALIDATION MAPE (test not used)."""
    validation = metrics[
        (metrics["validation_scheme"] == SCHEME_VALIDATION)
        & metrics["model"].isin(MODEL_REGISTRY)
    ]
    means = validation.groupby("model")["mape"].mean()
    return str(means.idxmin())


def tuning_effect(metrics: pd.DataFrame) -> pd.DataFrame:
    """Does tuning help? default vs tuned, on validation and on test."""
    rows = []
    for default_name, tuned_name in (
        ("catboost_raw", "catboost_raw_tuned"),
        ("catboost_log", "catboost_log_tuned"),
    ):
        row = {"target_transform": MODEL_REGISTRY[default_name]["target_transform"]}
        for scheme, label in ((SCHEME_VALIDATION, "validation"), (SCHEME_TEST, "test")):
            part = metrics[metrics["validation_scheme"] == scheme]
            base = part.loc[part["model"] == default_name, "mape"].mean()
            tuned = part.loc[part["model"] == tuned_name, "mape"].mean()
            row[f"{label}_default"] = base
            row[f"{label}_tuned"] = tuned
            row[f"{label}_improvement_pp"] = base - tuned
        rows.append(row)
    return pd.DataFrame(rows)
