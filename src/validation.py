"""Temporal validation scheme shared by ALL models (baselines and CatBoost).

Three disjoint roles for time periods (train data covers 2010-2016):

    2010 ........ 2015 | 2016
    development data   | independent TEST (never used for any decision)

* Development data (dates < TEST_YEAR) is used for training, for choosing
  hyperparameters and for choosing between models. Inside it, the
  expanding-window folds validate on 2013, 2014 and 2015:

      fold_1_2013: train 2010-2012 -> validate 2013
      fold_2_2014: train 2010-2013 -> validate 2014
      fold_3_2015: train 2010-2014 -> validate 2015

* TEST (2016) is touched only once per final model: the model is fitted on
  all development data (2010-2015) and scored on 2016. Test results must
  never be used to pick hyperparameters or models.

The test period is a full year, so a one-year-ahead forecast is evaluated.
For that horizon seasonal_naive_last_year really uses last year's actual sales
(2015 for the 2016 test), which is what the report says about it.

The scheme is enforced in code: `expanding_window_splits` refuses validation
years that overlap the test year and removes test-year rows from its input, so
tuning code cannot see the test period even by accident.
"""

from __future__ import annotations

import pandas as pd

VALIDATION_YEARS = (2013, 2014, 2015)
TEST_YEAR = 2016


def _with_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Column 'date' was not found.")
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df


def split_development_test(
    df: pd.DataFrame,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into development data (before test_year) and the test year.

    Rows after the test year (if any) are ignored.
    """
    df = _with_datetime(df)
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")

    dev_df = df[df["date"] < test_start].copy()
    test_df = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()

    if dev_df.empty:
        raise ValueError("Development set is empty.")
    if test_df.empty:
        raise ValueError(f"Test set for year {test_year} is empty.")
    if dev_df["date"].max() >= test_df["date"].min():
        raise ValueError("Temporal leakage detected between development and test.")

    return dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def assert_no_test_data(df: pd.DataFrame, test_year: int = TEST_YEAR) -> None:
    """Raise if df contains rows from the test year or later."""
    df = _with_datetime(df)
    if (df["date"] >= pd.Timestamp(f"{test_year}-01-01")).any():
        raise ValueError(
            f"Data from {test_year} or later is the independent test period "
            "and must not be used for training, tuning or model selection."
        )


def expanding_window_splits(
    df: pd.DataFrame,
    validation_years=VALIDATION_YEARS,
    test_year: int = TEST_YEAR,
):
    """Yield (train_df, val_df) expanding-window folds for model development.

    For validation year Y, train = all rows before Y, val = rows of year Y.
    The test year can never appear in train or val.
    """
    validation_years = tuple(validation_years)
    if max(validation_years) >= test_year:
        raise ValueError(
            f"Validation years {validation_years} overlap the test year "
            f"{test_year}. Validation must end before the test period."
        )

    df = _with_datetime(df)
    # Test-year rows are removed up front: folds are built from development
    # data only.
    df = df[df["date"] < pd.Timestamp(f"{test_year}-01-01")]

    for year in validation_years:
        val_start = pd.Timestamp(f"{year}-01-01")
        val_end = pd.Timestamp(f"{year + 1}-01-01")

        train_df = df[df["date"] < val_start].copy()
        val_df = df[(df["date"] >= val_start) & (df["date"] < val_end)].copy()

        if train_df.empty:
            raise ValueError(f"Training set is empty for validation year {year}.")
        if val_df.empty:
            raise ValueError(f"Validation set is empty for validation year {year}.")
        if train_df["date"].max() >= val_df["date"].min():
            raise ValueError(f"Temporal leakage detected for validation year {year}.")

        yield train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def fold_name(index: int, val_df: pd.DataFrame) -> str:
    """Canonical fold label, e.g. 'fold_1_2013'."""
    return f"fold_{index}_{int(val_df['date'].dt.year.iloc[0])}"


def test_name(test_year: int = TEST_YEAR) -> str:
    return f"test_{test_year}"


def describe_scheme(
    validation_years=VALIDATION_YEARS,
    test_year: int = TEST_YEAR,
) -> str:
    """Human-readable description of the scheme (used in README / notebooks)."""
    lines = []
    for i, year in enumerate(validation_years, start=1):
        lines.append(f"fold_{i}_{year}: train 2010-{year - 1} -> validate {year}")
    lines.append(
        f"test_{test_year}: train 2010-{test_year - 1} -> test {test_year} "
        "(independent, final evaluation only)"
    )
    return "\n".join(lines)
