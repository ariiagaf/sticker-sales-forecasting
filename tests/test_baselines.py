import numpy as np
import pytest

from src.baselines import (
    group_median_predict, seasonal_naive_coverage, seasonal_naive_predict,
)
from src.preprocessing import drop_missing_target, preprocess_data
from src.validation import expanding_window_splits, split_development_test
from tests.synthetic import make_synthetic_train


@pytest.fixture(scope="module")
def df():
    return drop_missing_target(preprocess_data(make_synthetic_train()))


def test_seasonal_naive_all_folds_keep_length_and_use_last_year(df):
    dev, test = split_development_test(df)
    splits = list(expanding_window_splits(dev)) + [(dev, test)]
    for train, val in splits:
        pred = seasonal_naive_predict(train, val)
        assert len(pred) == len(val)           # 29 Feb must not duplicate rows
        assert np.isfinite(pred).all()
        cov = seasonal_naive_coverage(train, val)
        assert cov["share_from_last_year"] > 0.9


def test_seasonal_naive_value_is_previous_year_actual(df):
    dev, test = split_development_test(df)
    pred = seasonal_naive_predict(dev, test)
    row = test.iloc[100]
    prev = dev[(dev["date"] == row["date"] - __import__("pandas").DateOffset(years=1))
               & (dev["country"] == row["country"]) & (dev["store"] == row["store"])
               & (dev["product"] == row["product"])]
    assert pred[100] == prev["num_sold"].iloc[0]


def test_seasonal_naive_falls_back_beyond_one_year(df):
    """A multi-year window loses real last-year data after year one."""
    train = df[df["date"] < "2014-01-01"]
    val = df[(df["date"] >= "2014-01-01") & (df["date"] < "2017-01-01")]
    cov = seasonal_naive_coverage(train, val)
    assert cov["share_from_last_year"] < 0.4
    assert len(seasonal_naive_predict(train, val)) == len(val)
