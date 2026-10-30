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


def test_group_median_uses_training_data_only():
    import pandas as pd

    train = pd.DataFrame({
        "country": ["Finland", "Finland", "Finland"],
        "store": ["KaggleMart"] * 3,
        "product": ["Sticker"] * 3,
        "num_sold": [10, 20, 30],
    })

    val = pd.DataFrame({
        "country": ["Finland", "Norway"],
        "store": ["KaggleMart"] * 2,
        "product": ["Sticker"] * 2,
        "num_sold": [9999, 9999],
    })

    pred = group_median_predict(train, val)

    assert np.allclose(pred, [20, 20])


def test_seasonal_naive_does_not_use_validation_targets(df):
    dev, test = split_development_test(df)

    original = seasonal_naive_predict(dev, test)

    modified_test = test.copy()
    modified_test["num_sold"] = 999999

    modified = seasonal_naive_predict(dev, modified_test)

    np.testing.assert_array_equal(original, modified)


def test_ridge_predictions_are_valid(df):
    from src.baselines import ridge_calendar_predict

    dev, test = split_development_test(df)

    pred = ridge_calendar_predict(dev, test)

    assert len(pred) == len(test)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_seasonal_naive_leap_year():
    import pandas as pd

    train = pd.DataFrame({
        "date": pd.to_datetime([
            "2012-02-28",
            "2012-02-29",
            "2012-03-01",
        ]),
        "country": ["Finland"] * 3,
        "store": ["KaggleMart"] * 3,
        "product": ["Sticker"] * 3,
        "num_sold": [10, 99, 30],
    })

    val = pd.DataFrame({
        "date": pd.to_datetime([
            "2013-02-28",
            "2013-03-01",
        ]),
        "country": ["Finland"] * 2,
        "store": ["KaggleMart"] * 2,
        "product": ["Sticker"] * 2,
    })

    predictions = seasonal_naive_predict(train, val)

    np.testing.assert_array_equal(predictions, [10, 30])