import pandas as pd
import pytest

from src.preprocessing import drop_missing_target, preprocess_data
from src.validation import (
    TEST_YEAR, VALIDATION_YEARS, assert_no_test_data,
    expanding_window_splits, split_development_test,
)
from tests.synthetic import make_synthetic_train


@pytest.fixture(scope="module")
def df():
    return drop_missing_target(preprocess_data(make_synthetic_train()))


def test_validation_years_do_not_touch_test(df):
    assert max(VALIDATION_YEARS) < TEST_YEAR


def test_folds_are_temporal_and_never_see_test(df):
    n = 0
    for train, val in expanding_window_splits(df):
        n += 1
        assert train["date"].max() < val["date"].min()
        assert train["date"].max() < pd.Timestamp(f"{TEST_YEAR}-01-01")
        assert val["date"].max() < pd.Timestamp(f"{TEST_YEAR}-01-01")
    assert n == len(VALIDATION_YEARS)


def test_overlapping_validation_year_rejected(df):
    with pytest.raises(ValueError):
        list(expanding_window_splits(df, validation_years=(2014, 2016)))


def test_dev_test_split(df):
    dev, test = split_development_test(df)
    assert dev["date"].max() < test["date"].min()
    assert test["date"].dt.year.unique().tolist() == [TEST_YEAR]
    assert_no_test_data(dev)
    with pytest.raises(ValueError):
        assert_no_test_data(df)
