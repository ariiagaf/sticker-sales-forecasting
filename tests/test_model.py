import numpy as np
import pytest

from src.features import add_date_features
from src.model import (
    TARGET_COL, fit_model, inverse_transform_target, predict_model, transform_target,
)
from src.preprocessing import drop_missing_target, preprocess_data
from tests.synthetic import make_synthetic_train

FAST = {"iterations": 25, "depth": 4, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def data():
    df = make_synthetic_train(start="2013-01-01", end="2015-12-31")
    df = add_date_features(drop_missing_target(preprocess_data(df)))
    return df[df["date"] < "2015-07-01"], df[df["date"] >= "2015-07-01"]


def test_transform_roundtrip():
    y = np.array([0.0, 1.0, 17.0, 1234.5])
    for t in ("none", "log1p"):
        assert np.allclose(inverse_transform_target(transform_target(y, t), t), y)


@pytest.mark.parametrize("transform", ["none", "log1p"])
def test_both_modes_same_interface_original_scale(data, transform):
    train, val = data
    model = fit_model(train, FAST, target_transform=transform)
    pred = predict_model(model, val)
    assert pred.shape == (len(val),)
    assert (pred >= 0).all() and np.isfinite(pred).all()
    # predictions are in ORIGINAL units: same order of magnitude as the target
    ratio = pred.mean() / val[TARGET_COL].mean()
    assert 0.3 < ratio < 3


def test_raw_and_log_are_different_models(data):
    train, val = data
    p_raw = predict_model(fit_model(train, FAST, "none"), val)
    p_log = predict_model(fit_model(train, FAST, "log1p"), val)
    assert not np.allclose(p_raw, p_log)


def test_unknown_transform_rejected(data):
    with pytest.raises(ValueError):
        fit_model(data[0], FAST, "sqrt")
