# Sticker Sales Forecasting

Forecasting Sticker Sales Kaggle project.

## Data Pipeline

The project uses a shared preprocessing, feature engineering, validation, and evaluation pipeline implemented in `src/`.

### Preprocessing

Load and preprocess the data:

```python
from src.preprocessing import (
    load_data,
    preprocess_data,
    drop_missing_target,
)

train, test = load_data(
    "data/train.csv",
    "data/test.csv",
)

train = preprocess_data(train)
test = preprocess_data(test)

# Rows with unknown target cannot be used for supervised training.
train = drop_missing_target(train)
```

The preprocessing pipeline:

- converts `date` to pandas datetime;
- sorts observations chronologically;
- preserves the original test data;
- removes rows with missing `num_sold` only from supervised training data.

Missing target values are not artificially imputed.

### Feature Engineering

Shared calendar features can be generated with:

```python
from src.features import add_date_features

train = add_date_features(train)
test = add_date_features(test)
```

The following features are added:

- year
- month
- day
- day of week
- day of year
- ISO week of year
- quarter
- weekend indicator

### Validation

Random train/validation splitting must not be used for model comparison because this is a forecasting problem.

A single temporal holdout can be created with:

```python
from src.validation import temporal_train_val_split

train_part, val_part = temporal_train_val_split(
    train,
    val_start_date="2016-01-01",
)
```

For the main model comparison, expanding-window validation is available:

```python
from src.validation import expanding_window_splits

for train_part, val_part in expanding_window_splits(train):
    # fit model on train_part
    # predict val_part
    pass
```

The default folds are:

| Fold | Training period | Validation period |
|---|---|---|
| 1 | 2010–2013 | 2014 |
| 2 | 2010–2014 | 2015 |
| 3 | 2010–2015 | 2016 |

All models should use the same validation procedure for a fair comparison.

### Long-Horizon Validation

In addition to the yearly expanding-window folds, a long-horizon
backtest is used to better approximate the final forecasting setting:

| Training period | Validation period | Forecast horizon |
|---|---|---|
| 2010–2013 | 2014–2016 | 3 years |

It can be created with:

```python
train_part, val_part = temporal_train_val_split(
    train,
    val_start_date="2014-01-01",
)
```

The yearly expanding-window folds are used to evaluate model stability
across different time periods, while the three-year holdout provides
an additional long-horizon evaluation.

### Metric

The shared evaluation metric is MAPE:

```python
from src.metrics import mape

score = mape(y_true, y_pred)
print(score)
```

The function returns MAPE in percent.

All baseline and proposed models should use the implementation from `src/metrics.py` so that reported results are directly comparable.
