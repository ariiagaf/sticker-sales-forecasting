# Sticker Sales Forecasting

ML project for the Kaggle "Forecasting Sticker Sales" competition
(Playground Series S5E1). The task is to predict daily `num_sold` for
90 (country, store, product) time series covering 2010–2016 (train) and
2017–2019 (test).

## Setup

Use Python 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle competition data and place the files in `data/`:

```
data/train.csv
data/test.csv
data/sample_submission.csv
```

`data/*.csv` is git-ignored, so this step is required after every fresh clone.
Get the files from:
https://www.kaggle.com/competitions/playground-series-s5e1/data

## Project structure

```
.
├── data/                       # train.csv / test.csv (git-ignored, see Setup)
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_baselines.ipynb      # baseline models + metrics
│   ├── 03_model.ipynb          # CatBoost v1 (raw) -> v2 (log1p) -> v3 (tuned)
│   └── 04_evaluation.ipynb     # final evaluation + error analysis (Stage 6)
├── src/
│   ├── preprocessing.py        # load_data, preprocess_data, drop_missing_target
│   ├── features.py             # add_date_features()
│   ├── validation.py           # expanding_window_splits, temporal_train_val_split
│   ├── metrics.py              # mape()
│   ├── baselines.py            # group_median / seasonal_naive_last_year / ridge_calendar_one_hot
│   └── model.py                # CatBoost: create_model, fit_model, predict_model
├── results/
│   ├── metrics.csv             # all baseline + model results
│   └── error_analysis_*.csv/.png   # produced by 04_evaluation.ipynb
├── project.pdf                 # technical report
├── requirements.txt
└── README.md
```

## Data Pipeline

The project uses a shared preprocessing, feature engineering, validation,
and evaluation pipeline implemented in `src/`.

### Preprocessing

```python
from src.preprocessing import load_data, preprocess_data, drop_missing_target

train, test = load_data("data/train.csv", "data/test.csv")

train = preprocess_data(train)
test = preprocess_data(test)

# Rows with unknown target cannot be used for supervised training.
train = drop_missing_target(train)
```

- converts `date` to pandas datetime and sorts observations chronologically;
- `drop_missing_target()` removes the ~3.85% of train rows with missing
  `num_sold` (mostly Holographic Goose for Canada and Kenya) — this is
  **not** applied to `test`, and missing targets are never imputed.

### Feature engineering

```python
from src.features import add_date_features

train = add_date_features(train)
test = add_date_features(test)
```

Adds: `year, month, day, day_of_week, day_of_year, week_of_year, quarter,
is_weekend, time_idx` and weekly/yearly cyclical `sin`/`cos` features.

### Validation

Random train/validation splitting is **not** used, since this is a
forecasting problem (it would leak future information into training).

```python
from src.validation import expanding_window_splits, temporal_train_val_split

# expanding-window folds: 2014, 2015, 2016
for train_part, val_part in expanding_window_splits(train):
    ...

# long-horizon backtest: train 2010-2013 -> validate 2014-2016
train_part, val_part = temporal_train_val_split(train, val_start_date="2014-01-01")
```

All models use the same validation procedure and the same `mape()` from
`src/metrics.py`, so results are directly comparable.

## Baselines

Implemented in `src/baselines.py`: `group_median`, `seasonal_naive_last_year`,
`ridge_calendar_one_hot`.

```bash
python -m src.baselines
```

Writes results to `results/metrics.csv`.

## Model

CatBoost model, implemented in `src/model.py` (`create_model`, `fit_model`,
`predict_model`). Trained/evaluated across three versions in
`notebooks/03_model.ipynb`:

- **v1 (`catboost_raw`)** — raw `num_sold` target, default parameters.
- **v2 (`catboost_log`)** — same parameters, `log1p(num_sold)` target
  (the single change that gave the largest improvement).
- **v3 (`catboost_log_tuned`)** — log1p target + a small hyperparameter
  search (depth, learning_rate, iterations, l2_leaf_reg) selected on the
  expanding-window folds only; best config: depth=8, l2_leaf_reg=3.

## Evaluation & error analysis (Stage 6)

`notebooks/04_evaluation.ipynb` trains the final v3 configuration on the
long-horizon split and compares it against `seasonal_naive_last_year`,
breaking MAPE down by country / store / product / year and plotting the
worst-performing series. Outputs go to `results/error_analysis_*`.

## Results summary

| Model | Expanding-window MAPE (%) | Long-horizon MAPE (%) |
|---|---|---|
| group_median (v1) | 16.32 | 17.46 |
| seasonal_naive_last_year (v1, best baseline) | 13.27 | 17.02 |
| ridge_calendar_one_hot (v1) | 17.86 | 23.91 |
| catboost_raw (v1) | 18.24 | 22.52 |
| catboost_log (v2, log1p target) | 9.17 | 14.94 |
| **catboost_log_tuned (v3, log1p + tuned)** | **9.04** | **14.60** |

Exact per-fold numbers are in `results/metrics.csv`. Full discussion,
methodology and error analysis are in `project.pdf`.

## Reproducing the results end-to-end

1. `git clone https://github.com/ariiagaf/sticker-sales-forecasting.git && cd sticker-sales-forecasting`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Place `train.csv` / `test.csv` in `data/`
5. Run `notebooks/01_eda.ipynb`
6. Run `python -m src.baselines` (or `notebooks/02_baselines.ipynb`)
7. Run `notebooks/03_model.ipynb`
8. Run `notebooks/04_evaluation.ipynb`
9. Check that `results/metrics.csv` matches the table above.

Tested with Python 3.14.4, pandas 3.0.5, NumPy 2.5.3, CatBoost 1.2.10
(versions are not pinned in `requirements.txt`).
