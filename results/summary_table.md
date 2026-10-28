| Model | Validation MAPE, % (mean of folds 2013-2015) | Test MAPE, % (2016) |
|---|---|---|
| group_median (v1) | 14.39 | 18.16 |
| seasonal_naive_last_year (v1) | 12.89 | 12.69 |
| ridge_calendar_one_hot (v1) | 16.10 | 18.22 |
| catboost_raw (v1) | 18.52 | 15.06 |
| catboost_log (v2) | 9.17 | 6.57 |
| catboost_raw_tuned (v3) | 14.33 | 10.76 |
| **catboost_log_tuned (v3)** | 9.17 | 6.57 |

Final model (lowest validation MAPE): catboost_log_tuned
