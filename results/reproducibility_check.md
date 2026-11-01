# Reproducibility check

- Date: 2026-11-01
- Environment: clean GitHub Actions Ubuntu runner
- OS: Linux 7.0.0-1017-azure (x86_64)
- Python: 3.14.9
- pandas: 3.1.0
- numpy: 2.5.4
- scikit-learn: 1.9.1
- catboost: 1.2.10
- matplotlib: 3.11.2
- jupyter: 1.1.1
- ipykernel: 7.4.0
- nbclient: 0.11.0
- pytest: 9.2.0

The clean dependency set above was first resolved and verified from a fresh environment, then these exact versions were pinned in `requirements.txt`. The pinned file was installed again successfully for the final Role 5 checks.

## Full-pipeline reproducibility

1. The test suite completed successfully: **20 tests passed, 4 subtests passed**.
2. The full project pipeline was executed twice independently from scratch with:
   `python -m src.repro_check --train-path data/train.csv --final-results-dir results`.
3. Both full runs completed without errors.
4. Both runs produced the same evaluation rows.
5. Maximum absolute MAPE difference between the two runs: **0**.
6. Maximum absolute prediction difference between the two runs: **0**.
7. In a separate clean full-pipeline run, all **28** regenerated metric rows were compared with the committed `results/metrics.csv`; maximum absolute MAPE difference: **0**.

## Error-analysis reproducibility

1. `notebooks/04_evaluation.ipynb` was executed from top to bottom successfully with the pinned environment.
2. Regenerated country, country-year, product, store, year, Norway-level and worst-series tables match the committed error-analysis values.
3. The evaluation notebook was executed twice independently in the pinned environment; the regenerated error-analysis tables and `error_analysis_worst_series.png` were identical between the two runs.

## Result

**SUCCESS — the reported metrics, saved predictions, error-analysis tables and error-analysis plot are reproducible in the verified clean environment.**
