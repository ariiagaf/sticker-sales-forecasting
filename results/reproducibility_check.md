# Reproducibility check

- Date: 2026-11-01 14:55 UTC
- Environment: GitHub Actions clean Ubuntu runner
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

## Verification

1. The project dependencies were installed in a clean environment from the pinned `requirements.txt`.
2. The test suite completed successfully: **20 tests passed, 4 subtests passed**.
3. The full project pipeline was executed twice independently from scratch with:
   `python -m src.repro_check --train-path data/train.csv --final-results-dir results`.
4. Both full runs completed without errors.
5. Both runs produced the same evaluation rows.
6. Maximum absolute MAPE difference between the two runs: **0**.
7. Maximum absolute prediction difference between the two runs: **0**.
8. In a separate clean full-pipeline run, all **28** regenerated metric rows were compared with the committed `results/metrics.csv`; maximum absolute MAPE difference: **0**.
9. `notebooks/04_evaluation.ipynb` was executed from top to bottom successfully in the same pinned environment.

## Result

**SUCCESS — the reported metrics and predictions are reproducible in the verified clean environment.**
