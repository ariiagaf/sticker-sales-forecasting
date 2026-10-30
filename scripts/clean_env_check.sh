#!/usr/bin/env bash
# Reproducibility check in a CLEAN virtual environment.
# Usage: bash scripts/clean_env_check.sh [path/to/train.csv]
set -euo pipefail

TRAIN="${1:-data/train.csv}"
cd "$(dirname "$0")/.."

rm -rf .venv_check
python3 -m venv .venv_check
source .venv_check/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m pytest -q tests
python -m src.repro_check --train-path "$TRAIN" --final-results-dir results
python -m src.report_tables --results-dir results --readme README.md
echo "Report: results/reproducibility_check.md"
