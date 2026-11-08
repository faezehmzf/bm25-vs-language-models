#!/usr/bin/env bash
# Rebuild the indexes and every file in results/. Needs the collection in data/ (see data/README.md).
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY scripts/build_index.py --all
$PY scripts/run_models.py
$PY scripts/run_feedback.py
$PY scripts/run_preprocessing.py
$PY -m pytest -q
