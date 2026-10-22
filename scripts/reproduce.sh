#!/bin/bash
# Rebuild the planted worlds and the README tables from saved results. No model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for g in worlds/make_planted.py worlds/make_planted2.py worlds/make_planted3.py; do $PY "$g"; done
$PY scripts/tables.py
$PY scripts/figure.py
