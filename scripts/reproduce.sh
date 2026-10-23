#!/bin/bash
# Rebuild the planted worlds and the README tables from saved results. No model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for g in worlds/make_planted.py worlds/make_planted2.py worlds/make_planted3.py worlds/make_planted4.py worlds/make_planted5.py \
         worlds/make_planted6.py worlds/make_planted7.py; do $PY "$g"; done
$PY worlds/make_random.py 1 2 3     # dev seeds; exam seeds (>= 100000) are not built here
$PY scripts/tables.py
$PY scripts/figure.py
