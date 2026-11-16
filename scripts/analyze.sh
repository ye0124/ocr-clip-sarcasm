#!/usr/bin/env bash
# Analyses on saved predictions (no new training, no new test evaluation), inference cost and label statistics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}; OCR_PY=${OCR_PY:-python}
DATA=${SARCASM_DATA:-data}; OUT=${SARCASM_OUTPUTS:-outputs}
$PY -m sarcasm.analysis.test_bootstrap 10000
$PY -m sarcasm.analysis.cost_model "$OUT/main_model/main_seed42"
$OCR_PY src/sarcasm/analysis/cost_ocr.py "$(cat "$DATA/arrow_root.txt")" "$OUT/analysis/cost_ocr.json"
$PY -m sarcasm.analysis.shortcut_diagnostics --config configs/text_swap_variant.json \
    --run-dir "$OUT/main_model/main_seed42" --output "$OUT/analysis/diag_main_seed42.json"
$PY -m sarcasm.analysis.label_statistics
$PY scripts/make_tables.py
