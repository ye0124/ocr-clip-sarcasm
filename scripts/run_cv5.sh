#!/usr/bin/env bash
# 5-fold image-grouped CV over MMSD2.0 train+validation (the test split is never read).
# Per fold k: CLIP + wide MLP, main model, text-swap variant, main model without OCR text, standalone text branch
# with OCR text (to) and without (tn); each trained on the fold's inner-train rows and evaluated once on its held-out
# rows. ~4 GPU hours per fold (RTX 2070 SUPER).
# Restartable: finished stages are skipped; training resumes from last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
OUT=${SARCASM_OUTPUTS:-outputs}
for k in 0 1 2 3 4; do
  for arm in wide full S45S2 noocr; do
    run=${arm}_f$k; stage=train; [ $arm = wide ] && stage=train_wide
    [ -f "$OUT/cv5/$run/result.json" ] || $PY -m sarcasm.cv.cv_train --config configs/cv5.json --stage $stage --run $run
    [ -f "$OUT/cv5/$run/heldout_metrics.json" ] || $PY -m sarcasm.cv.cv_train --config configs/cv5.json --stage heldout --run $run
  done
  for run in to_f$k tn_f$k; do
    [ -f "$OUT/cv5_ocr_branch/$run/result.json" ] || $PY -m sarcasm.cv.cv_ocr_branch --config configs/cv5_ocr_branch.json --stage train --run $run
    [ -f "$OUT/cv5_ocr_branch/$run/heldout_metrics.json" ] || $PY -m sarcasm.cv.cv_ocr_branch --config configs/cv5_ocr_branch.json --stage heldout --run $run
  done
done
$PY -m sarcasm.cv.fuse_cv
$PY -m sarcasm.cv.analyze_cv
$PY -m sarcasm.cv.analyze_cv_supplement
$PY -m sarcasm.cv.analyze_cv_textonly
