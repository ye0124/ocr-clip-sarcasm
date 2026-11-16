#!/usr/bin/env bash
# Pipeline check (~10 min): 1-2 epochs on 32 training rows for each trainer. Writes to outputs/smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}; S=${SARCASM_OUTPUTS:-outputs}/smoke
rm -rf "$S"
$PY -m sarcasm.train.train_clip --config configs/clip_baselines.json --run wide_mlp_seed42 --stage train --smoke --output "$S/wide"
$PY -m sarcasm.train.train_ocr_branch --config configs/ocr_text_branch.json --run ocr_branch_lr1e-05_seed42 --stage train --smoke --output "$S/ocr"
$PY -m sarcasm.train.train_main --config configs/main_model.json --run main_seed42 --stage train --smoke --output "$S/main"
$PY -m sarcasm.train.train_main --config configs/text_swap_variant.json --run variant_seed42 --stage train --smoke --output "$S/variant"
$PY -m sarcasm.cv.cv_train --config configs/cv5.json --stage train --run full_f0 --smoke --output "$S/cv_full"
$PY -m sarcasm.cv.cv_train --config configs/cv5.json --stage heldout --run full_f0 --smoke --output "$S/cv_full"
echo "smoke test passed"
