#!/usr/bin/env bash
# Data preparation after scripts/download_assets.py. Run from the repository root.
#   1. PaddleOCR on every image (separate environment, see requirements-ocr.txt)  ~1 h on an RTX 2070 SUPER
#   2. frozen CLIP vectors for the standalone OCR text branch                     ~10 min
#   3. bias-line pools for the text-swap variant (official split and CV folds)     CPU, ~2 min
#   4. DINOv2 cache, only with --dino (interaction-gated model)                    ~15 min
# Environment variables: PY (main env python, default python), OCR_PY (PaddleOCR env python, default python).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}; OCR_PY=${OCR_PY:-python}
DATA=${SARCASM_DATA:-data}; MODELS=${SARCASM_MODELS:-models}
ARROW=${SARCASM_ARROW_ROOT:-$(cat "$DATA/arrow_root.txt")}

$OCR_PY src/sarcasm/data/ocr_paddle.py --arrow-root "$ARROW" --output "$DATA/evidence"
$PY -m sarcasm.data.cache_clip --arrow-root "$ARROW" --clip "$MODELS/clip-vit-base-patch32" \
    --output "$DATA/evidence/clip" --splits train validation test

mkdir -p "$DATA/bias_pools"
$PY -m sarcasm.data.prepare_labels --splits train --output "$DATA/bias_pools/train_labels.json"
$PY -m sarcasm.data.bias_lines --ocr "$DATA/evidence/train/ocr.jsonl" --labels "$DATA/bias_pools/train_labels.json" \
    --output "$DATA/bias_pools/train_bias_pools.json" --quantile 0.02 --probe-ocr "$DATA/evidence/validation/ocr.jsonl"
$PY -m sarcasm.data.prepare_labels --splits train validation --output "$DATA/cv5/pool_labels.json"
$PY -m sarcasm.cv.prep_pools

if [ "${1:-}" = "--dino" ]; then $PY -m sarcasm.data.cache_dino; fi
$PY scripts/check_data.py
