#!/usr/bin/env bash
# Train and test every model on the official MMSD2.0 split (all runs reported in the paper).
#   bash scripts/train_official.sh            # everything (~2.5 GPU days on an RTX 2070 SUPER)
#   bash scripts/train_official.sh main       # only the main model, seeds 42-44 (~3 h)
# Each run: train (checkpoint on validation Macro-F1), then evaluate best.pt once on test.
# Finished stages are skipped, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
OUT=${SARCASM_OUTPUTS:-outputs}
runs() { $PY -c "import json,sys; print(' '.join(r['name'] for r in json.load(open(sys.argv[1]))['runs']))" "$1"; }
go() {  # module config run
  local dir="$OUT/$(basename "$2" .json)/$3"
  [ -f "$dir/result.json" ] || $PY -m "$1" --config "$2" --run "$3" --stage train
  [ -f "$dir/test_metrics.json" ] || $PY -m "$1" --config "$2" --run "$3" --stage test
}
if [ "${1:-all}" = main ]; then
  for s in 42 43 44; do go sarcasm.train.train_main configs/main_model.json main_seed$s; done; exit 0
fi
for r in $(runs configs/clip_baselines.json); do go sarcasm.train.train_clip configs/clip_baselines.json "$r"; done
for r in $(runs configs/ocr_text_branch.json); do go sarcasm.train.train_ocr_branch configs/ocr_text_branch.json "$r"; done
for r in $(runs configs/main_model.json); do go sarcasm.train.train_main configs/main_model.json "$r"; done
for r in $(runs configs/text_swap_variant.json); do go sarcasm.train.train_main configs/text_swap_variant.json "$r"; done
$PY -m sarcasm.evaluation.fuse --split validation --seeds 42 43 44
$PY -m sarcasm.evaluation.fuse --split test --seeds 42 43 44
