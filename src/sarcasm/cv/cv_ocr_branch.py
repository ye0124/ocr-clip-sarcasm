"""5-fold CV for the standalone OCR text branch (sarcasm.train.train_ocr_branch with fold-wise data).

FoldCachedData pools the frozen CLIP cache and PaddleOCR of MMSD2.0 train+validation and keeps the rows that
folds.json assigns to the fold's inner-train / inner-validation / held-out part. Tokenisation, model,
optimiser, schedule, selection and evaluation are the standalone trainer unchanged. The test split is never read.

  python -m sarcasm.cv.cv_ocr_branch --config configs/cv5_ocr_branch.json --stage train --run to_f0
  python -m sarcasm.cv.cv_ocr_branch --config configs/cv5_ocr_branch.json --stage heldout --run to_f0
"""
import argparse, json
from pathlib import Path

import numpy as np

from ..config import default_output, load_config, run_config
from ..train import train_ocr_branch as S

ORIGINAL = S.CachedData


def fold_data(cfg):
    """CachedData subclass bound to the fold of ``cfg``."""

    class FoldCachedData(ORIGINAL):
        """Same attributes as CachedData; split in {"train", "validation", "heldout"} = fold parts."""

        def __init__(self, root, split, tokenizer, mode, evidence, limit=None):
            parts = [ORIGINAL(root, s, tokenizer, mode, evidence) for s in ("train", "validation")]
            records = parts[0].records + parts[1].records
            vectors = np.concatenate([np.asarray(p.vectors) for p in parts])
            views = parts[0].views + parts[1].views
            presence = np.concatenate([p.presence for p in parts])
            ids = [r["id"] for r in records]
            assert len(set(ids)) == len(ids) == 22226
            assert S.sha(Path(cfg["cv_folds"])) == cfg["cv_folds_sha256"], "folds changed"
            wanted = json.loads(Path(cfg["cv_folds"]).read_text())["folds"][str(cfg["fold"])][split]
            row = {i: k for k, i in enumerate(ids)}
            positions = sorted(row[str(i)] for i in wanted)
            assert len(positions) == len(wanted)
            labels = [r["label"] for r in records]
            if limit:
                positions = sorted(i for y in (0, 1) for i in [j for j in positions if labels[j] == y][:limit // 2])
            self.records = [records[p] for p in positions]
            self.vectors = vectors[positions]
            self.views = [views[p] for p in positions]
            self.presence = presence[positions]
            self.ids = [r["id"] for r in self.records]
            self.labels = np.array([r["label"] for r in self.records])
            # truncation counts of the whole train+validation pool (CachedData keeps no per-row flags)
            self.truncation = dict(scope="train+validation pool",
                                   post=sum(p.truncation["post"] for p in parts),
                                   ocr=sum(p.truncation["ocr"] for p in parts))

    return FoldCachedData


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=("train", "heldout"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--output", type=Path, help="default: ${OUTPUTS}/cv5_ocr_branch/<run>")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(run_config(load_config(args.config), args.run), smoke=args.smoke)
    if args.smoke:
        cfg.update(epochs=1, patience=1, checkpoint_steps=1)
    out = args.output or default_output(args.config, args.run)
    data_cls = fold_data(cfg)
    if args.stage == "train":
        S.train(cfg, out, data_cls)
    else:
        S.evaluate_split(out, "heldout", data_cls, 64 if args.smoke else None)


if __name__ == "__main__":
    main()
