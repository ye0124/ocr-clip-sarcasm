"""Pre-registered analysis of the CV supplement (frozen with the protocol before training).

Arms (held-out predictions, same folds and seeds 42+k):
  wide, full, S45S2   ${OUTPUTS}/cv5: CLIP + wide MLP, main model, text-swap + masking variant
  noocr               ${OUTPUTS}/cv5: main model with the OCR text removed from the text-branch input
  sep                 ${OUTPUTS}/cv5: separately trained fusion model, wide + standalone OCR text branch (fuse_cv)
Primary: full - sep, full - noocr.  Secondary: sep - wide, noocr - wide, full - wide, S45S2 - wide, S45S2 - full.
Every comparison is repeated on held-out rows from the original MMSD2.0 train split only (original validation
rows, used for earlier model selection, removed) and on the text subgroups (see cv_stats).
Usage: python -m sarcasm.cv.analyze_cv_supplement   -> ${OUTPUTS}/cv5/results_supplement.json
"""
from pathlib import Path

from ..config import roots
from .cv_stats import run

RUNS = Path(roots()["OUTPUTS"]) / "cv5"
SOURCES = {a: str(RUNS / f"{a}_f{{k}}") for a in ("wide", "full", "S45S2", "noocr", "sep")}
COMPARISONS = [("full", "sep"), ("full", "noocr"), ("sep", "wide"), ("noocr", "wide"), ("full", "wide"),
               ("S45S2", "wide"), ("S45S2", "full")]
PRIMARY = [("full", "sep"), ("full", "noocr")]

if __name__ == "__main__":
    run(SOURCES, COMPARISONS, PRIMARY, 10000, 20261109, RUNS / "results_supplement.json")
