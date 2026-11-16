"""Pre-registered analysis of the no-OCR text-branch control (frozen with the protocol before training).

Question: is the gain of the separately trained fusion model over CLIP + wide MLP due to the in-image text, or to
ensembling a second (DeBERTa) model?
Arms (held-out predictions, same folds and seeds 42+k):
  wide, full, sep, sepn   ${OUTPUTS}/cv5: wide MLP, main model, wide + to fusion, wide + tn fusion (fuse_cv)
  to, tn                  ${OUTPUTS}/cv5_ocr_branch: standalone text branch with / without OCR text
Primary: sep - sepn (OCR contribution inside the fusion).
Secondary: sepn - wide (ensemble effect without OCR), to - tn (OCR effect in the standalone branch), full - sepn,
sep - wide (reference).
Decision rule fixed before training: OCR is said to contribute to the fusion gain only if sep - sepn > 0 on all rows
and its bootstrap 95% CI excludes 0.
Usage: python -m sarcasm.cv.analyze_cv_textonly   -> ${OUTPUTS}/cv5/results_textonly.json
"""
from pathlib import Path

from ..config import roots
from .cv_stats import run

OUT = Path(roots()["OUTPUTS"])
SOURCES = {"wide": "cv5", "full": "cv5", "to": "cv5_ocr_branch", "sep": "cv5", "tn": "cv5_ocr_branch", "sepn": "cv5"}
SOURCES = {a: str(OUT / sub / f"{a}_f{{k}}") for a, sub in SOURCES.items()}
COMPARISONS = [("sep", "sepn"), ("sepn", "wide"), ("to", "tn"), ("full", "sepn"), ("sep", "wide")]
PRIMARY = [("sep", "sepn")]

if __name__ == "__main__":
    run(SOURCES, COMPARISONS, PRIMARY, 10000, 20261109, OUT / "cv5" / "results_textonly.json")
