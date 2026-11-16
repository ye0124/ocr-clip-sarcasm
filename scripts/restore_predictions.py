"""Restore the shipped per-sample predictions (results/predictions) into the ${OUTPUTS} layout.

Lets the statistical analyses run without any training, GPU or dataset (the text subgroups then use
results/ocr_word_counts.csv.gz):
  python scripts/restore_predictions.py
  python -m sarcasm.analysis.test_bootstrap        # paired bootstrap + text/no-text split on test
  python -m sarcasm.cv.analyze_cv                  # 5-fold CV comparison
  python -m sarcasm.cv.analyze_cv_supplement       # CV: separately trained fusion, no-OCR main model
  python -m sarcasm.cv.analyze_cv_textonly         # CV: fusion with vs without OCR text
Existing prediction files are never overwritten.
"""
import csv, gzip, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sarcasm.config import roots  # noqa: E402


def rows(path):
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            p = float(r["p_sarcasm"])
            out = dict(id=r["id"], label=int(r["label"]), prediction=int(r["prediction"]), probabilities=[1 - p, p])
            if "logit_0" in r:
                out["logits"] = [float(r["logit_0"]), float(r["logit_1"])]
            yield out


def main():
    out = Path(roots()["OUTPUTS"]); n = 0
    for f in sorted((REPO / "results/predictions/test").glob("*.csv.gz")):
        config, run = f.name[:-len(".csv.gz")].split("__")
        target = out / config / run / "test_predictions.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(list(rows(f)))); n += 1
    for f in sorted((REPO / "results/predictions/cv5").glob("*.csv.gz")):
        run = f.name[:-len(".csv.gz")]
        sub = "cv5_ocr_branch" if run.startswith(("to_", "tn_")) else "cv5"
        target = out / sub / run / "heldout_predictions.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(list(rows(f)))); n += 1
    print(f"restored {n} prediction files under {out}")


if __name__ == "__main__":
    main()
