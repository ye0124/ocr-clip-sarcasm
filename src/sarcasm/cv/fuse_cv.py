"""Separately trained fusion per CV fold: P = 0.5*P_wide + 0.5*P_text, sarcasm iff P > 0.5.

Same rule as sarcasm.evaluation.fuse (no weight or threshold search). P_wide: held-out predictions of
${OUTPUTS}/cv5/wide_f{k}; P_text: softmax of the held-out logits of a standalone text branch in ${OUTPUTS}/cv5_ocr_branch.
  sep_f{k}  = wide + to_f{k}  (text branch reads post + OCR text: the separately trained fusion model)
  sepn_f{k} = wide + tn_f{k}  (same text branch without OCR text)
Writes ${OUTPUTS}/cv5/{sep,sepn}_f{k}/heldout_predictions.json and heldout_metrics.json; fusions whose text-branch
predictions are missing are skipped.
"""
import json
from pathlib import Path

from ..config import roots
from ..evaluation.fuse import probability
from ..utils import atomic_json, score

FUSIONS = {"sep": "to", "sepn": "tn"}


def main():
    out_root = Path(roots()["OUTPUTS"])
    for fused, branch in FUSIONS.items():
        for k in range(5):
            out = out_root / f"cv5/{fused}_f{k}"
            source = out_root / f"cv5_ocr_branch/{branch}_f{k}/heldout_predictions.json"
            if (out / "heldout_predictions.json").exists() or not source.exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            wide = json.loads((out_root / f"cv5/wide_f{k}/heldout_predictions.json").read_text())
            text = {r["id"]: r for r in json.loads(source.read_text())}
            assert len(text) == len(wide)
            rows = []
            for w in wide:
                t = text[w["id"]]; assert t["label"] == w["label"]
                pw, pt = probability(w), probability(t); p = 0.5 * pw + 0.5 * pt
                rows.append({"id": w["id"], "label": w["label"], "prediction": int(p > 0.5), "p_sarcasm": p,
                             "p_wide": pw, f"p_{branch}": pt})
            y = [r["label"] for r in rows]
            m = {fused: score(y, [r["prediction"] for r in rows]),
                 "wide": score(y, [int(r["p_wide"] > 0.5) for r in rows]),
                 branch: score(y, [int(r[f"p_{branch}"] > 0.5) for r in rows]), "fold": k, "n": len(rows)}
            m["macro_f1"] = m[fused]["macro_f1"]
            atomic_json(rows, out / "heldout_predictions.json"); atomic_json(m, out / "heldout_metrics.json")
            print(fused, k, round(100 * m["macro_f1"], 2), flush=True)


if __name__ == "__main__":
    main()
