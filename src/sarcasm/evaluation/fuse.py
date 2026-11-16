"""Separately trained fusion model on the official split: P = 0.5*P_wide + 0.5*P_OCR, sarcasm iff P > 0.5.

No weight or threshold is searched. Seeds are paired (wide_mlp_seedS with ocr_branch_lr1e-05_seedS).

  python -m sarcasm.evaluation.fuse --split test --seeds 42 43 44
Reads ${OUTPUTS}/clip_baselines/wide_mlp_seed{S}/{split}_predictions.json and
${OUTPUTS}/ocr_text_branch/ocr_branch_lr1e-05_seed{S}/{split}_predictions.json; writes
${OUTPUTS}/fusion/fusion_{split}.json.
"""
import argparse, json, math
from pathlib import Path

from ..config import roots
from ..utils import atomic_json, score

WEIGHT_B = 0.5


def probability(row):
    """P(sarcasm) from a prediction row (probabilities if stored, else softmax of logits)."""
    if "probabilities" in row:
        return float(row["probabilities"][1])
    a, b = row["logits"]
    m = max(a, b)
    return math.exp(b - m) / (math.exp(a - m) + math.exp(b - m))


def load(path):
    rows = json.loads(Path(path).read_text())
    out = {str(r["id"]): (int(r["label"]), probability(r)) for r in rows}
    assert len(out) == len(rows)
    return out


def metrics(pairs):
    return score([y for y, _ in pairs], [int(p > 0.5) for _, p in pairs])


def main():
    out_root = Path(roots()["OUTPUTS"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--wide-dir", type=Path, default=out_root / "clip_baselines")
    ap.add_argument("--ocr-dir", type=Path, default=out_root / "ocr_text_branch")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    output = a.output or out_root / "fusion" / f"fusion_{a.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(split=a.split, weight_wide=WEIGHT_B, threshold=0.5, seeds={})
    for seed in a.seeds:
        paths = dict(wide=a.wide_dir / f"wide_mlp_seed{seed}" / f"{a.split}_predictions.json",
                     ocr=a.ocr_dir / f"ocr_branch_lr1e-05_seed{seed}" / f"{a.split}_predictions.json")
        B, T = load(paths["wide"]), load(paths["ocr"])
        assert set(B) == set(T)
        ids = sorted(B)
        assert all(B[i][0] == T[i][0] for i in ids)
        fused = [(B[i][0], WEIGHT_B * B[i][1] + (1 - WEIGHT_B) * T[i][1]) for i in ids]
        report["seeds"][str(seed)] = dict(rows=len(ids), wide=metrics([B[i] for i in ids]),
                                          ocr_branch=metrics([T[i] for i in ids]), fusion=metrics(fused))
    for model in ("wide", "ocr_branch", "fusion"):
        for key in ("macro_f1", "accuracy", "sarcasm_f1"):
            values = [report["seeds"][s][model][key] for s in report["seeds"]]
            report.setdefault("mean", {}).setdefault(model, {})[key] = sum(values) / len(values)
    atomic_json(report, output)
    print(json.dumps(report["mean"], indent=1), flush=True)


if __name__ == "__main__":
    main()
