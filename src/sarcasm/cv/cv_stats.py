"""Shared statistics of the pre-registered CV analyses (analyze_cv_supplement, analyze_cv_textonly).

Per arm: held-out Macro-F1 per fold, mean ± sd, mean FPR / FNR. Per comparison a - b: per-fold paired Macro-F1
differences, mean ± sd, positive folds, paired t (df 4, scipy), pooled within-fold bootstrap (each replicate
resamples every fold's held-out rows and averages the 5 fold differences).
Subgroups: all rows, original-MMSD2.0-train rows only, images with / without in-image text, OCR > 60 words.
"""
import json, math
from pathlib import Path

import numpy as np
from scipy import stats

from ..utils import ocr_word_counts

K = 5
SUBGROUPS = (("all", None), ("train_only", "train_only"), ("with_text", "has_text"), ("no_text", "no_text"),
             ("ocr_gt60_words", "gt60"))


def load_folds(sources):
    """sources: arm -> run directory pattern with {k}. Returns per-fold labels, predictions and row flags."""
    train_words = ocr_word_counts("train")
    words = {**train_words, **ocr_word_counts("validation")}  # id -> words of OCR text
    train_ids = set(train_words)
    assert len(train_ids) == 19816
    arms = tuple(sources)
    folds = {}
    for k in range(K):
        rows = {a: json.loads((Path(sources[a].format(k=k)) / "heldout_predictions.json").read_text()) for a in arms}
        ids = [r["id"] for r in rows[arms[0]]]
        by = {a: {r["id"]: r for r in rows[a]} for a in arms}
        for a in arms:
            assert set(by[a]) == set(ids) and len(by[a]) == len(ids)
            assert all(by[a][i]["label"] == by[arms[0]][i]["label"] for i in ids)
        folds[k] = dict(y=np.array([by[arms[0]][i]["label"] for i in ids]),
                        p={a: np.array([int(by[a][i]["prediction"]) for i in ids]) for a in arms},
                        has_text=np.array([words[i] > 0 for i in ids]),
                        gt60=np.array([words[i] > 60 for i in ids]),
                        train_only=np.array([i in train_ids for i in ids]))
    return folds


def f1(y, p):
    """Macro-F1 (in %) along the last axis."""
    tp1 = ((y == 1) & (p == 1)).sum(-1); fp1 = ((y == 0) & (p == 1)).sum(-1); fn1 = ((y == 1) & (p == 0)).sum(-1)
    tp0 = ((y == 0) & (p == 0)).sum(-1)
    return 100 * (tp1 / np.maximum(tp1 + (fp1 + fn1) / 2, 1) + tp0 / np.maximum(tp0 + (fp1 + fn1) / 2, 1)) / 2


def rates(y, p):
    return dict(fpr=100 * float((p[y == 0] == 1).mean()), fnr=100 * float((p[y == 1] == 0).mean()))


def analyse(folds, key, comparisons, primary, n_boot, seed):
    rng = np.random.default_rng(seed)
    arms = tuple(folds[0]["p"])
    sel = {k: (np.ones(len(f["y"]), bool) if key is None else ~f["has_text"] if key == "no_text" else f[key])
           for k, f in folds.items()}
    out = dict(n=int(sum(m.sum() for m in sel.values())), per_arm={}, comparisons=[])
    for a in arms:
        s = [float(f1(folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]])) for k in range(K)]
        r = [rates(folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]]) for k in range(K)]
        out["per_arm"][a] = dict(per_fold=[round(v, 2) for v in s], mean=round(float(np.mean(s)), 2),
                                 sd=round(float(np.std(s, ddof=1)), 2),
                                 fpr_mean=round(float(np.mean([x["fpr"] for x in r])), 2),
                                 fnr_mean=round(float(np.mean([x["fnr"] for x in r])), 2))
    idx = {k: rng.integers(0, int(sel[k].sum()), size=(n_boot, int(sel[k].sum()))) for k in range(K)}
    for a, b in comparisons:
        d, boot = [], np.zeros(n_boot)
        for k in range(K):
            y, pa, pb = folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]], folds[k]["p"][b][sel[k]]
            d.append(float(f1(y, pa) - f1(y, pb)))
            boot += (f1(y[idx[k]], pa[idx[k]]) - f1(y[idx[k]], pb[idx[k]])) / K
        d = np.asarray(d); sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / math.sqrt(K))) if sd > 0 else float("inf")
        p = float(2 * stats.t.sf(abs(t), K - 1)) if math.isfinite(t) else 0.0
        out["comparisons"].append(dict(
            a=a, b=b, primary=(a, b) in primary, per_fold=[round(float(x), 2) for x in d],
            mean=round(float(d.mean()), 2), sd=round(sd, 2), positive_folds=int((d > 0).sum()), t=round(t, 2),
            p_two_sided=round(p, 4), bootstrap_ci95=[round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
            bootstrap_p_le_0=round(float((boot <= 0).mean()), 4)))
    return out


def run(sources, comparisons, primary, n_boot, seed, output):
    folds = load_folds(sources)
    res = dict(n_boot=n_boot, seed=seed, primary=[f"{a} - {b}" for a, b in primary])
    for name, key in SUBGROUPS:
        res[name] = analyse(folds, key, comparisons, primary, n_boot, seed)
    Path(output).write_text(json.dumps(res, indent=1))
    for name, _ in SUBGROUPS:
        r = res[name]; print(f"\n== {name} (n={r['n']})")
        for a, v in r["per_arm"].items():
            print(f"  {a:6s} {v['per_fold']} mean {v['mean']} ± {v['sd']}  FPR {v['fpr_mean']} FNR {v['fnr_mean']}")
        for c in r["comparisons"]:
            print(f"  {'*' if c['primary'] else ' '}{c['a']}-{c['b']}: {c['per_fold']} mean {c['mean']:+.2f} ± {c['sd']} "
                  f"pos {c['positive_folds']}/5 t {c['t']} p {c['p_two_sided']} boot {c['bootstrap_ci95']}")
    return res
