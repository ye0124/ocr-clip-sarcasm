"""Pre-registered analysis of the 5-fold CV held-out predictions (frozen with protocol.json).

Usage: python -m sarcasm.cv.analyze_cv   -> ${OUTPUTS}/cv5/results.json
Arms: wide (CLIP + wide MLP), full (main model), S45S2 (text-swap + masking variant).
"""
import json, math, sys
from pathlib import Path
import numpy as np

from ..config import roots
from ..utils import ocr_word_counts

RUNS = Path(roots()["OUTPUTS"]) / "cv5"
ARMS, K, N_BOOT, SEED = ("wide", "full", "S45S2"), 5, 10000, 20261108
COMPARISONS = [("S45S2", "wide"), ("S45S2", "full"), ("full", "wide")]

words = {**ocr_word_counts("train"), **ocr_word_counts("validation")}  # id -> words of OCR text

folds = {}
for k in range(K):
    rows = {a: json.loads((RUNS / f"{a}_f{k}/heldout_predictions.json").read_text()) for a in ARMS}
    ids = [r["id"] for r in rows["wide"]]
    for a in ARMS:
        assert [r["id"] for r in rows[a]] == ids
        assert [r["label"] for r in rows[a]] == [r["label"] for r in rows["wide"]]
    folds[k] = dict(y=np.array([r["label"] for r in rows["wide"]]),
                    p={a: np.array([r["prediction"] for r in rows[a]]) for a in ARMS},
                    has_text=np.array([words[i] > 0 for i in ids]),
                    gt60=np.array([words[i] > 60 for i in ids]))


def f1(y, p):
    """Macro-F1 (%) along the last axis."""
    tp1 = ((y == 1) & (p == 1)).sum(-1); fp1 = ((y == 0) & (p == 1)).sum(-1); fn1 = ((y == 1) & (p == 0)).sum(-1)
    tp0 = ((y == 0) & (p == 0)).sum(-1)
    return 100 * (tp1 / np.maximum(tp1 + (fp1 + fn1) / 2, 1) + tp0 / np.maximum(tp0 + (fp1 + fn1) / 2, 1)) / 2


def rates(y, p):
    return dict(fpr=round(100 * float((p[y == 0] == 1).mean()), 2), fnr=round(100 * float((p[y == 1] == 0).mean()), 2))


def t_stat(d):
    d = np.asarray(d, float)
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("inf")


def p_two_sided_t(t, df):
    # Student t CDF via regularized incomplete beta (continued fraction); df small, no scipy dependency
    x = df / (df + t * t)
    def betacf(a, b, x):
        qab, qap, qam, c, d = a + b, a + 1, a - 1, 1.0, 1 - (a + b) * x / (a + 1)
        d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
        for m in range(1, 200):
            m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30); c = 1 + aa / c; h *= d * c
        return h
    a, b = df / 2, 0.5
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    ibeta = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    return float(min(1.0, ibeta))


def analyse(mask_key=None):
    rng = np.random.default_rng(SEED)
    out = dict(per_arm={}, comparisons=[])
    def mask(f):
        if mask_key is None:
            return np.ones(len(f["y"]), bool)
        return ~f["has_text"] if mask_key == "no_text" else f[mask_key]
    sel = {k: mask(f) for k, f in folds.items()}
    for a in ARMS:
        s = [float(f1(folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]])) for k in range(K)]
        r = [rates(folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]]) for k in range(K)]
        out["per_arm"][a] = dict(per_fold=[round(v, 2) for v in s], mean=round(float(np.mean(s)), 2),
                                 sd=round(float(np.std(s, ddof=1)), 2),
                                 fpr_mean=round(float(np.mean([x["fpr"] for x in r])), 2),
                                 fnr_mean=round(float(np.mean([x["fnr"] for x in r])), 2))
    out["n"] = int(sum(m.sum() for m in sel.values()))
    # bootstrap indices shared by all comparisons: resample rows within each fold
    idx = {k: rng.integers(0, int(sel[k].sum()), size=(N_BOOT, int(sel[k].sum()))) for k in range(K)}
    for a, b in COMPARISONS:
        d = [float(f1(folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]]) - f1(folds[k]["y"][sel[k]], folds[k]["p"][b][sel[k]]))
             for k in range(K)]
        boot = np.zeros(N_BOOT)
        for k in range(K):
            y, pa, pb = folds[k]["y"][sel[k]], folds[k]["p"][a][sel[k]], folds[k]["p"][b][sel[k]]
            boot += (f1(y[idx[k]], pa[idx[k]]) - f1(y[idx[k]], pb[idx[k]])) / K
        t = t_stat(d)
        out["comparisons"].append(dict(
            a=a, b=b, per_fold=[round(x, 2) for x in d], mean=round(float(np.mean(d)), 2),
            sd=round(float(np.std(d, ddof=1)), 2), positive_folds=int(sum(x > 0 for x in d)),
            t=round(t, 2), p_two_sided=round(p_two_sided_t(t, K - 1), 4) if math.isfinite(t) else 0.0,
            bootstrap_ci95=[round(float(v), 2) for v in np.percentile(boot, [2.5, 97.5])],
            bootstrap_p_le_0=round(float((boot <= 0).mean()), 4), _boot=boot))
    return out


res = {"n_boot": N_BOOT, "seed": SEED}
for name, key in (("all", None), ("with_text", "has_text"), ("no_text", "no_text"), ("ocr_gt60_words", "gt60")):
    res[name] = analyse(key)
res["interaction_with_minus_without"] = []
for cw, cn in zip(res["with_text"]["comparisons"], res["no_text"]["comparisons"]):
    diff = cw["_boot"] - cn["_boot"]
    res["interaction_with_minus_without"].append(dict(a=cw["a"], b=cw["b"], diff=round(cw["mean"] - cn["mean"], 2),
                                                      ci95=[round(float(v), 2) for v in np.percentile(diff, [2.5, 97.5])]))
for name in ("all", "with_text", "no_text", "ocr_gt60_words"):
    for c in res[name]["comparisons"]:
        c.pop("_boot")
(RUNS / "results.json").write_text(json.dumps(res, indent=1))
for name in ("all", "with_text", "no_text", "ocr_gt60_words"):
    r = res[name]; print(f"\n== {name} (n={r['n']})")
    for a, v in r["per_arm"].items():
        print(f"  {a:6s} {v['per_fold']} mean {v['mean']} ± {v['sd']}  FPR {v['fpr_mean']} FNR {v['fnr_mean']}")
    for c in r["comparisons"]:
        print(f"  {c['a']}-{c['b']}: {c['per_fold']} mean {c['mean']:+.2f} ± {c['sd']} pos {c['positive_folds']}/5 "
              f"t {c['t']} p {c['p_two_sided']} boot {c['bootstrap_ci95']}")
print("\ninteraction", res["interaction_with_minus_without"])
