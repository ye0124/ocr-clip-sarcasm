"""Paired bootstrap + with/without-embedded-text analysis on saved test predictions.

Uses only predictions already saved by the --stage test runs (no new test evaluation).
Usage: python -m sarcasm.analysis.test_bootstrap [n_boot]   -> ${OUTPUTS}/analysis/test_bootstrap.json
Models: "old" = main model (OCR-enhanced CLIP), "new" = text-swap + masking variant, "wide" = CLIP + wide MLP,
"old_noocr" = main model without OCR text, "sepfusion" = separately trained fusion model.
"""
import json, sys
from pathlib import Path
import numpy as np

from ..config import roots
from ..utils import ocr_sha256, ocr_word_counts

OUT = Path(roots()["OUTPUTS"])
INP = OUT
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
RNG_SEED = 20261108


def load(path):
    rows = json.loads((INP / path).read_text())
    return {r["id"]: r for r in rows}


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


S = [42, 43, 44]
S5 = [42, 43, 44, 45, 46]
paths = {
    "new": {s: f"text_swap_variant/variant_seed{s}/test_predictions.json" for s in S5},
    "old": {s: f"main_model/main_seed{s}/test_predictions.json" for s in S5},
    "wide": {s: f"clip_baselines/wide_mlp_seed{s}/test_predictions.json" for s in S},
    "old_noocr": {s: f"main_model/no_ocr_text_seed{s}/test_predictions.json" for s in S},
}
to_paths = {s: f"ocr_text_branch/ocr_branch_lr1e-05_seed{s}/test_predictions.json" for s in S}

# reference order and labels
ref = json.loads((INP / paths["wide"][42]).read_text())
ids = [r["id"] for r in ref]
y = np.array([r["label"] for r in ref])
N = len(ids)

preds = {}
for m, d in paths.items():
    preds[m] = {}
    for s, p in d.items():
        rows = load(p)
        assert set(rows) == set(ids), (m, s)
        assert all(rows[i]["label"] == y[k] for k, i in enumerate(ids))
        preds[m][s] = np.array([rows[i]["prediction"] for i in ids])

# separately trained fusion: equal-weight average of wide-MLP and OCR-text-branch probabilities
preds["sepfusion"] = {}
for s in S:
    w = load(paths["wide"][s]); t = load(to_paths[s])
    pw = np.array([w[i]["probabilities"] for i in ids])
    pt = softmax([t[i]["logits"] for i in ids])
    preds["sepfusion"][s] = ((pw + pt) / 2).argmax(1)

# OCR groups: images with / without recognised text, and long OCR text
words = ocr_word_counts("test")
n_words = np.array([words[i] for i in ids])
has_text = n_words > 0


def macro_f1_counts(y, p, idx=None):
    """Vectorised over bootstrap: idx is (B, n) index matrix or None."""
    if idx is None:
        yy, pp = y[None], p[None]
    else:
        yy, pp = y[idx], p[idx]
    tp1 = ((yy == 1) & (pp == 1)).sum(-1); fp1 = ((yy == 0) & (pp == 1)).sum(-1); fn1 = ((yy == 1) & (pp == 0)).sum(-1)
    tp0 = ((yy == 0) & (pp == 0)).sum(-1)
    f1_1 = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1)
    f1_0 = 2 * tp0 / np.maximum(2 * tp0 + fn1 + fp1, 1)
    return 100 * (f1_0 + f1_1) / 2


def rates(y, p):
    neg, pos = y == 0, y == 1
    return {"fpr_nonsarcastic": round(100 * (p[neg] == 1).mean(), 2),
            "fnr_sarcastic": round(100 * (p[pos] == 0).mean(), 2),
            "n_nonsarcastic": int(neg.sum()), "n_sarcastic": int(pos.sum())}


rng = np.random.default_rng(RNG_SEED)


def boot_idx(n):
    return rng.integers(0, n, size=(N_BOOT, n))


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def paired(a, b, seeds, mask=None):
    """Paired bootstrap over test samples (same resample for both models and all seeds)."""
    sel = np.arange(N) if mask is None else np.where(mask)[0]
    yy = y[sel]
    idx = boot_idx(len(sel))
    per_seed, boot_mean = {}, np.zeros(N_BOOT)
    for s in seeds:
        pa, pb = preds[a][s][sel], preds[b][s][sel]
        d_obs = float(macro_f1_counts(yy, pa)[0] - macro_f1_counts(yy, pb)[0])
        d_boot = macro_f1_counts(yy, pa, idx) - macro_f1_counts(yy, pb, idx)
        boot_mean += d_boot / len(seeds)
        per_seed[s] = {"diff": round(d_obs, 2), "ci95": ci(d_boot), "p_le_0": round(float((d_boot <= 0).mean()), 4)}
    obs = np.mean([per_seed[s]["diff"] for s in seeds])
    # two-level: resample seeds (with replacement) and samples
    seed_draw = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    d_by_seed = np.stack([macro_f1_counts(yy, preds[a][s][sel], idx) - macro_f1_counts(yy, preds[b][s][sel], idx) for s in seeds], 1)
    two_level = np.take_along_axis(d_by_seed, seed_draw, 1).mean(1)
    return {"a": a, "b": b, "seeds": seeds, "n": int(len(sel)),
            "mean_diff": round(float(obs), 2),
            "ci95_samples": ci(boot_mean), "p_le_0_samples": round(float((boot_mean <= 0).mean()), 4),
            "ci95_samples_and_seeds": ci(two_level),
            "per_seed": per_seed}, boot_mean


def summary(m, seeds, mask=None):
    sel = np.ones(N, bool) if mask is None else mask
    f = [float(macro_f1_counts(y[sel], preds[m][s][sel])[0]) for s in seeds]
    r = [rates(y[sel], preds[m][s][sel]) for s in seeds]
    return {"macro_f1_per_seed": [round(v, 2) for v in f], "mean": round(float(np.mean(f)), 2),
            "sd": round(float(np.std(f, ddof=1)), 2),
            "fpr_nonsarcastic_mean": round(float(np.mean([x["fpr_nonsarcastic"] for x in r])), 2),
            "fnr_sarcastic_mean": round(float(np.mean([x["fnr_sarcastic"] for x in r])), 2),
            "n": int(sel.sum()), "n_nonsarcastic": r[0]["n_nonsarcastic"], "n_sarcastic": r[0]["n_sarcastic"]}


out = {"n_boot": N_BOOT, "rng_seed": RNG_SEED, "n_test": N,
       "n_with_text": int(has_text.sum()), "n_without_text": int((~has_text).sum()),
       "ocr_sha256": ocr_sha256("test")}

# 1. overall summaries (sanity check against recorded numbers)
out["overall"] = {m: summary(m, S5 if m in ("new", "old") else S) for m in preds}
out["overall_seeds_42_44"] = {m: summary(m, S) for m in preds}

# 2. paired bootstrap, full test set
comparisons = [("new", "old", S5), ("new", "old", S), ("new", "wide", S), ("new", "sepfusion", S),
               ("old", "wide", S), ("sepfusion", "wide", S), ("old", "old_noocr", S)]
out["paired_overall"] = [paired(a, b, s)[0] for a, b, s in comparisons]

# 3. with / without embedded text
groups = {"with_text": has_text, "without_text": ~has_text, "ocr_gt60_words": n_words > 60}
out["groups"] = {}
for g, mask in groups.items():
    out["groups"][g] = {"summary": {m: summary(m, S5 if m in ("new", "old") else S, mask) for m in preds},
                        "paired": []}
for a, b, s in comparisons:
    r_t, b_t = paired(a, b, s, has_text)
    r_n, b_n = paired(a, b, s, ~has_text)
    out["groups"]["with_text"]["paired"].append(r_t)
    out["groups"]["without_text"]["paired"].append(r_n)
    out.setdefault("interaction_with_minus_without", []).append(
        {"a": a, "b": b, "seeds": s, "diff": round(r_t["mean_diff"] - r_n["mean_diff"], 2),
         "ci95": ci(b_t - b_n)})  # independent resamples of disjoint groups
for a, b, s in comparisons:
    out["groups"]["ocr_gt60_words"]["paired"].append(paired(a, b, s, n_words > 60)[0])

(OUT / "analysis").mkdir(parents=True, exist_ok=True)
(OUT / "analysis/test_bootstrap.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("n_with_text", "n_without_text")}))
for m, v in out["overall"].items():
    print(f"{m:10s} {v['macro_f1_per_seed']} mean {v['mean']} sd {v['sd']} FPR {v['fpr_nonsarcastic_mean']} FNR {v['fnr_sarcastic_mean']}")
print("\npaired, full test")
for r in out["paired_overall"]:
    print(f"{r['a']}-{r['b']} seeds{r['seeds'][0]}-{r['seeds'][-1]}: {r['mean_diff']:+.2f} CI {r['ci95_samples']} (seeds+samples {r['ci95_samples_and_seeds']}) p<=0 {r['p_le_0_samples']}  per-seed {[v['diff'] for v in r['per_seed'].values()]}")
for g in ("with_text", "without_text", "ocr_gt60_words"):
    print(f"\n{g} (n={out['groups'][g]['summary']['new']['n']})")
    for m, v in out["groups"][g]["summary"].items():
        print(f"  {m:10s} F1 {v['mean']} ± {v['sd']}  FPR {v['fpr_nonsarcastic_mean']} FNR {v['fnr_sarcastic_mean']}")
    for r in out["groups"][g]["paired"]:
        print(f"  {r['a']}-{r['b']} s{r['seeds'][0]}-{r['seeds'][-1]}: {r['mean_diff']:+.2f} CI {r['ci95_samples']} p<=0 {r['p_le_0_samples']}")
print("\ninteraction (with − without)")
for r in out["interaction_with_minus_without"]:
    print(f"  {r['a']}-{r['b']} s{r['seeds'][0]}-{r['seeds'][-1]}: {r['diff']:+.2f} CI {r['ci95']}")
