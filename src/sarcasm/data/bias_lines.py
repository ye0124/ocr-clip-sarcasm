"""Line-level bias model (train only): cross-fitted multinomial Naive Bayes on OCR line text.

Score = P(sarcastic | line text) under equal class priors, from a model that never saw the line's image
(5 folds by sha256(image id) % 5). Only eligible donor lines are scored. Output: JSON with per-line scores
and the two donor pools (top / bottom `quantile` of scores).
"""
import argparse, hashlib, json, math, re, collections
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")


def eligible(line):
    x0, y0, x1, y1 = line["box_normalized"]
    return line["score"] >= 0.8 and (y1 - y0) >= 0.02 and (x1 - x0) >= 0.06 and len(line["text"].strip()) >= 3


def features(text):
    toks = TOKEN.findall(text.lower())
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


def fold(image_id):
    return int(hashlib.sha256(str(image_id).encode()).hexdigest(), 16) % 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", type=Path, required=True)        # train ocr.jsonl
    ap.add_argument("--labels", type=Path, required=True)     # json {id: label}
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.02)
    ap.add_argument("--probe-ocr", type=Path)  # validation ocr.jsonl: scored by the all-train model (diagnostics only)
    a = ap.parse_args()
    labels = json.loads(a.labels.read_text())
    rows = [json.loads(l) for l in a.ocr.read_text().splitlines()]
    lines = []  # (image id, line index, fold, label, feats)
    for r in rows:
        for k, line in enumerate(r["lines"]):
            if eligible(line):
                lines.append((r["id"], k, fold(r["id"]), labels[r["id"]], features(line["text"]), line["text"]))
    counts = [[collections.Counter(), collections.Counter()] for _ in range(5)]
    totals = [[0, 0] for _ in range(5)]
    for _, _, f, y, fe, _ in lines:
        counts[f][y].update(fe); totals[f][y] += len(fe)
    allc = [sum((counts[f][y] for f in range(5)), collections.Counter()) for y in (0, 1)]
    allt = [sum(totals[f][y] for f in range(5)) for y in (0, 1)]
    vocab = len(set(allc[0]) | set(allc[1]))
    scored = []
    for iid, k, f, y, fe, text in lines:
        ll = [0.0, 0.0]
        for c in (0, 1):
            n = allc[c]; t = allt[c] - totals[f][c]
            for w in fe:
                ll[c] += math.log((n[w] - counts[f][c][w] + 1) / (t + vocab))
        d = max(-50, min(50, ll[1] - ll[0]))
        scored.append(dict(id=iid, line=k, score=1 / (1 + math.exp(-d)), label=y, text=text))
    order = sorted(scored, key=lambda s: s["score"])
    n = max(1, int(len(order) * a.quantile))
    out = dict(rule=dict(model="multinomial NB, unigrams+bigrams, alpha=1, equal priors, 5-fold by sha256(id)%5",
                         eligible="score>=0.8, h>=0.02, w>=0.06, len>=3", quantile=a.quantile),
               n_eligible=len(scored), n_images=len({s["id"] for s in scored}),
               positive_pool=[[s["id"], s["line"]] for s in order[-n:]],
               negative_pool=[[s["id"], s["line"]] for s in order[:n]],
               uniform_pool=[[s["id"], s["line"]] for s in scored])
    if a.probe_ocr:
        probe = []
        for line_json in a.probe_ocr.read_text().splitlines():
            r = json.loads(line_json)
            for k, line in enumerate(r["lines"]):
                if eligible(line):
                    fe = features(line["text"])
                    ll = [sum(math.log((allc[c][w] + 1) / (allt[c] + vocab)) for w in fe) for c in (0, 1)]
                    d = max(-50, min(50, ll[1] - ll[0]))
                    probe.append((1 / (1 + math.exp(-d)), r["id"], k))
        probe.sort()
        m = max(1, int(len(probe) * a.quantile))
        out.update(probe_n_eligible=len(probe), probe_positive_pool=[[i, k] for _, i, k in probe[-m:]],
                   probe_negative_pool=[[i, k] for _, i, k in probe[:m]])
    a.output.write_text(json.dumps(out))
    print(json.dumps(dict(n_eligible=len(scored), pool=n,
          pos_pool_label1=sum(s["label"] for s in order[-n:]) / n,
          neg_pool_label1=sum(s["label"] for s in order[:n]) / n,
          pos_examples=[s["text"] for s in order[-n:][::max(1, n // 25)]][:25],
          neg_examples=[s["text"] for s in order[:n][::max(1, n // 15)]][:15]), ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
