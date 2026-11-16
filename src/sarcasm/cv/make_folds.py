"""5-fold stratified, image-grouped split of MMSD2.0 train+validation (test is never read).

Rows with the same image (PaddleOCR image_sha256) always share a fold. Inside the 4 training folds of each
outer fold, ~10% of image groups are held out as the inner validation set (checkpoint selection only).
Usage: python -m sarcasm.cv.make_folds [--output ${DATA}/cv5/folds.json]
The shipped data/cv5/folds.json was produced by this script (sha256 710e0905...).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import Dataset, concatenate_datasets

SEED, K, INNER_BINS = 20261108, 5, 10
from ..config import roots

_R = roots()
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--arrow-root", type=Path, default=Path(_R["ARROW_ROOT"]))
_ap.add_argument("--evidence-root", type=Path, default=Path(_R["DATA"]) / "evidence")
_ap.add_argument("--output", type=Path, default=Path(_R["DATA"]) / "cv5" / "folds.json")
_a = _ap.parse_args()
root, evid, out = _a.arrow_root, _a.evidence_root, _a.output
out.parent.mkdir(parents=True, exist_ok=True)
ids, labels, sha, split_of = [], [], [], []
for split, n in (("train", 19816), ("validation", 2410)):
    parts = [Dataset.from_file(str(p)) for p in sorted(root.glob(f"mmsd2.0-{split}*.arrow"))]
    d = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    d = d.remove_columns([c for c in d.column_names if c not in ("id", "label")])
    assert len(d) == n
    ocr = [json.loads(l) for l in (evid / split / "ocr.jsonl").read_text().splitlines()]
    for o, i, y in zip(ocr, d["id"], d["label"]):
        assert o["id"] == str(i)
        ids.append(str(i)); labels.append(int(y)); sha.append(o["image_sha256"]); split_of.append(split)
assert len(set(ids)) == len(ids)
labels = np.array(labels)


def assign(members, bins, seed):
    """Greedy stratified assignment of image groups to bins."""
    groups = defaultdict(list)
    for r in members:
        groups[sha[r]].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    keys.sort(key=lambda g: -len(groups[g]))  # stable: large groups first, random order within size
    total = np.bincount(labels[members], minlength=2).astype(float)
    counts = np.zeros((bins, 2))
    out = [[] for _ in range(bins)]
    for g in keys:
        c = np.bincount(labels[groups[g]], minlength=2)
        load = ((counts + c) / total).max(1)
        b = int(np.lexsort((counts.sum(1), load))[0])
        counts[b] += c; out[b].extend(groups[g])
    return out


outer = assign(list(range(len(ids))), K, SEED)
folds, stats = {}, {}
for k in range(K):
    held = sorted(outer[k])
    rest = sorted(r for j in range(K) if j != k for r in outer[j])
    inner = assign(rest, INNER_BINS, SEED + 1 + k)
    val = sorted(inner[0]); val_set = set(val)
    train = [r for r in rest if r not in val_set]
    for a, b in ((train, val), (train, held), (val, held)):
        assert not ({sha[r] for r in a} & {sha[r] for r in b})
    folds[str(k)] = {"train": [ids[r] for r in train], "validation": [ids[r] for r in val], "heldout": [ids[r] for r in held]}
    stats[str(k)] = {name: {"n": len(rs), "sarcastic": int(labels[rs].sum()),
                            "from_original_validation": sum(split_of[r] == "validation" for r in rs)}
                     for name, rs in (("train", train), ("validation", val), ("heldout", held))}
assert sorted(i for k in folds for i in folds[k]["heldout"]) == sorted(ids)
n_groups = len(set(sha))
out.write_text(json.dumps({"seed": SEED, "k": K, "inner_bins": INNER_BINS, "pool": "train+validation",
                           "n_rows": len(ids), "n_image_groups": n_groups, "stats": stats, "folds": folds}))
print(json.dumps({"n_rows": len(ids), "n_image_groups": n_groups, "stats": stats}, indent=1))
