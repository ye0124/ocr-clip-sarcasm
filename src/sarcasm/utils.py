"""Small shared helpers: atomic writes, hashing, metrics, OCR word counts for the subgroup analyses."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import REPO, roots

# SHA-256 of the PaddleOCR output of the original run (${DATA}/evidence/<split>/ocr.jsonl).
ORIGINAL_OCR_SHA256 = {"train": "cfd5b5a5c1bb5e8f427ebdda38338803b36a774028a6793939400e8bb016d9e1",
                       "validation": "eea458b30bce595df867599ca9dbb19073e7aa6c162232844def19474efaa706",
                       "test": "d31f5389c64dc1d2547fbbcd0b010cbe6eb06218ea7f50f923fd21bee1e9ae8f"}
SHIPPED_WORD_COUNTS = REPO / "results" / "ocr_word_counts.csv.gz"


def ocr_word_counts(split):
    """{id: number of whitespace-separated words in the image's OCR text} for one split.

    The subgroup analyses only need this count (0 = image without text, > 60 = long text). It is read from
    ${DATA}/evidence/<split>/ocr.jsonl when that file exists (after scripts/prepare_data.sh); otherwise from
    results/ocr_word_counts.csv.gz, the counts of the original OCR run, so that the statistics can be
    recomputed from the shipped predictions without the dataset or a GPU.
    """
    path = Path(roots()["DATA"]) / "evidence" / split / "ocr.jsonl"
    if path.exists():
        counts = {}
        for line in path.read_text().splitlines():
            r = json.loads(line)
            counts[r["id"]] = len(" ".join(l["text"] for l in r["lines"]).split())
        return counts
    with gzip.open(SHIPPED_WORD_COUNTS, "rt", newline="") as f:
        return {r["id"]: int(r["n_words"]) for r in csv.DictReader(f) if r["split"] == split}


def ocr_sha256(split):
    """SHA-256 of the local OCR file of a split, or of the original run's file when only the shipped counts exist."""
    path = Path(roots()["DATA"]) / "evidence" / split / "ocr.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ORIGINAL_OCR_SHA256[split]


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def atomic_save(obj, path):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as output:
        torch.save(obj, output)
        output.flush(); os.fsync(output.fileno())
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(labels, predictions):
    """Accuracy, Macro-F1, sarcasm-class F1 and the 2x2 confusion matrix (rows = true label)."""
    confusion = np.zeros((2, 2), dtype=np.int64)
    for y, p in zip(labels, predictions):
        confusion[int(y), int(p)] += 1
    f1 = []
    for k in range(2):
        denom = confusion[k].sum() + confusion[:, k].sum()
        f1.append(2 * confusion[k, k] / denom if denom else 0.)
    return dict(rows=len(labels), accuracy=float(np.trace(confusion) / len(labels)),
                macro_f1=float(np.mean(f1)), sarcasm_f1=float(f1[1]), confusion_matrix=confusion.tolist())
