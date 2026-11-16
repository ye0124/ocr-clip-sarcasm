"""Label statistics per split by amount and content of in-image text (paper: error analysis).

Share of sarcastic posts among all posts, posts whose image has OCR text, posts with more than 20 OCR words, and
posts whose OCR text contains the string "sarcas". Uses labels and OCR output only; no model predictions.

  python -m sarcasm.analysis.label_statistics   -> ${OUTPUTS}/analysis/label_statistics.json
Needs the dataset and the OCR files (scripts/download_assets.py, scripts/prepare_data.sh).
"""
import json
from pathlib import Path

import pyarrow as pa

from ..config import roots

SPLITS = {"train": 19816, "validation": 2410, "test": 2409}


def labels(arrow_root, split):
    """{id: label} of one split, read from the MMSD2.0 arrow files."""
    tables = []
    for path in sorted(Path(arrow_root).glob(f"mmsd2.0-{split}*.arrow")):
        with pa.memory_map(str(path), "r") as source:
            tables.append(pa.ipc.open_stream(source).read_all().select(["id", "label"]))
    table = pa.concat_tables(tables)
    assert len(table) == SPLITS[split], (split, len(table))
    return {str(i): int(y) for i, y in zip(table.column("id").to_pylist(), table.column("label").to_pylist())}


def share(rows):
    """Number of posts and percentage of sarcastic posts."""
    return {"n": len(rows), "sarcastic_percent": round(100 * sum(rows) / len(rows), 1) if rows else None}


def main():
    R = roots(); out = {}
    for split in SPLITS:
        y = labels(R["ARROW_ROOT"], split)
        text = {}
        for line in (Path(R["DATA"]) / "evidence" / split / "ocr.jsonl").read_text().splitlines():
            r = json.loads(line); text[str(r["id"])] = " ".join(l["text"] for l in r["lines"])
        assert set(text) == set(y)
        words = {i: len(t.split()) for i, t in text.items()}
        out[split] = {"all": share(list(y.values())),
                      "with_ocr_text": share([y[i] for i in y if words[i] > 0]),
                      "without_ocr_text": share([y[i] for i in y if words[i] == 0]),
                      "ocr_gt20_words": share([y[i] for i in y if words[i] > 20]),
                      "ocr_gt60_words": share([y[i] for i in y if words[i] > 60]),
                      "ocr_contains_sarcas": share([y[i] for i in y if "sarcas" in text[i].lower()])}
    target = Path(R["OUTPUTS"]) / "analysis" / "label_statistics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + "\n")
    for split, groups in out.items():
        print(split, {g: (v["n"], v["sarcastic_percent"]) for g, v in groups.items()})


if __name__ == "__main__":
    main()
