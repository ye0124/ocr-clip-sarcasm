"""PaddleOCR for the exact MMSD2.0 arrow rows; image only, never posts or labels.

Output rows, one JSON object per image:
{"id", "index", "image_sha256", "lines": [{"text", "score", "box_normalized"}], "error"?}
Lines are ordered by the frozen reading-order rule: greedy rows anchored at ascending
normalized vertical centre (top first) within 0.015, then left-to-right inside a row.
Empty recognised strings are dropped; no confidence filter is applied.

Runs in a separate environment (PaddleOCR / PaddlePaddle-GPU, see requirements-ocr.txt):
  python src/sarcasm/data/ocr_paddle.py --arrow-root $(cat data/arrow_root.txt) --output data/evidence
Writes data/evidence/<split>/ocr.jsonl and ocr_summary.json (sha256 of ocr.jsonl, checked by every trainer).
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

SPLITS = {"train": 19816, "validation": 2410, "test": 2409}
ROW_TOLERANCE = 0.015
MAX_FAILED = 20


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reading_order(items):
    """items: (text, score, x_left, y_mid, box) with normalized coordinates."""
    rows, current, anchor = [], [], None
    for item in sorted(items, key=lambda r: (r[3], r[2])):
        if anchor is None or abs(item[3] - anchor) <= ROW_TOLERANCE:
            if anchor is None:
                anchor = item[3]
            current.append(item)
        else:
            rows.append(current)
            current, anchor = [item], item[3]
    if current:
        rows.append(current)
    return [item for row in rows for item in sorted(row, key=lambda r: r[2])]


def load_table(arrow_root, split):
    import pyarrow as pa
    tables = []
    for path in sorted(arrow_root.glob(f"mmsd2.0-{split}*.arrow")):
        with pa.memory_map(str(path), "r") as source:
            tables.append(pa.ipc.open_stream(source).read_all())
    table = pa.concat_tables(tables).select(["id", "image"])
    assert len(table) == SPLITS[split], (split, len(table))
    return table


def recognise(ocr, raw):
    import numpy as np
    from PIL import Image
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    result = ocr.predict(np.ascontiguousarray(np.asarray(image)[:, :, ::-1]))[0]
    items = []
    for text, score, poly in zip(result["rec_texts"], result["rec_scores"], result["rec_polys"]):
        if not str(text).strip():
            continue
        xs = [float(p[0]) / width for p in poly]
        ys = [float(p[1]) / height for p in poly]
        box = [min(xs), min(ys), max(xs), max(ys)]
        items.append((str(text), float(score), min(xs), (min(ys) + max(ys)) / 2, box))
    return [dict(text=t, score=round(s, 6), box_normalized=[round(v, 6) for v in b])
            for t, s, _, _, b in reading_order(items)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR
    config = dict(engine="PaddleOCR", paddleocr=paddleocr.__version__, paddle=paddle.__version__,
                  lang="en", detection="PP-OCRv5_server_det", recognition="en_PP-OCRv5_mobile_rec",
                  use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                  device="gpu:0", image_decode="PIL convert RGB -> BGR array",
                  reading_order=f"greedy rows anchored at ascending normalized midY within {ROW_TOLERANCE}, left-to-right in row",
                  empty_text="dropped", confidence_filter="none", input="image only; no post or label",
                  failure_policy=dict(retries_per_image=1, failure_output="empty lines with explicit error; keep sample",
                                      max_failed_images=MAX_FAILED))
    ocr = PaddleOCR(lang="en", device="gpu:0", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    for split in args.splits:
        folder = args.output / split
        folder.mkdir(parents=True, exist_ok=True)
        summary_path = folder / "ocr_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            assert sha_file(folder / "ocr.jsonl") == summary["output_sha256"]
            print(json.dumps(dict(split=split, status="verified_existing")), flush=True)
            continue
        config_path = folder / "ocr_config.json"
        if config_path.exists():
            assert json.loads(config_path.read_text()) == config, "OCR config changed; use another output"
        else:
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        out_path = folder / "ocr.jsonl"
        done = 0
        if out_path.exists():
            # Drop a possibly torn final line, then resume after the last complete row.
            lines = out_path.read_text().splitlines()
            good = []
            for line in lines:
                try:
                    good.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            for i, row in enumerate(good):
                assert row["index"] == i
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good))
            done = len(good)
        failed = 0
        if done:
            failed = sum(1 for line in out_path.read_text().splitlines() if "error" in json.loads(line))
        table = load_table(args.arrow_root, split)
        n = SPLITS[split]
        started = time.time()
        with out_path.open("a") as output:
            for start in range(done, n, 64):
                batch = table.slice(start, min(64, n - start)).to_pylist()
                for offset, row in enumerate(batch):
                    raw = row["image"].get("bytes") or Path(row["image"]["path"]).read_bytes()
                    record = dict(id=str(row["id"]), index=start + offset, image_sha256=sha_bytes(raw))
                    for attempt in range(2):
                        try:
                            record["lines"] = recognise(ocr, raw)
                            record.pop("error", None)
                            break
                        except Exception as exc:  # recorded, retried once, then kept as empty evidence
                            record["lines"] = []
                            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    if "error" in record:
                        failed += 1
                        assert failed <= MAX_FAILED, f"Too many OCR failures in {split}"
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                total = min(start + 64, n)
                progress = dict(split=split, rows=total, total=n, failed=failed,
                                seconds=round(time.time() - started, 1), updated=time.time())
                (folder / "progress.json").write_text(json.dumps(progress))
                if total % 1024 < 64 or total == n:
                    print(json.dumps(progress), flush=True)
        rows = [json.loads(line) for line in out_path.read_text().splitlines()]
        assert len(rows) == n and [r["index"] for r in rows] == list(range(n))
        assert len({r["id"] for r in rows}) == n
        summary = dict(status="complete", split=split, count=n, failed_images=failed,
                       with_text=sum(1 for r in rows if r["lines"]), output_sha256=sha_file(out_path),
                       config_sha256=sha_file(config_path))
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
