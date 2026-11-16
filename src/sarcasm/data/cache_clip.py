"""Cache frozen (pretrained, no LoRA) CLIP image and text embeddings for every MMSD2.0 row.

Input of the standalone OCR text branch (its image vector). Run once:
  python -m sarcasm.data.cache_clip --arrow-root $(cat data/arrow_root.txt) --clip models/clip-vit-base-patch32 \
      --output data/evidence/clip --splits train validation test
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SPLITS = {"train": 19816, "validation": 2410, "test": 2409}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrow-root", type=Path, required=True)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", required=True)
    args = p.parse_args()
    import numpy as np
    import pyarrow as pa
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    processor = CLIPProcessor.from_pretrained(str(args.clip), local_files_only=True)
    model = CLIPModel.from_pretrained(str(args.clip), local_files_only=True).cuda().eval()
    config = dict(clip="openai/clip-vit-base-patch32@3d74acf9", precision="float32", normalization="L2", batch_size=32,
                  text_length=77, source="original pretrained CLIP; no LoRA or task training",
                  versions=dict(torch=torch.__version__, numpy=np.__version__))
    config_path = args.output/"config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "CLIP cache config differs from original"
    else:
        config_path.write_text(json.dumps(config, indent=2))
    for split in args.splits:
        n = SPLITS[split]
        complete = args.output/f"{split}_manifest.json"
        if complete.exists():
            manifest = json.loads(complete.read_text())
            for name, digest in manifest["file_hashes"].items():
                assert file_hash(args.output/name) == digest
            print(json.dumps(dict(split=split, status="verified_existing")), flush=True)
            continue
        partial_path = args.output/f"{split}_partial.json"
        tpath, ipath = args.output/f"{split}_text.npy", args.output/f"{split}_image.npy"
        records = []
        if partial_path.exists():
            records = json.loads(partial_path.read_text())["records"]
            text_vectors = np.lib.format.open_memmap(tpath, mode="r+")
            image_vectors = np.lib.format.open_memmap(ipath, mode="r+")
        else:
            text_vectors = np.lib.format.open_memmap(tpath, mode="w+", dtype="float32", shape=(n,512))
            image_vectors = np.lib.format.open_memmap(ipath, mode="w+", dtype="float32", shape=(n,512))
        tables = []
        for path in sorted(args.arrow_root.glob(f"mmsd2.0-{split}*.arrow")):
            with pa.memory_map(str(path),"r") as source:
                tables.append(pa.ipc.open_stream(source).read_all())
        table = pa.concat_tables(tables)
        assert len(table) == n
        start_time = time.time()
        for start in range(len(records), n, 32):
            batch = table.slice(start, min(32,n-start)).to_pylist()
            images, metadata = [], []
            for offset, row in enumerate(batch):
                raw = row["image"].get("bytes") or Path(row["image"]["path"]).read_bytes()
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                metadata.append(dict(index=start+offset, id=str(row["id"]), label=int(row["label"]),
                                     text=str(row["text"]), image_sha256=hashlib.sha256(raw).hexdigest()))
            x = processor(images=images, text=[r["text"] for r in metadata], return_tensors="pt",
                          padding="max_length", truncation=True, max_length=77).to("cuda")
            with torch.inference_mode():
                v = model.vision_model(pixel_values=x["pixel_values"]).pooler_output
                t = model.text_model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).pooler_output
                v = torch.nn.functional.normalize(model.visual_projection(v), dim=-1)
                t = torch.nn.functional.normalize(model.text_projection(t), dim=-1)
            image_vectors[start:start+len(batch)] = v.cpu().numpy()
            text_vectors[start:start+len(batch)] = t.cpu().numpy()
            records.extend(metadata)
            image_vectors.flush(); text_vectors.flush()
            tmp = partial_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(records=records)))
            tmp.replace(partial_path)
            if len(records)%512 == 0 or len(records)==n:
                print(json.dumps(dict(split=split, rows=len(records), total=n, seconds=time.time()-start_time)), flush=True)
        assert len({r["id"] for r in records}) == n
        assert np.isfinite(image_vectors).all() and np.isfinite(text_vectors).all()
        assert np.allclose(np.linalg.norm(image_vectors, axis=1),1,atol=1e-5)
        rpath=args.output/f"{split}_records.json"
        rpath.write_text(json.dumps(records, ensure_ascii=False))
        manifest=dict(split=split,rows=n,file_hashes={f.name:file_hash(f) for f in (rpath,tpath,ipath)},config=config)
        complete.write_text(json.dumps(manifest,indent=2))
        print(json.dumps(dict(split=split,status="complete",seconds=time.time()-start_time)),flush=True)


if __name__ == "__main__":
    main()
