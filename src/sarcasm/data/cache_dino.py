"""Cache DINOv2-small patch tokens (fp16) for every MMSD2.0 row; only the interaction-gated model reads them.

  python -m sarcasm.data.cache_dino --splits train validation test
Writes ${DATA}/dinov2-small/<split>/shard-*.pt (lists of {"dino_hidden", "label", "id"}) and meta.json.
The models mean-pool the 256 patch tokens to one 384-d vector (sarcasm.train.clip_common.load_dino).
Same computation as the original cache script; the dataset is read from the local arrow files.
"""
import argparse, json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

from ..config import roots
from ..train.clip_common import load_split


def main():
    R = roots()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default=str(Path(R["MODELS"]) / "dinov2-small"))
    ap.add_argument("--out-root", type=Path, default=Path(R["DATA"]) / "dinov2-small")
    a = ap.parse_args()
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = Dinov2Model.from_pretrained(a.model).to(d).eval(); model.requires_grad_(False)
    for split in a.splits:
        part = load_split({"arrow_root": R["ARROW_ROOT"]}, split)
        out = a.out_root / split; out.mkdir(parents=True, exist_ok=True)
        for p in out.glob("shard-*.pt"):
            p.unlink()
        items, ids, shard = [], [], 0
        for st in tqdm(range(0, len(part), a.batch_size), desc=split):
            rows = [part[i] for i in range(st, min(st + a.batch_size, len(part)))]
            pix = proc(images=[r["image"].convert("RGB") for r in rows], return_tensors="pt")["pixel_values"].to(d)
            with torch.inference_mode():
                z = model(pixel_values=pix).last_hidden_state[:, 1:].cpu().half()
            items.append({"dino_hidden": z, "label": torch.tensor([int(r["label"]) for r in rows]),
                          "id": [str(r["id"]) for r in rows]})
            ids += [str(r["id"]) for r in rows]
            if len(items) * a.batch_size >= 256 or st + a.batch_size >= len(part):
                torch.save(items, out / f"shard-{shard:04d}.pt"); shard += 1; items = []
        (out / "meta.json").write_text(json.dumps({"split": split, "rows": len(part), "model": "facebook/dinov2-small",
                                                   "shape": [z.shape[1], z.shape[2]], "unique_ids": len(set(ids))}, indent=2))


if __name__ == "__main__":
    main()
