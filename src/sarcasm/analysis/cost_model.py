"""Inference cost of CLIP + wide MLP vs OCR-enhanced CLIP model (validation images only; no labels used, no test).

  python -m sarcasm.analysis.cost_model [RUN_DIR]   -> ${OUTPUTS}/analysis/cost_model.json
RUN_DIR: a trained OCR-enhanced CLIP run (default ${OUTPUTS}/main_model/main_seed42). The wide MLP is timed as the
model's image-text branch, which has the same architecture as the standalone baseline; timing does not depend on
the weights. OCR time is measured separately by cost_ocr.py (PaddleOCR environment).
"""
import json, sys, time, os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from ..config import roots
from ..train.train_main import Split, build, load_trainable, to_cuda

N_IMG, WARM = 256, 5
OUT = Path(roots()["OUTPUTS"]) / "analysis/cost_model.json"; OUT.parent.mkdir(parents=True, exist_ok=True)
run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(roots()["OUTPUTS"]) / "main_model/main_seed42"
cfg = json.loads((run / "run_config.json").read_text())
tokenizer, model = build(cfg)
best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
load_trainable(model, best["trainable"])
model = model.cuda().eval()
data = Split(cfg, "validation", tokenizer)
idx = list(range(N_IMG))

def count(m):
    return dict(total=sum(p.numel() for p in m.parameters()), trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))

res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, n_images=N_IMG, split="validation (images only)",
           params=dict(ocr_enhanced_clip=count(model), image_text_branch_eq_wide_mlp=count(model.b), ocr_text_branch=count(model.t)))

def sync(): torch.cuda.synchronize()

@torch.no_grad()
def bench(bs, which):
    batches = [idx[i:i + bs] for i in range(0, N_IMG, bs)]
    pre, fwd = [], []
    torch.cuda.reset_peak_memory_stats()
    for k, b in enumerate(batches):
        t0 = time.perf_counter()
        if which == "wide":
            imgs = [data.part[data.positions[i]]["image"].convert("RGB") for i in b]
            if data.processor is None: data.collate(b[:1])
            c = data.processor(text=[data.texts[i] for i in b], images=imgs, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=77)
            clip = {k2: c[k2].cuda() for k2 in ("input_ids", "attention_mask", "pixel_values")}
        else:
            batch = data.collate(b); clip, tokens, presence, _ = to_cuda(batch)
        sync(); t1 = time.perf_counter()
        if which == "wide":
            model.b(clip)
        else:
            lb, lt = model(clip, tokens, presence); model.fuse(lb, lt)
        sync(); t2 = time.perf_counter()
        if k >= WARM // max(1, bs // 4) or bs == 1 and k >= WARM:
            pre.append(t1 - t0); fwd.append(t2 - t1)
    n = len(pre) * bs
    return dict(batch=bs, measured_images=n,
                preprocess_ms_per_image=round(1000 * sum(pre) / n, 2),
                forward_ms_per_image=round(1000 * sum(fwd) / n, 2),
                forward_ms_per_batch_median=round(1000 * float(np.median(fwd)), 2),
                peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1))

res["timing"] = {f"{w}_bs{bs}": bench(bs, w) for w in ("wide", "ocr_enhanced_clip") for bs in (1, 16)}
OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
