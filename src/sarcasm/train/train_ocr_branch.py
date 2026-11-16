"""Train / test the standalone OCR text branch on the official MMSD2.0 split.

Input: DeBERTa-v3-small over "[POST] post [IMAGE_TEXT] OCR text" plus a frozen, pre-computed CLIP image vector
(``scripts/cache_clip.sh``). This branch is half of the separately trained fusion model
(``sarcasm.evaluation.fuse``). Training loop moved unchanged from the original ``train_ser.py``.

  python -m sarcasm.train.train_ocr_branch --config configs/ocr_text_branch.json --run ocr_branch_seed42 --stage train
  python -m sarcasm.train.train_ocr_branch --config configs/ocr_text_branch.json --run ocr_branch_seed42 --stage test
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from ..config import default_output, load_config, run_config
from ..models.ocr_text_branch import SERModel, make_tokenizer
from ..utils import atomic_json, atomic_save, score, sha256_file as sha

SPLITS = {"train": 19816, "validation": 2410, "test": 2409}


class CachedData:
    """Rows of one split: frozen CLIP image vectors (evidence/clip) + PaddleOCR lines (evidence/<split>)."""

    def __init__(self, root, split, tokenizer, mode, evidence, limit=None):
        clip = root / "clip"
        manifest = json.loads((clip / f"{split}_manifest.json").read_text())
        for name, digest in manifest["file_hashes"].items():
            assert sha(clip / name) == digest, f"Changed CLIP cache: {name}"
        records = json.loads((clip / f"{split}_records.json").read_text())
        assert len(records) == SPLITS[split]
        vectors = np.load(clip / f"{split}_image.npy", mmap_mode="r")
        ocr = {}
        if evidence == "ocr":
            folder = root / split
            ocr_summary = json.loads((folder / "ocr_summary.json").read_text())
            assert sha(folder / "ocr.jsonl") == ocr_summary["output_sha256"]
            for line in (folder / "ocr.jsonl").read_text().splitlines():
                row = json.loads(line)
                assert row["id"] not in ocr
                ocr[row["id"]] = row
            assert len(ocr) == len(records)
        if limit:
            selected = sorted([i for label in (0, 1) for i in
                               [j for j, r in enumerate(records) if r["label"] == label][:limit // 2]])
            self.records = [records[i] for i in selected]
            self.vectors = np.asarray(vectors[selected])
        else:
            self.records = records
            self.vectors = vectors
        self.ids = [r["id"] for r in self.records]
        self.labels = np.array([r["label"] for r in self.records])
        self.views = []; self.presence = []
        self.truncation = dict(post=0, ocr=0)
        tmark, omark = tokenizer.convert_tokens_to_ids(["[POST]", "[IMAGE_TEXT]"])
        for r in self.records:
            t = tokenizer.encode(r["text"])
            raw = ""
            if evidence == "ocr":
                item = ocr[r["id"]]
                assert item["image_sha256"] == r["image_sha256"] and item["index"] == r["index"]
                raw = "\n".join(line["text"] for line in item["lines"])
            o = tokenizer.encode(raw)
            self.truncation["post"] += len(t) > 96; self.truncation["ocr"] += len(o) > 154
            t, o = t[:96], o[:154]
            post = [1, tmark] + t + [2]
            if mode == "base": v = [post]
            elif mode == "joint": v = [post + ([omark] + o + [2] if o else [])]
            else: raise ValueError(mode)
            self.views.append(v)
            self.presence.append([bool(raw.strip()), False])
        self.presence = np.asarray(self.presence, dtype=bool)

    def __len__(self): return len(self.records)

    def batch(self, indices):
        sequences = [self.views[i][view] for view in range(len(self.views[0])) for i in indices]
        width = max(map(len, sequences))
        ids = torch.zeros((len(sequences), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, seq in enumerate(sequences):
            ids[j, :len(seq)] = torch.tensor(seq); mask[j, :len(seq)] = 1
        return (dict(input_ids=ids.cuda(), attention_mask=mask.cuda()),
                torch.tensor(np.asarray(self.vectors[indices]), device="cuda"),
                torch.tensor(self.presence[indices], device="cuda"),
                torch.tensor(self.labels[indices], device="cuda"))


@torch.no_grad()
def evaluate(model, data, batch_size):
    model.eval(); preds = []; rows = []; loss_sum = 0.
    for start in range(0, len(data), batch_size):
        indices = list(range(start, min(start + batch_size, len(data))))
        tokens, images, presence, y = data.batch(indices)
        with torch.autocast("cuda", dtype=torch.float16): logits = model(tokens, images, presence)
        assert torch.isfinite(logits).all()
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        p = logits.argmax(-1).cpu().tolist(); preds.extend(p)
        rows.extend(dict(id=data.ids[i], label=int(data.labels[i]), prediction=pred, logits=logit)
                    for i, pred, logit in zip(indices, p, logits.float().cpu().tolist()))
    result = score(data.labels, preds); result["ce_loss"] = loss_sum / len(data)
    return result, rows


def configure(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(cfg, output, data_cls=CachedData):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == cfg, "Configuration changed; use a new output directory"
    else:
        atomic_json(cfg, config_path)
    if (output / "result.json").exists():
        print("Run already complete"); return
    configure(cfg["seed"])
    tokenizer = make_tokenizer(cfg["encoder_path"])
    root = Path(cfg["evidence_root"])
    train_data = data_cls(root, "train", tokenizer, cfg["mode"], cfg["evidence"], 32 if cfg["smoke"] else None)
    val = data_cls(root, "validation", tokenizer, cfg["mode"], cfg["evidence"], 16 if cfg["smoke"] else None)
    assert set(train_data.ids).isdisjoint(val.ids)
    model = SERModel(cfg["encoder_path"], len(tokenizer), mode=cfg["mode"], seed=cfg["seed"],
                     gradient_checkpointing=cfg["gradient_checkpointing"]).cuda()
    groups = []
    for encoder in [True, False]:
        for decay in [True, False]:
            params = [p for n, p in model.named_parameters() if n.startswith("encoder.") == encoder
                      and (p.ndim > 1) == decay and p.requires_grad]
            if params: groups.append(dict(params=params, lr=cfg["encoder_lr"] if encoder else cfg["head_lr"],
                                          weight_decay=cfg["weight_decay"] if decay else 0.))
    optimizer = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=512)
    epoch, cursor, global_step, epoch_loss, epoch_count = 1, 0, 0, 0., 0
    best, stale, history = -1., 0, []
    last = output / "last.pt"
    if last.exists():  # resume mid-epoch
        state = torch.load(last, map_location="cpu", weights_only=True)
        assert state["config"] == cfg
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch, cursor, global_step, epoch_loss, epoch_count = state["position"]
        best, stale, history = state["selection"]
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    started = time.time()
    effective = cfg["effective_batch"]; micro = cfg["micro_batch"]
    total_steps = math.ceil(len(train_data) / effective) * cfg["epochs"]
    warmup = max(1, int(total_steps * cfg["warmup_fraction"]))
    torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        atomic_save(dict(config=cfg, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), position=[epoch, cursor, global_step, epoch_loss, epoch_count],
                         selection=[best, stale, history], cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state()), last)

    while epoch <= cfg["epochs"] and stale < cfg["patience"]:
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(cfg["seed"] + 1_000_003 * epoch)).tolist()
        model.train()
        while cursor < len(order):
            indices = order[cursor:cursor + effective]
            # linear warmup over the first 10% of steps, then linear decay
            factor = (global_step + 1) / warmup if global_step < warmup else max(0., (total_steps - global_step) / (total_steps - warmup))
            for group, lr in zip(optimizer.param_groups, base_lrs): group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for offset in range(0, len(indices), micro):
                tokens, images, presence, y = train_data.batch(indices[offset:offset + micro])
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(tokens, images, presence)
                    loss = F.cross_entropy(logits.float(), y, reduction="sum") / len(indices)
                assert torch.isfinite(loss), "Nonfinite training loss"
                scaler.scale(loss).backward(); loss_sum += loss.item() * len(indices)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            assert torch.isfinite(norm), "Nonfinite gradients"
            scaler.step(optimizer); scaler.update()
            cursor += len(indices); global_step += 1; epoch_loss += loss_sum; epoch_count += len(indices)
            if global_step % cfg["checkpoint_steps"] == 0: checkpoint()
            if global_step % 50 == 0 or cursor == len(order):
                status = dict(status="training", run=cfg["name"], epoch=epoch, seen=cursor, total=len(train_data),
                              step=global_step, best_val_macro_f1=best, seconds=time.time() - started)
                atomic_json(status, output / "progress.json"); print(json.dumps(status), flush=True)
        metrics, predictions = evaluate(model, val, cfg["eval_batch"])
        improved = metrics["macro_f1"] > best
        if improved:
            best = metrics["macro_f1"]; stale = 0
            atomic_save(dict(config=cfg, model=model.state_dict(), epoch=epoch, metrics=metrics), output / "best.pt")
            atomic_json(predictions, output / "validation_predictions.json")
        else: stale += 1
        record = dict(epoch=epoch, train_loss=epoch_loss / epoch_count, validation=metrics, improved=improved, step=global_step)
        history.append(record); atomic_json(history, output / "history.json")
        print(json.dumps(record), flush=True)
        epoch += 1; cursor = 0; epoch_loss = 0.; epoch_count = 0
        checkpoint()
    selected = max(history, key=lambda r: r["validation"]["macro_f1"])
    result = dict(status="complete", run=cfg["name"], smoke=cfg["smoke"], best_epoch=selected["epoch"],
                  validation=selected["validation"], epochs_completed=len(history), history=history,
                  train_truncation=train_data.truncation, validation_truncation=val.truncation,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, process_seconds=time.time() - started)
    atomic_json(result, output / "result.json"); atomic_json(result, output / "progress.json")
    print(json.dumps(result), flush=True)


def evaluate_split(output, split, data_cls=CachedData, limit=None):
    """Evaluate best.pt once on ``split`` (test, or the CV held-out part); write <split>_*.json."""
    pred_path, metric_path = output / f"{split}_predictions.json", output / f"{split}_metrics.json"
    assert not pred_path.exists(), f"{split} already evaluated (each run is evaluated once)"
    cfg = json.loads((output / "run_config.json").read_text())
    assert json.loads((output / "result.json").read_text())["status"] == "complete"
    configure(cfg["seed"])
    tokenizer = make_tokenizer(cfg["encoder_path"])
    data = data_cls(Path(cfg["evidence_root"]), split, tokenizer, cfg["mode"], cfg["evidence"], limit)
    model = SERModel(cfg["encoder_path"], len(tokenizer), mode=cfg["mode"], seed=cfg["seed"],
                     gradient_checkpointing=False).cuda()
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert best["config"] == cfg
    model.load_state_dict(best["model"])
    metrics, rows = evaluate(model, data, cfg["eval_batch"])
    metrics.update(run=cfg["name"], best_epoch=best["epoch"], n=len(data), truncation=data.truncation)
    atomic_json(rows, pred_path); atomic_json(metrics, metric_path)
    print(json.dumps(metrics), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stage", choices=("train", "test"), required=True)
    p.add_argument("--output", type=Path, help="default: ${OUTPUTS}/<config name>/<run>")
    p.add_argument("--smoke", action="store_true", help="1 epoch on 32 training rows (pipeline check)")
    a = p.parse_args()
    cfg = dict(run_config(load_config(a.config), a.run), smoke=a.smoke)
    if a.smoke:
        cfg.update(epochs=1, patience=1, checkpoint_steps=1)
    out = a.output or default_output(a.config, a.run)
    if a.stage == "train":
        train(cfg, out)
    else:
        evaluate_split(out, "test")


if __name__ == "__main__":
    main()
