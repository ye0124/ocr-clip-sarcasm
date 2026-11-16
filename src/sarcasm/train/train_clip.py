"""Train / test the CLIP-only models on the official MMSD2.0 split.

Covers every CLIP-only model in the paper: CLIP + small MLP, CLIP + wide MLP (capacity-matched),
the interaction-gated model and its ablations, and the local patch-attention model and its ablations.
The training loop is the one used for all reported runs (moved from ``train_e6_ablation.py``); only the
protocol bookkeeping of the original research code (file locks, freeze checks) was removed.

  python -m sarcasm.train.train_clip --config configs/clip_baselines.json --run wide_mlp_seed42 --stage train
  python -m sarcasm.train.train_clip --config configs/clip_baselines.json --run wide_mlp_seed42 --stage test

Checkpoint selection: best validation Macro-F1 (strictly greater, earliest epoch on ties); early stopping
after 4 epochs without improvement. ``--stage test`` evaluates best.pt once on the test split.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from ..config import default_output, load_config, run_config
from ..models.interaction_gated import ABLATION_ORDER, AblationModel
from ..models.clip_variants import SUPPLEMENT_ORDER, SupplementModel
from ..models.local_patch_attention import VARIANTS as LER_VARIANTS, LERModel
from .clip_common import (Collator, atomic_json, atomic_torch, autocast, configure_runtime, environment,
                          evaluate, load_dino, load_split, load_trainable, make_loader, make_processor,
                          metrics, object_hash, order_for_epoch, restore_rng, rng_state, seed_all,
                          sha256_file, trainable_state)

# Paper names -> constructor variants. "wide_mlp" is the image-text branch of the main model.
MODELS = {
    "small_mlp": (AblationModel, "clip_lora_mlp"),
    "interaction_gated": (AblationModel, "e6_full"),
    "interaction_gated_no_difference": (AblationModel, "e6_no_difference"),
    "interaction_gated_no_product": (AblationModel, "e6_no_product"),
    "interaction_gated_no_both": (AblationModel, "e6_no_both"),
    "interaction_gated_no_dino": (AblationModel, "e6_no_dino"),
    "interaction_gated_no_dino_no_both": (SupplementModel, "e6_no_dino_no_both"),
    "interaction_gated_bilinear_view": (SupplementModel, "e6_full_bilinear_view"),
    "wide_mlp": (LERModel, "B"),
    "local_patch_attention": (LERModel, "LER"),
    "local_patch_attention_no_null": (LERModel, "N"),
    "local_patch_attention_no_reference": (LERModel, "R"),
    "local_patch_attention_no_null_no_reference": (LERModel, "A"),
    "local_patch_attention_uniform": (LERModel, "U"),
    "local_patch_attention_capacity_control": (LERModel, "W"),
}
assert {v for c, v in MODELS.values() if c is AblationModel} <= set(ABLATION_ORDER)
assert {v for c, v in MODELS.values() if c is SupplementModel} <= set(SUPPLEMENT_ORDER)
assert {v for c, v in MODELS.values() if c is LERModel} <= set(LER_VARIANTS)


def log(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def build(cfg):
    seed_all(cfg["seed"])
    cls, variant = MODELS[cfg["model"]]
    model = cls(variant, cfg["clip_path"], cfg["seed"]).cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    return model, opt, scaler


def train_epoch(model, loader, cfg, opt, scaler, epoch, out=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    predictions, labels = [], []
    loss_sum, seen, steps, amp_skips = 0., 0, 0, 0
    micro_count = len(loader)
    accumulation = cfg["effective_batch"] // cfg["micro_batch"]
    assert accumulation == cfg["accumulation"]
    total_rows = len(loader.sampler)
    start = time.monotonic()
    for index, batch in enumerate(loader):
        batch.pop("ids")
        y = batch.pop("labels").cuda(non_blocking=True)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        group_start = (index // accumulation) * cfg["effective_batch"]
        group_size = min(cfg["effective_batch"], total_rows - group_start)
        with autocast(cfg):
            logits = model(batch)
            ce_sum = F.cross_entropy(logits.float(), y, reduction="sum")
            loss = ce_sum / group_size
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss epoch={epoch} micro_batch={index}")
        scaler.scale(loss).backward()
        loss_sum += ce_sum.detach().item()
        seen += len(y)
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.detach().argmax(-1).cpu().tolist())
        if (index + 1) % accumulation == 0 or index + 1 == micro_count:
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
            finite = bool(torch.isfinite(norm))
            old_scale = scaler.get_scale()
            if not finite and not scaler.is_enabled():
                raise RuntimeError(f"Nonfinite gradient epoch={epoch} update={steps}")
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            amp_skips += int(skipped)
            opt.zero_grad(set_to_none=True)
            steps += 1
            if steps % 50 == 0 or skipped:
                event = {"event": "progress", "epoch": epoch, "optimizer_steps": steps, "rows": seen,
                         "ce_loss": loss_sum / seen, "amp_skips": amp_skips,
                         "gradient_norm": float(norm) if finite else "nonfinite",
                         "elapsed_seconds": time.monotonic() - start}
                log(event)
                if out:
                    atomic_json(event, out / "progress.json")
    result = metrics(labels, predictions, loss_sum / seen)
    result.update({"seconds": time.monotonic() - start, "optimizer_steps": steps, "amp_skips": amp_skips,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    return result


def data_for(cfg, model, split, processor, smoke=False):
    part = load_split(cfg, split)
    if smoke:  # 16 rows per class
        ys = part["label"]
        part = part.select([i for lab in (0, 1) for i in [j for j, y in enumerate(ys) if int(y) == lab][:16]])
    dino = load_dino(cfg, split) if model.uses_dino else None
    return part, Collator(processor, dino)


def checkpoint(model, opt, scaler, cfg, epoch, history, best_epoch, best, bad, global_step, elapsed):
    return {"trainable": trainable_state(model), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "global_step": global_step, "history": history, "best_epoch": best_epoch,
            "best_macro_f1": best, "patience_count": bad, "rng": rng_state(), "config": cfg,
            "elapsed_seconds": elapsed}


def write_best(saved, out):
    best = {"trainable": saved["trainable"], "epoch": saved["epoch"],
            "validation": saved["best_pending"]["validation"], "config": saved["config"]}
    atomic_torch(best, out / "best.pt")
    atomic_json(saved["best_pending"]["validation"], out / "validation_metrics.json")
    atomic_json(saved["best_pending"]["predictions"], out / "validation_predictions.json")


def train(cfg, out, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        log({"event": "already_complete", "run": str(out)}); return
    model, opt, scaler = build(cfg)
    config = dict(cfg, smoke=smoke, initialization=model.initialization_hashes())
    if cfg.get("expected_initialization_full"):
        assert config["initialization"]["full"] == cfg["expected_initialization_full"], "initialization differs"
    config["config_hash"] = object_hash(config)
    if (out / "run_config.json").exists():
        previous = json.loads((out / "run_config.json").read_text())
        assert previous["config_hash"] == config["config_hash"], "Configuration changed; use a new output directory"
    else:
        atomic_json(config, out / "run_config.json")
        atomic_json(environment(), out / "environment.json")
        atomic_json(model.parameter_statistics(), out / "parameter_audit.json")
    proc = make_processor(cfg)
    tr, tc = data_for(cfg, model, "train", proc, smoke)
    va, vc = data_for(cfg, model, "validation", proc, smoke)
    history, best, bad, best_epoch, global_step, elapsed = [], -1., 0, None, 0, 0.
    start_epoch = 1
    if (out / "last.pt").exists():  # resume at an epoch boundary
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        assert saved["config"]["config_hash"] == config["config_hash"]
        if "best_pending" in saved:
            write_best(saved, out)
        load_trainable(model, saved["trainable"])
        opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        history, best, bad = saved["history"], saved["best_macro_f1"], saved["patience_count"]
        best_epoch, global_step, elapsed = saved["best_epoch"], saved["global_step"], saved["elapsed_seconds"]
        start_epoch = saved["epoch"] + 1
        restore_rng(saved["rng"])
        log({"event": "resume", "epoch": start_epoch})
    elif (out / "best.pt").exists():
        raise RuntimeError("best.pt without last.pt; inspect the output directory")
    start = time.monotonic()
    epochs = 2 if smoke else cfg["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        if bad >= cfg["patience"]:
            break
        order = order_for_epoch(cfg["seed"], epoch, len(tr))
        torch.cuda.reset_peak_memory_stats()
        trm = train_epoch(model, make_loader(tr, tc, cfg, order), cfg, opt, scaler, epoch, out)
        vm, predictions = evaluate(model, make_loader(va, vc, cfg), cfg)
        global_step += trm["optimizer_steps"]
        improved = vm["macro_f1"] > best
        if improved:
            best, bad, best_epoch = vm["macro_f1"], 0, epoch
        else:
            bad += 1
        row = {"epoch": epoch, "train": trm, "validation": vm, "best_epoch": best_epoch, "patience_count": bad}
        history.append(row)
        current = checkpoint(model, opt, scaler, config, epoch, history, best_epoch, best, bad, global_step,
                             elapsed + time.monotonic() - start)
        # last.pt is written first and owns the epoch boundary; best.pt is repaired from it after a crash.
        if improved:
            current["best_pending"] = {"validation": vm, "predictions": predictions}
        atomic_torch(current, out / "last.pt")
        if improved:
            write_best(current, out)
        atomic_json(history, out / "history.json")
        log(row)
    best_cp = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert best_cp["epoch"] == best_epoch
    result = {"status": "complete", "run": cfg["name"], "model": cfg["model"], "seed": cfg["seed"], "smoke": smoke,
              "best_epoch": best_epoch, "validation": best_cp["validation"], "epochs_completed": len(history),
              "stop_reason": "patience" if bad >= cfg["patience"] else "max_epochs",
              "training_seconds": elapsed + time.monotonic() - start, "best_pt_sha256": sha256_file(out / "best.pt"),
              "peak_allocated_bytes": max(r["train"]["peak_allocated_bytes"] for r in history)}
    atomic_json(result, out / "result.json")
    log(result)


def test(cfg, out):
    assert (out / "result.json").exists(), "train first"
    assert not (out / "test_metrics.json").exists(), "test already evaluated (each run is evaluated once)"
    model, _, _ = build(cfg)
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    load_trainable(model, saved["trainable"])
    part, col = data_for(cfg, model, "test", make_processor(cfg))
    result, predictions = evaluate(model, make_loader(part, col, cfg), cfg)
    result.update({"checkpoint_epoch": saved["epoch"], "seed": cfg["seed"], "model": cfg["model"]})
    atomic_json(predictions, out / "test_predictions.json")
    atomic_json(result, out / "test_metrics.json")
    log(result)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--stage", choices=("train", "test"), required=True)
    p.add_argument("--output", type=Path, help="default: ${OUTPUTS}/<config name>/<run>")
    p.add_argument("--smoke", action="store_true", help="2 epochs on 32 training rows (pipeline check)")
    a = p.parse_args()
    configure_runtime()
    cfg = run_config(load_config(a.config), a.run)
    out = a.output or default_output(a.config, a.run)
    if a.stage == "train":
        train(cfg, out, a.smoke)
    else:
        test(cfg, out)


if __name__ == "__main__":
    sys.exit(main())
