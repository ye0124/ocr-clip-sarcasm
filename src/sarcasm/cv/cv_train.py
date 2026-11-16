"""5-fold CV wrapper around the main-model trainer (sarcasm.train.train_main).

Only the data split changes: FoldSplit draws rows from MMSD2.0 train+validation according to folds.json.
  --stage train         joint model (old main = arm "full", new main = arm "S45S2") via train_joint_v3.train
  --stage train_wide    CLIP + wide MLP alone (= the joint model's image-text branch; settings of the
                        original capacity-matched run: fp32, AdamW 1e-4 / wd 0.01, batch 16, clip 1.0,
                        <=20 epochs, patience 4, best inner-validation Macro-F1, earliest on tie)
  --stage heldout       evaluate best.pt once on the fold's held-out rows (writes heldout_*.json)
The MMSD2.0 test split is never read.

  python -m sarcasm.cv.cv_train --config configs/cv5.json --stage train_wide --run wide_f0
  python -m sarcasm.cv.cv_train --config configs/cv5.json --stage train --run full_f0
  python -m sarcasm.cv.cv_train --config configs/cv5.json --stage heldout --run full_f0
"""
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from ..config import default_output, load_config, run_config
from ..data.ocr_region_dropout import DropoutConfig, OcrRegionDropout
from ..data.text_swap import SwapConfig, TextSwap
from ..models.local_patch_attention import LERModel
from ..train import train_main as T

POOL = {"train": 19816, "validation": 2410}
_CACHE = {}


def pool_data(cfg):
    """MMSD2.0 train+validation rows (arrow) and their OCR, concatenated in that order."""
    key = (cfg["arrow_root"], cfg["evidence_root"])
    if key not in _CACHE:
        from datasets import Dataset, concatenate_datasets
        parts, ocr = [], []
        for split, n in POOL.items():
            files = sorted(Path(cfg["arrow_root"]).glob(f"mmsd2.0-{split}*.arrow"))
            p = [Dataset.from_file(str(f)) for f in files]
            p = concatenate_datasets(p) if len(p) > 1 else p[0]
            assert len(p) == n
            folder = Path(cfg["evidence_root"]) / split
            summary = json.loads((folder / "ocr_summary.json").read_text())
            assert T.sha(folder / "ocr.jsonl") == summary["output_sha256"]
            o = [json.loads(line) for line in (folder / "ocr.jsonl").read_text().splitlines()]
            assert len(o) == n
            parts.append(p); ocr.extend(o)
        _CACHE[key] = (concatenate_datasets(parts), ocr)
    return _CACHE[key]


class FoldSplit(T.Split):
    """T.Split with rows chosen by folds.json; split in {"train", "validation", "heldout"}.

    Sequence construction, masking, text swap and collate are inherited unchanged.
    """

    def __init__(self, cfg, split, tokenizer, limit=None):
        self.part, ocr = pool_data(cfg)
        folds = json.loads(Path(cfg["cv_folds"]).read_text())
        assert T.sha(cfg["cv_folds"]) == cfg["cv_folds_sha256"], "folds changed"
        wanted = folds["folds"][str(cfg["fold"])][split]
        ids = [str(i) for i in self.part["id"]]
        texts = [str(t) for t in self.part["text"]]
        labels = [int(y) for y in self.part["label"]]
        assert len(ocr) == len(ids) and len(set(ids)) == len(ids)
        for o, i in zip(ocr, ids):
            assert o["id"] == i
        row = {i: k for k, i in enumerate(ids)}
        positions = sorted(row[i] for i in wanted)
        assert len(positions) == len(wanted)
        if limit:
            positions = sorted(i for label in (0, 1) for i in [j for j in positions if labels[j] == label][:limit // 2])
        self.positions = positions
        self.ids = [ids[p] for p in positions]
        self.labels = np.array([labels[p] for p in positions])
        self.texts = [texts[p] for p in positions]
        tmark, omark = tokenizer.convert_tokens_to_ids(["[POST]", "[IMAGE_TEXT]"])
        self.sequences, presence = [], []
        self.truncation = dict(post=0, ocr=0)
        for p in positions:
            raw = "\n".join(line["text"] for line in ocr[p]["lines"]) if cfg["use_ocr"] else ""
            t, o = tokenizer.encode(texts[p]), tokenizer.encode(raw)
            self.truncation["post"] += len(t) > 96; self.truncation["ocr"] += len(o) > 154
            t, o = t[:96], o[:154]
            self.sequences.append([1, tmark] + t + [2] + ([omark] + o + [2] if o else []))
            presence.append([bool(raw.strip()), False])
        self.presence = np.asarray(presence, dtype=bool)
        self.processor = None
        self.clip_path = cfg["clip_path"]
        self.lines = [ocr[p]["lines"] for p in positions]
        self.tokenizer, self.marks, self.use_ocr, self.epoch = tokenizer, (tmark, omark), cfg["use_ocr"], 0
        self.swap = self.dropout = None
        if split == "train" and cfg.get("text_swap"):
            pools = json.loads(Path(cfg["bias_pools"]).read_text())
            assert T.sha(cfg["bias_pools"]) == cfg["bias_pools_sha256"], "bias pools changed"
            train_ids = set(wanted)
            for name in ("uniform_pool", "positive_pool", "negative_pool"):
                assert all(str(i) in train_ids for i, _ in pools[name]), "pool row outside inner train"
            self.ocr_all = ocr
            self.id2row = row
            conv = lambda pool: [(str(a), int(b)) for a, b in pool]
            self.swap = TextSwap(SwapConfig(**cfg["text_swap"], seed=cfg["seed"]), conv(pools["uniform_pool"]),
                                 conv(pools["positive_pool"]), conv(pools["negative_pool"]))
        if split == "train" and cfg.get("ocr_dropout"):
            self.dropout = OcrRegionDropout(DropoutConfig(**cfg["ocr_dropout"], seed=cfg["seed"]))


T.Split = FoldSplit  # train_joint_v3.train looks Split up as a module global


def run_cfg(protocol, name, smoke):
    cfg = dict(run_config(protocol, name), smoke=smoke)
    if smoke:
        cfg.update(name=f"smoke_{name}", epochs=2, patience=2, checkpoint_steps=1)
    return cfg


# ---------------------------------------------------------------- wide MLP
def wide_model(cfg):
    return LERModel("B", cfg["clip_path"], cfg["seed"])


@torch.no_grad()
def wide_eval(model, data, cfg):
    model.eval()
    logits = []
    for batch in T.loader(data, list(range(len(data))), cfg["eval_batch"], cfg["num_workers"]):
        clip, _, _, _ = T.to_cuda(batch)
        lb = model(clip); assert torch.isfinite(lb).all(); logits.append(lb.float().cpu())
    lb = torch.cat(logits); prob = lb.softmax(-1); pred = lb.argmax(-1).tolist(); labels = data.labels.tolist()
    metrics = dict(joint=T.score(labels, pred)); metrics["macro_f1"] = metrics["joint"]["macro_f1"]
    rows = [dict(id=data.ids[i], label=labels[i], prediction=pred[i], probabilities=prob[i].tolist(),
                 logits=lb[i].tolist()) for i in range(len(data))]
    return metrics, rows


def train_wide(args, protocol):
    cfg = run_cfg(protocol, args.run, args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output / "run_config.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == cfg
    else:
        T.atomic_json(cfg, cp)
    if (args.output / "result.json").exists():
        print("Run already complete"); return
    T.configure(cfg["seed"])
    tokenizer = T.make_tokenizer(protocol["common"]["encoder_path"])  # only to reuse FoldSplit; not used by the model
    model = wide_model(cfg)
    assert model.initialization_hashes()["full"] == protocol["b_initialization_full"][str(cfg["seed"])]
    model = model.cuda()
    train_data = FoldSplit(cfg, "train", tokenizer, 64 if args.smoke else None)
    val_data = FoldSplit(cfg, "validation", tokenizer, 32 if args.smoke else None)
    assert set(train_data.ids).isdisjoint(val_data.ids)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["wide_lr"], weight_decay=cfg["wide_weight_decay"])
    best, stale, history, epoch = -1., 0, [], 1
    last = args.output / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        assert st["config"] == cfg
        T.load_trainable(model, st["trainable"]); opt.load_state_dict(st["optimizer"])
        epoch, (best, stale, history) = st["epoch"] + 1, st["selection"]
        torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        print(json.dumps(dict(event="resume", next_epoch=epoch)), flush=True)
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    bs = cfg["wide_batch"]
    while epoch <= cfg["wide_epochs"] and stale < cfg["wide_patience"]:
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(cfg["seed"] + 1_000_003 * epoch)).tolist()
        model.train(); train_data.epoch = epoch; loss_sum, n = 0., 0
        for batch in T.loader(train_data, order, bs, cfg["num_workers"]):
            clip, _, _, y = T.to_cuda(batch)
            loss = F.cross_entropy(model(clip).float(), y)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]); opt.step()
            loss_sum += loss.item() * len(y); n += len(y)
        metrics, rows = wide_eval(model, val_data, cfg)
        improved = metrics["macro_f1"] > best
        best, stale = (metrics["macro_f1"], 0) if improved else (best, stale + 1)
        history.append(dict(epoch=epoch, train_ce=loss_sum / n, validation=metrics, improved=improved,
                            seconds=time.time() - started))
        if improved:
            T.atomic_save(dict(config=cfg, trainable=T.trainable_state(model), epoch=epoch, metrics=metrics),
                          args.output / "best.pt")
            T.atomic_json(rows, args.output / "validation_predictions.json")
        T.atomic_save(dict(config=cfg, trainable=T.trainable_state(model), optimizer=opt.state_dict(), epoch=epoch,
                           selection=[best, stale, history], cpu_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state()), last)
        T.atomic_json(history, args.output / "history.json")
        print(json.dumps(history[-1]), flush=True)
        epoch += 1
    sel = max(history, key=lambda r: r["validation"]["macro_f1"])
    result = dict(status="complete", run=cfg["name"], smoke=args.smoke, best_epoch=sel["epoch"],
                  validation=sel["validation"], epochs_completed=len(history),
                  best_pt_sha256=T.sha(args.output / "best.pt"),
                  trainable_parameters=sum(p.numel() for p in params),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, process_seconds=time.time() - started)
    T.atomic_json(result, args.output / "result.json"); print(json.dumps(result), flush=True)


# ---------------------------------------------------------------- held-out evaluation
def heldout(args, protocol):
    pred_path, metric_path = args.output / "heldout_predictions.json", args.output / "heldout_metrics.json"
    assert not pred_path.exists(), "held-out already evaluated (evaluated once)"
    cfg = json.loads((args.output / "run_config.json").read_text())
    result = json.loads((args.output / "result.json").read_text())
    assert result["status"] == "complete" and T.sha(args.output / "best.pt") == result["best_pt_sha256"]
    assert cfg["smoke"] == args.smoke
    T.configure(cfg["seed"])
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    assert best["config"] == cfg
    tokenizer = T.make_tokenizer(protocol["common"]["encoder_path"])
    data = FoldSplit(cfg, "heldout", tokenizer, 64 if args.smoke else None)
    if cfg["arm"] == "wide":
        model = wide_model(cfg); T.load_trainable(model, best["trainable"]); model = model.cuda()
        metrics, rows = wide_eval(model, data, cfg)
    else:
        _, model = T.build(cfg); T.load_trainable(model, best["trainable"]); model = model.cuda()
        metrics, rows = T.evaluate(model, data, cfg)
    metrics.update(run=cfg["name"], fold=cfg["fold"], best_epoch=best["epoch"], n=len(data))
    T.atomic_json(rows, pred_path); T.atomic_json(metrics, metric_path)
    print(json.dumps(dict(run=cfg["name"], heldout_macro_f1=metrics["macro_f1"], n=len(data))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=("train", "train_wide", "heldout"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--output", type=Path, help="default: ${OUTPUTS}/cv5/<run>")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-after-steps", type=int)
    args = ap.parse_args()
    protocol = load_config(args.config)
    args.output = args.output or default_output(args.config, args.run)
    if args.stage == "train":
        T.train(args)  # smoke: T.train finds the run in protocol["runs"] and shortens it
    elif args.stage == "train_wide":
        train_wide(args, protocol)
    else:
        heldout(args, protocol)


if __name__ == "__main__":
    main()
