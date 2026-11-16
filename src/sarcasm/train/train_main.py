"""Train / test the OCR-enhanced CLIP model (main model), its ablations and the text-swap + masking variant.

  python -m sarcasm.train.train_main --config configs/main_model.json --run main_seed42 --stage train
  python -m sarcasm.train.train_main --config configs/main_model.json --run main_seed42 --stage test

Loss = CE(image-text branch) + CE(OCR text branch), each a mean over the effective batch of 16 (8 x 2
accumulation). The image-text branch keeps the baseline optimizer settings (constant LR 1e-4, weight decay
on all trainable tensors, fp32); the OCR branch uses encoder LR 1e-5 / head LR 1e-4 with linear warmup-decay
and no decay on 1-D tensors (fp16 autocast). Gradient norms are clipped per branch (1.0 each). Checkpoint =
strict best validation Macro-F1 of the fused prediction, earliest epoch on ties.

Switches in the run config (absent / default = main model):
  use_ocr = false           ablation "No OCR text" (the OCR branch reads the post only)
  to_image = "zero"         ablation "OCR branch without image"
  to_image = "detach"       ablation "OCR loss does not update LoRA"
  loss = "fused_nll"        ablation "Fusion loss only" (-log of the fused probability)
  lora = false              ablation "No LoRA"
  text_swap / cf_lambda / ocr_dropout   variant "text-swap + masking training": bias-guided OCR text swap
                            (counterfactual copy per selected sample, CE on both branches), symmetric-KL
                            consistency weight, and random OCR-region masking.
Moved from the original ``train_joint_v3.py``; the computation is unchanged. The source-hash checks of the
research protocol were replaced by the initialization-hash check of the image-text branch.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from ..config import default_output, load_config, run_config
from ..data.ocr_region_dropout import DropoutConfig, OcrRegionDropout
from ..data.text_swap import SwapConfig, TextSwap
from ..models.ocr_clip import JointModel, JointModelV2
from ..models.ocr_text_branch import make_tokenizer

SPLITS = {"train": 19816, "validation": 2410, "test": 2409}


def atomic_json(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def atomic_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as output:
        torch.save(obj, output)
        output.flush(); os.fsync(output.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(labels, predictions):
    confusion = np.zeros((2, 2), dtype=np.int64)
    for y, p in zip(labels, predictions):
        confusion[y, p] += 1
    f1 = []
    for k in range(2):
        denom = confusion[k].sum() + confusion[:, k].sum()
        f1.append(2 * confusion[k, k] / denom if denom else 0.)
    return dict(rows=len(labels), accuracy=float(np.trace(confusion) / len(labels)),
                macro_f1=float(np.mean(f1)), sarcasm_f1=float(f1[1]), confusion_matrix=confusion.tolist())


class Split(torch.utils.data.Dataset):
    """Raw MMSD2.0 arrow rows (images for CLIP) joined by row index with PaddleOCR evidence.

    DeBERTa sequences are built exactly as train_ser.CachedData in "joint" mode.
    """

    def __init__(self, cfg, split, tokenizer, limit=None):
        from datasets import Dataset, concatenate_datasets
        files = sorted(Path(cfg["arrow_root"]).glob(f"mmsd2.0-{split}*.arrow"))
        parts = [Dataset.from_file(str(p)) for p in files]
        self.part = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        assert len(self.part) == SPLITS[split]
        folder = Path(cfg["evidence_root"]) / split
        summary = json.loads((folder / "ocr_summary.json").read_text())
        assert sha(folder / "ocr.jsonl") == summary["output_sha256"]
        ocr = [json.loads(line) for line in (folder / "ocr.jsonl").read_text().splitlines()]
        ids = [str(i) for i in self.part["id"]]
        texts = [str(t) for t in self.part["text"]]
        labels = [int(y) for y in self.part["label"]]
        assert len(ocr) == len(ids) and len(set(ids)) == len(ids)
        for k, (o, i) in enumerate(zip(ocr, ids)):
            assert o["index"] == k and o["id"] == i, (split, k)
        positions = list(range(len(ids)))
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
        # v3: raw lines for masking / swapping (independent of use_ocr: pixels still carry the text)
        self.lines = [ocr[p]["lines"] for p in positions]
        self.tokenizer, self.marks, self.use_ocr, self.epoch = tokenizer, (tmark, omark), cfg["use_ocr"], 0
        self.swap = self.dropout = None
        if split == "train" and cfg.get("text_swap"):
            pools = json.loads(Path(cfg["bias_pools"]).read_text())
            assert sha(cfg["bias_pools"]) == cfg["bias_pools_sha256"], "bias pools changed"
            self.ocr_all = ocr
            self.id2row = {i: k for k, i in enumerate(ids)}
            conv = lambda pool: [(str(a), int(b)) for a, b in pool]
            self.swap = TextSwap(SwapConfig(**cfg["text_swap"], seed=cfg["seed"]), conv(pools["uniform_pool"]),
                                 conv(pools["positive_pool"]), conv(pools["negative_pool"]))
        if split == "train" and cfg.get("ocr_dropout"):
            self.dropout = OcrRegionDropout(DropoutConfig(**cfg["ocr_dropout"], seed=cfg["seed"]))

    def ocr_text(self, lines):
        return "\n".join(line["text"] for line in lines) if self.use_ocr else ""

    def build_sequence(self, i, raw):
        """Same construction as __init__ for one sample with a given OCR string."""
        tmark, omark = self.marks
        t, o = self.tokenizer.encode(self.texts[i]), self.tokenizer.encode(raw)
        t, o = t[:96], o[:154]
        return [1, tmark] + t + [2] + ([omark] + o + [2] if o else []), bool(raw.strip())

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return i

    def collate(self, indices):
        if self.processor is None:
            from transformers import CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(self.clip_path, local_files_only=True, use_fast=False)
        images, texts, sequences, presence, labels, cf = [], [], [], [], [], []
        for j, i in enumerate(indices):
            image = self.part[self.positions[i]]["image"].convert("RGB")
            lines, seq, pres = self.lines[i], self.sequences[i], [bool(v) for v in self.presence[i]]
            if self.dropout is not None:
                hidden, rng = self.dropout.choose(len(lines), self.epoch, self.ids[i])
                if any(hidden):
                    image = self.dropout.mask_image(image, lines, hidden, rng)
                    lines = [l for l, h in zip(lines, hidden) if not h]
                    seq, pres[0] = self.build_sequence(i, self.ocr_text(lines))
            images.append(image); texts.append(self.texts[i]); sequences.append(seq)
            presence.append(pres); labels.append(int(self.labels[i]))
            if self.swap is not None:
                plan = self.swap.plan(self.epoch, self.ids[i], int(self.labels[i]))
                if plan is not None:
                    rng, key, k, guided = plan
                    row = self.id2row[key]
                    cimg, clines = self.swap.apply(image, lines, self.part[row]["image"], self.ocr_all[row]["lines"][k], rng)
                    cf.append((j, i, cimg, clines, guided))
        n_orig = len(indices)
        for j, i, cimg, clines, guided in cf:
            seq, has = self.build_sequence(i, self.ocr_text(clines))
            images.append(cimg); texts.append(self.texts[i]); sequences.append(seq)
            presence.append([has, False]); labels.append(labels[j])
        clip = self.processor(text=texts, images=images, return_tensors="pt",
                              padding="max_length", truncation=True, max_length=77)
        width = max(map(len, sequences))
        ids = torch.zeros((len(sequences), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, seq in enumerate(sequences):
            ids[j, :len(seq)] = torch.tensor(seq); mask[j, :len(seq)] = 1
        return dict(clip=dict(input_ids=clip["input_ids"], attention_mask=clip["attention_mask"],
                              pixel_values=clip["pixel_values"]),
                    tokens=dict(input_ids=ids, attention_mask=mask),
                    presence=torch.tensor(np.asarray(presence, dtype=bool)),
                    labels=torch.tensor(np.asarray(labels, dtype=np.int64)), indices=list(indices),
                    n_orig=n_orig, cf_src=torch.tensor([c[0] for c in cf], dtype=torch.long),
                    cf_guided=sum(c[4] for c in cf))


def loader(data, order, batch, workers):
    return torch.utils.data.DataLoader(data, batch_size=batch, sampler=order, shuffle=False,
                                       collate_fn=data.collate, num_workers=workers, pin_memory=True,
                                       generator=torch.Generator().manual_seed(9_173), persistent_workers=False)


def to_cuda(batch):
    return ({k: v.cuda(non_blocking=True) for k, v in batch["clip"].items()},
            {k: v.cuda(non_blocking=True) for k, v in batch["tokens"].items()},
            batch["presence"].cuda(non_blocking=True), batch["labels"].cuda(non_blocking=True))


@torch.no_grad()
def evaluate(model, data, cfg):
    model.eval()
    rows, lb_all, lt_all = [], [], []
    for batch in loader(data, list(range(len(data))), cfg["eval_batch"], cfg["num_workers"]):
        clip, tokens, presence, y = to_cuda(batch)
        lb, lt = model(clip, tokens, presence)
        assert torch.isfinite(lb).all() and torch.isfinite(lt).all()
        lb_all.append(lb.float().cpu()); lt_all.append(lt.float().cpu())
    lb, lt = torch.cat(lb_all), torch.cat(lt_all)
    fused = JointModel.fuse(lb, lt)
    y = torch.tensor(data.labels)
    pred = (fused[:, 1] > 0.5).long().tolist()
    labels = data.labels.tolist()
    result = dict(joint=score(labels, pred), b_branch=score(labels, lb.argmax(-1).tolist()),
                  to_branch=score(labels, lt.argmax(-1).tolist()),
                  ce_b=F.cross_entropy(lb, y).item(), ce_to=F.cross_entropy(lt, y).item())
    result["macro_f1"] = result["joint"]["macro_f1"]
    for i in range(len(data)):
        rows.append(dict(id=data.ids[i], label=labels[i], prediction=pred[i],
                         probabilities=fused[i].tolist(), logits_b=lb[i].tolist(), logits_to=lt[i].tolist()))
    return result, rows


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(state) == names, (names - set(state), set(state) - names)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v)


def configure(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build(cfg):
    tokenizer = make_tokenizer(cfg["encoder_path"])
    model = JointModelV2(cfg["clip_path"], cfg["encoder_path"], len(tokenizer), cfg["seed"],
                         to_image=cfg["to_image"], lora=cfg["lora"])
    return tokenizer, model


def train(args):
    protocol = load_config(args.config)
    cfg = dict(run_config(protocol, args.run), smoke=bool(args.smoke))
    if args.smoke:
        cfg.update(name="smoke", epochs=2, patience=2, checkpoint_steps=1)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "run_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == cfg, "Run config changed; use another output"
    else:
        atomic_json(cfg, config_path)
    if (args.output / "result.json").exists():
        print("Run already complete"); return
    configure(cfg["seed"])
    tokenizer, model = build(cfg)
    init = model.b.initialization_hashes()["full"]
    expected = protocol.get("b_initialization_full", {}).get(str(cfg["seed"]))
    if expected:  # same seed -> identical to the standalone CLIP + wide MLP baseline
        assert init == expected, f"image-text branch initialization differs from the baseline: {init}"
    model = model.cuda()
    train_data = Split(cfg, "train", tokenizer, 64 if args.smoke else None)
    val_data = Split(cfg, "validation", tokenizer, 32 if args.smoke else None)
    assert set(train_data.ids).isdisjoint(val_data.ids)
    b_params = [p for p in model.b.parameters() if p.requires_grad]
    t_params = [p for p in model.t.parameters() if p.requires_grad]
    groups = [dict(params=b_params, lr=cfg["clip_lr"], weight_decay=cfg["clip_weight_decay"])]
    for encoder in (True, False):
        for decay in (True, False):
            params = [p for n, p in model.t.named_parameters() if n.startswith("encoder.") == encoder
                      and (p.ndim > 1) == decay and p.requires_grad]
            if params:
                groups.append(dict(params=params, lr=cfg["encoder_lr"] if encoder else cfg["head_lr"],
                                   weight_decay=cfg["to_weight_decay"] if decay else 0.))
    optimizer = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg["init_scale"])
    epoch, cursor, global_step, epoch_loss, epoch_count, amp_skips = 1, 0, 0, [0., 0.], 0, 0
    cf_stats = [0, 0, 0., 0.]  # counterfactuals, guided, CE sum (both branches), symmetric-KL sum (both branches)
    lam = float(cfg.get("cf_lambda") or 0.)
    if train_data.swap is not None or train_data.dropout is not None:
        assert cfg["loss"] == "branch_sum"
    best, stale, history = -1., 0, []
    last = args.output / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        assert state["config"] == cfg
        load_trainable(model, state["trainable"])
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        epoch, cursor, global_step, epoch_loss, epoch_count, amp_skips, cf_stats = state["position"]
        best, stale, history = state["selection"]
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
        print(json.dumps(dict(event="resume", epoch=epoch, cursor=cursor, step=global_step)), flush=True)
        del state
    elif (args.output / "best.pt").exists():
        raise RuntimeError("best.pt without last.pt; inspect before resuming")
    started = time.time()
    effective, micro = cfg["effective_batch"], cfg["micro_batch"]
    steps_per_epoch = math.ceil(len(train_data) / effective)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(total_steps * cfg["warmup_fraction"]))
    torch.cuda.reset_peak_memory_stats()

    def checkpoint():
        atomic_save(dict(config=cfg, trainable=trainable_state(model), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(),
                         position=[epoch, cursor, global_step, epoch_loss, epoch_count, amp_skips, cf_stats],
                         selection=[best, stale, history], cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state()), last)

    while epoch <= cfg["epochs"] and stale < cfg["patience"]:
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(
            cfg["seed"] + 1_000_003 * epoch)).tolist()
        model.train()
        train_data.epoch = epoch  # augmentation decisions depend only on (seed, epoch, sample id)
        batches = iter(loader(train_data, order[cursor:], micro, cfg["num_workers"]))
        while cursor < len(order):
            size = min(effective, len(order) - cursor)
            factor = (global_step + 1) / warmup if global_step < warmup else \
                max(0., (total_steps - global_step) / (total_steps - warmup))
            optimizer.param_groups[0]["lr"] = base_lrs[0]  # B branch: constant, as standalone B
            for group, lr in zip(optimizer.param_groups[1:], base_lrs[1:]):
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            seen, losses = 0, [0., 0.]
            while seen < size:
                batch = next(batches)
                clip, tokens, presence, y_all = to_cuda(batch)
                n, src = batch["n_orig"], batch["cf_src"].cuda(non_blocking=True)
                lb_all, lt_all = model(clip, tokens, presence)
                lb, lt, y = lb_all[:n], lt_all[:n], y_all[:n]
                ce_b = F.cross_entropy(lb.float(), y, reduction="sum")
                ce_t = F.cross_entropy(lt.float(), y, reduction="sum")
                if cfg["loss"] == "branch_sum":
                    loss = (ce_b + ce_t) / size
                else:
                    fused = JointModel.fuse(lb, lt).gather(1, y[:, None]).squeeze(1)
                    loss = -torch.log(fused.clamp_min(1e-12)).sum() / size
                if len(src):
                    lbc, ltc, yc = lb_all[n:].float(), lt_all[n:].float(), y_all[n:]
                    cf_ce = F.cross_entropy(lbc, yc, reduction="sum") + F.cross_entropy(ltc, yc, reduction="sum")
                    kl = torch.zeros((), device=lbc.device)
                    for orig, counter in ((lb.float()[src], lbc), (lt.float()[src], ltc)):
                        p, q = F.log_softmax(orig, -1), F.log_softmax(counter, -1)
                        kl = kl + 0.5 * ((p.exp() * (p - q)).sum(-1) + (q.exp() * (q - p)).sum(-1)).sum()
                    loss = loss + (cf_ce + lam * kl) / size
                    cf_stats = [cf_stats[0] + len(src), cf_stats[1] + batch["cf_guided"],
                                cf_stats[2] + cf_ce.item(), cf_stats[3] + kl.item()]
                assert torch.isfinite(loss), "Nonfinite training loss"
                scaler.scale(loss).backward()
                losses[0] += ce_b.item(); losses[1] += ce_t.item(); seen += len(y)
            assert seen == size
            scaler.unscale_(optimizer)
            norm_b = torch.nn.utils.clip_grad_norm_(b_params, cfg["grad_clip"])
            norm_t = torch.nn.utils.clip_grad_norm_(t_params, cfg["grad_clip"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < old_scale:
                amp_skips += 1
                assert amp_skips <= cfg["max_amp_skips"], "Too many fp16 overflow skips"
            cursor += size; global_step += 1; epoch_count += size
            epoch_loss = [epoch_loss[0] + losses[0], epoch_loss[1] + losses[1]]
            if global_step % cfg["checkpoint_steps"] == 0 and cursor < len(order):
                checkpoint()
            if global_step % 50 == 0 or cursor == len(order):
                status = dict(status="training", pid=os.getpid(), run=cfg["name"], epoch=epoch, seen=cursor,
                              total=len(train_data), step=global_step, best_val_macro_f1=best, amp_skips=amp_skips,
                              grad_norm_b=float(norm_b), grad_norm_to=float(norm_t), seconds=time.time() - started)
                atomic_json(status, args.output / "progress.json"); print(json.dumps(status), flush=True)
            if args.stop_after_steps and global_step >= args.stop_after_steps and cursor < len(order):
                checkpoint()
                print(json.dumps(dict(event="intentional_stop", step=global_step)), flush=True)
                return
        del batches
        metrics, predictions = evaluate(model, val_data, cfg)
        improved = metrics["macro_f1"] > best
        if improved:
            best, stale = metrics["macro_f1"], 0
        else:
            stale += 1
        record = dict(epoch=epoch, train_ce_b=epoch_loss[0] / epoch_count, train_ce_to=epoch_loss[1] / epoch_count,
                      validation=metrics, improved=improved, step=global_step, amp_skips=amp_skips,
                      counterfactuals=cf_stats[0], guided=cf_stats[1],
                      cf_ce_mean=cf_stats[2] / max(1, cf_stats[0]), cf_sym_kl_mean=cf_stats[3] / max(1, cf_stats[0]))
        history.append(record)
        epoch += 1; cursor = 0; epoch_loss = [0., 0.]; epoch_count = 0; cf_stats = [0, 0, 0., 0.]
        if improved:
            # best first, then last: on a crash between them, last still points before this
            # evaluation, so the rest of the epoch is retrained and best.pt is rewritten.
            atomic_save(dict(config=cfg, trainable=trainable_state(model), epoch=record["epoch"],
                             metrics=metrics), args.output / "best.pt")
            atomic_json(predictions, args.output / "validation_predictions.json")
        atomic_json(history, args.output / "history.json")
        checkpoint()
        print(json.dumps(record), flush=True)
    selected = max(history, key=lambda r: r["validation"]["macro_f1"])
    best_state = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    assert best_state["epoch"] == selected["epoch"]
    result = dict(status="complete", run=cfg["name"], smoke=args.smoke, best_epoch=selected["epoch"],
                  validation=selected["validation"], epochs_completed=len(history), history=history,
                  amp_skips=amp_skips, best_pt_sha256=sha(args.output / "best.pt"),
                  train_truncation=train_data.truncation, validation_truncation=val_data.truncation,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, process_seconds=time.time() - started)
    atomic_json(result, args.output / "result.json"); atomic_json(result, args.output / "progress.json")
    print(json.dumps(result), flush=True)


def test(args):
    pred_path, metric_path = args.output / "test_predictions.json", args.output / "test_metrics.json"
    assert not pred_path.exists() and not metric_path.exists(), "Test output already exists; test is evaluated once"
    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    result = json.loads((args.run_dir / "result.json").read_text())
    assert result["status"] == "complete" and not cfg["smoke"]
    assert sha(args.run_dir / "best.pt") == result["best_pt_sha256"]
    configure(cfg["seed"])
    tokenizer, model = build(cfg)
    best = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["config"] == cfg
    load_trainable(model, best["trainable"])
    model = model.cuda()
    data = Split(cfg, "test", tokenizer)
    metrics, rows = evaluate(model, data, cfg)
    metrics.update(run=cfg["name"], best_epoch=best["epoch"], best_pt_sha256=result["best_pt_sha256"],
                   test_truncation=data.truncation)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(rows, pred_path)
    atomic_json(metrics, metric_path)
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "test"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, help="run directory; default: ${OUTPUTS}/<config name>/<run>")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    args.output = args.output or default_output(args.config, args.run)
    args.run_dir = args.output
    if args.stage == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    sys.exit(main())
