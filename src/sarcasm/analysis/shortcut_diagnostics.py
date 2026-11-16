"""Shortcut diagnostics on VALIDATION for trained runs (never reads test).

Per run (best.pt):
  clean     : standard validation inference (must equal the run's saved validation predictions)
  mask_all  : every OCR line hidden in pixels (ring-median fill) and removed from the OCR text
  inject    : one line inserted into every validation image (never covering existing text):
              negatives get a line from the validation 'probe_positive_pool' (most sarcasm-indicative
              validation lines under the all-train OCR-only Naive Bayes), positives get a line from the
              'probe_negative_pool'. Reported: FPR on negatives and FNR on positives, clean vs injected.
Model-independent (run once): paste-artifact probe. Frozen pretrained CLIP image features of the SAME
donor lines cropped (a) from their source image and (b) after pasting; 5-fold logistic regression AUC.
AUC near 0.5 = pasted text is not distinguishable from native text by CLIP features.

  python -m sarcasm.analysis.shortcut_diagnostics --config configs/text_swap_variant.json \
      --run-dir outputs/main_model/main_seed42 --output outputs/analysis/diag_main_seed42.json
The config supplies bias_pools (the probe pools); --run-dir may point to any trained OCR-enhanced CLIP run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from ..config import load_config
from ..data.ocr_region_dropout import DropoutConfig, OcrRegionDropout
from ..data.text_swap import SwapConfig, TextSwap, _padded
from ..models.ocr_clip import JointModel
from ..train import train_main as T

DEFAULTS = dict(use_ocr=True, to_image="shared", lora=True, loss="branch_sum")
PROBE_SEED = 2026


def macro(labels, pred):
    return T.score(labels, pred)


@torch.no_grad()
def infer(model, data, cfg, use_cf):
    model.eval()
    n_total = len(data)
    out = np.zeros((n_total, 2))
    covered = np.zeros(n_total, dtype=bool)
    for batch in T.loader(data, list(range(n_total)), cfg["eval_batch"], cfg["num_workers"]):
        clip, tokens, presence, _ = T.to_cuda(batch)
        lb, lt = model(clip, tokens, presence)
        fused = JointModel.fuse(lb, lt).cpu().numpy()
        idx = np.asarray(batch["indices"])
        n = batch["n_orig"]
        if use_cf:
            src = batch["cf_src"].numpy()
            out[idx[src]] = fused[n:]; covered[idx[src]] = True
            missing = np.setdiff1d(np.arange(n), src)
            out[idx[missing]] = fused[missing]
        else:
            out[idx] = fused[:n]; covered[idx] = True
    return out, covered


def run_diag(run_dir, cfg_overrides, protocol, pools, out_path):
    run_cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg = dict(DEFAULTS, **run_cfg, **cfg_overrides)
    for key in ("text_swap", "ocr_dropout"):
        cfg.pop(key, None)
    T.configure(cfg["seed"])
    tokenizer, model = T.build(cfg)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    T.load_trainable(model, best["trainable"])
    model = model.cuda()
    data = T.Split(cfg, "validation", tokenizer)
    labels = data.labels
    saved = {r["id"]: r for r in json.loads((run_dir / "validation_predictions.json").read_text())}
    # clean
    clean, _ = infer(model, data, cfg, False)
    diff = max(abs(clean[i, 1] - saved[data.ids[i]]["probabilities"][1]) for i in range(len(data)))
    # mask_all
    data.dropout = OcrRegionDropout(DropoutConfig(p_sample=1.0, p_all=1.0, p_blur=0.0, seed=PROBE_SEED))
    masked, _ = infer(model, data, cfg, False)
    data.dropout = None
    # inject
    ocr = [json.loads(l) for l in (Path(cfg["evidence_root"]) / "validation" / "ocr.jsonl").read_text().splitlines()]
    data.ocr_all = ocr
    data.id2row = {str(r["id"]): k for k, r in enumerate(ocr)}
    conv = lambda p: [(str(a), int(b)) for a, b in p]
    data.swap = TextSwap(SwapConfig(p_cf=1.0, p_guided=1.0, p_replace=0.0, seed=PROBE_SEED),
                         conv(pools["probe_positive_pool"]), conv(pools["probe_positive_pool"]),
                         conv(pools["probe_negative_pool"]))
    injected, covered = infer(model, data, cfg, True)
    data.swap = None
    pc, pm, pi = (clean[:, 1] > 0.5).astype(int), (masked[:, 1] > 0.5).astype(int), (injected[:, 1] > 0.5).astype(int)
    neg, pos = labels == 0, labels == 1
    has_ocr = data.presence[:, 0]
    result = dict(
        run=str(run_dir), clean_vs_saved_max_abs_diff=float(diff),
        clean=macro(labels.tolist(), pc.tolist()),
        mask_all=dict(metrics=macro(labels.tolist(), pm.tolist()),
                      flip_rate_with_ocr=float((pc != pm)[has_ocr].mean()),
                      mean_abs_dp_with_ocr=float(np.abs(clean[:, 1] - masked[:, 1])[has_ocr].mean())),
        inject=dict(covered=int(covered.sum()), metrics=macro(labels.tolist(), pi.tolist()),
                    fpr_clean=float(pc[neg].mean()), fpr_injected=float(pi[neg].mean()),
                    fnr_clean=float(1 - pc[pos].mean()), fnr_injected=float(1 - pi[pos].mean()),
                    mean_dp_neg=float((injected[:, 1] - clean[:, 1])[neg].mean()),
                    mean_dp_pos=float((injected[:, 1] - clean[:, 1])[pos].mean())))
    T.atomic_json(result, out_path)
    print(json.dumps(result), flush=True)


def auc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def logistic_cv_auc(X, y, folds=5, l2=1.0, steps=2000, lr=0.5):
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y)); parts = np.array_split(idx, folds)
    scores = np.zeros(len(y))
    for f in parts:
        tr = np.setdiff1d(idx, f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X[tr] - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            w -= lr * (Z.T @ (p - y[tr]) / len(tr) + l2 * w / len(tr)); b -= lr * (p - y[tr]).mean()
        scores[f] = ((X[f] - mu) / sd) @ w + b
    return auc(scores, y)


@torch.no_grad()
def artifact_probe(protocol, pools, out_path, n=600):
    from transformers import CLIPModel, CLIPProcessor
    cfg = protocol["common"]
    data = T.Split(dict(cfg, seed=0), "validation", T.make_tokenizer(cfg["encoder_path"]))
    ocr = [json.loads(l) for l in (Path(cfg["evidence_root"]) / "validation" / "ocr.jsonl").read_text().splitlines()]
    id2row = {str(r["id"]): k for k, r in enumerate(ocr)}
    pool = [(str(a), int(b)) for a, b in pools["probe_positive_pool"] + pools["probe_negative_pool"]]
    sw = TextSwap(SwapConfig(p_cf=1.0, p_guided=0.0, p_replace=0.0, seed=PROBE_SEED), pool, pool, pool)
    c = sw.cfg
    native, pasted = [], []
    rng_targets = np.random.RandomState(PROBE_SEED)
    targets = rng_targets.permutation(len(ocr))
    for t in targets:
        if len(native) >= n:
            break
        tid = str(ocr[t]["id"])
        plan = sw.plan(0, tid, 0)
        if plan is None:
            continue
        rng, key, k, _ = plan
        drow = id2row[key]
        donor_img = data.part[drow]["image"].convert("RGB")
        dline = ocr[drow]["lines"][k]
        dW, dH = donor_img.size
        native.append(donor_img.crop(_padded(dline["box_normalized"], dW, dH, c.crop_pad_x, c.crop_pad_y)))
        timg = data.part[t]["image"].convert("RGB")
        cimg, cl = sw.apply(timg, ocr[t]["lines"], donor_img, dline, rng)
        ins = [l for l in cl if l.get("donor")][0]
        W, H = cimg.size
        pasted.append(cimg.crop(_padded(ins["box_normalized"], W, H, c.crop_pad_x, c.crop_pad_y)))
    model = CLIPModel.from_pretrained(cfg["clip_path"], local_files_only=True).cuda().eval()
    proc = CLIPProcessor.from_pretrained(cfg["clip_path"], local_files_only=True)
    feats = []
    for group in (native, pasted):
        f = []
        for s in range(0, len(group), 64):
            px = proc(images=group[s:s + 64], return_tensors="pt")["pixel_values"].cuda()
            emb = model.get_image_features(pixel_values=px)
            if not torch.is_tensor(emb):  # transformers >= 5 returns BaseModelOutputWithPooling
                emb = emb.pooler_output   # = visual_projection(pooled), the 512-d image embedding
            f.append(emb.float().cpu().numpy())
        feats.append(np.concatenate(f))
    X = np.concatenate(feats); y = np.r_[np.zeros(len(native)), np.ones(len(pasted))]
    result = dict(n_native=len(native), n_pasted=len(pasted), cv_auc=logistic_cv_auc(X, y),
                  note="same donor lines; native crop vs pasted crop; frozen pretrained CLIP ViT-B/32 image features")
    T.atomic_json(result, out_path)
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--artifact", action="store_true")
    a = ap.parse_args()
    protocol = load_config(a.config)
    pools = json.loads(Path(protocol["common"]["bias_pools"]).read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.artifact:
        artifact_probe(protocol, pools, a.output)
    else:
        run_diag(a.run_dir, {}, protocol, pools, a.output)


if __name__ == "__main__":
    main()
