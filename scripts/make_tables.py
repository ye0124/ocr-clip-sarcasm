"""Build the paper's result tables (results/tables.md) from per-run metrics.

  python scripts/make_tables.py                  # from the shipped results/ (original runs)
  python scripts/make_tables.py --from-outputs   # from your own runs in ${OUTPUTS}; also writes
                                                 # ${OUTPUTS}/official_split_runs.csv to compare with results/
Means and sample standard deviations over seeds 42-44 unless stated; validation selects checkpoints, test is
reported once per run.
"""
import argparse, csv, json, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sarcasm.config import roots  # noqa: E402

CONFIGS = ["clip_baselines", "ocr_text_branch", "main_model", "text_swap_variant"]


def from_outputs(out):
    rows = []
    for cfg in CONFIGS:
        spec = json.loads((REPO / "configs" / f"{cfg}.json").read_text())
        for r in spec["runs"]:
            d = out / cfg / r["name"]
            if not (d / "result.json").exists():
                continue
            res = json.loads((d / "result.json").read_text())
            val = res["validation"].get("joint", res["validation"])
            t = json.loads((d / "test_metrics.json").read_text()) if (d / "test_metrics.json").exists() else None
            t = t.get("joint", t) if t else None
            model = r.get("model") or r["name"].rsplit("_seed", 1)[0]
            if cfg == "ocr_text_branch":
                model = "ocr_text_branch" if r["encoder_lr"] == 1e-05 else "ocr_text_branch_lr2e-05"
            rows.append(dict(config=cfg, run=r["name"], model=model, seed=r["seed"], val_macro_f1=round(100 * val["macro_f1"], 2),
                             test_accuracy=round(100 * t["accuracy"], 2) if t else "", test_macro_f1=round(100 * t["macro_f1"], 2) if t else "",
                             test_sarcasm_f1=round(100 * t["sarcasm_f1"], 2) if t else ""))
    fus = out / "fusion/fusion_test.json"
    if fus.exists():
        for s, v in json.loads(fus.read_text())["seeds"].items():
            rows.append(dict(config="fusion", run=f"fusion_seed{s}", model="separately_trained_fusion", seed=int(s), val_macro_f1="",
                             test_accuracy=round(100 * v["fusion"]["accuracy"], 2), test_macro_f1=round(100 * v["fusion"]["macro_f1"], 2),
                             test_sarcasm_f1=round(100 * v["fusion"]["sarcasm_f1"], 2)))
    return rows


def ms(values):
    values = [float(v) for v in values if v not in ("", None)]
    if not values:
        return "—"
    return f"{st.mean(values):.2f} ± {st.stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def table(rows, models, seeds=(42, 43, 44)):
    lines = ["| Model | n | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test sarcasm-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for label, (cfg, model) in models.items():
        sel = [r for r in rows if r["config"] == cfg and r["model"] == model and int(r["seed"]) in seeds]
        if sel:
            lines.append(f"| {label} | {len(sel)} | {ms(r['val_macro_f1'] for r in sel)} | {ms(r['test_accuracy'] for r in sel)} | "
                         f"{ms(r['test_macro_f1'] for r in sel)} | {ms(r['test_sarcasm_f1'] for r in sel)} |")
    return "\n".join(lines)


CV_ARMS = ("Arms: `wide` CLIP + wide MLP; `full` OCR-enhanced CLIP model (main); `S45S2` text-swap + masking variant; "
           "`noocr` main model without OCR text; `to` / `tn` standalone text branch with / without OCR text; "
           "`sep` = wide + to and `sepn` = wide + tn (equal-weight probability average).")


def cv_table(path, title):
    if not path.exists():
        return f"_{title}: {path.name} not found_"
    res = json.loads(path.read_text())
    out = [f"**{title}** (held-out Macro-F1 per fold, n = {res['all'].get('n', 22226)})", "",
           "| Arm | Folds 0-4 | Mean ± sd |", "|---|---|---:|"]
    for a, v in res["all"]["per_arm"].items():
        out.append(f"| {a} | {' / '.join(f'{x:.2f}' for x in v['per_fold'])} | {v['mean']:.2f} ± {v['sd']:.2f} |")
    out += ["", "| Comparison (A − B) | Per-fold difference | Mean | Positive folds | Paired t / p | Bootstrap 95% CI |", "|---|---|---:|---:|---|---|"]
    for c in res["all"]["comparisons"]:
        out.append(f"| {c['a']} − {c['b']} | {' '.join(f'{x:+.2f}' for x in c['per_fold'])} | {c['mean']:+.2f} | {c['positive_folds']}/5 | "
                   f"{c['t']:.2f} / {c['p_two_sided']:.3f} | [{c['bootstrap_ci95'][0]:+.2f}, {c['bootstrap_ci95'][1]:+.2f}] |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-outputs", action="store_true")
    a = ap.parse_args()
    if a.from_outputs:
        out = Path(roots()["OUTPUTS"]); rows = from_outputs(out); base = out
        with open(out / "official_split_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
        cv_dir = out / "cv5"; cv_files = (cv_dir / "results.json", cv_dir / "results_supplement.json", cv_dir / "results_textonly.json")
    else:
        rows = list(csv.DictReader(open(REPO / "results/official_split_runs.csv"))); base = REPO / "results"
        cv_files = tuple(REPO / "results/analysis" / f for f in ("cv5.json", "cv5_supplement.json", "cv5_textonly.json"))
    main_models = {
        "CLIP + small MLP": ("clip_baselines", "small_mlp"),
        "CLIP + wide MLP (capacity-matched)": ("clip_baselines", "wide_mlp"),
        "Interaction-gated model": ("clip_baselines", "interaction_gated"),
        "Local patch-attention model": ("clip_baselines", "local_patch_attention"),
        "OCR text branch (standalone)": ("ocr_text_branch", "ocr_text_branch"),
        "Separately trained fusion model": ("fusion", "separately_trained_fusion"),
        "**OCR-enhanced CLIP model (main)**": ("main_model", "main"),
        "Variant: text-swap + masking training": ("text_swap_variant", "variant"),
    }
    ablations = {"Main model": ("main_model", "main"), "No OCR text": ("main_model", "no_ocr_text"),
                 "OCR branch without image": ("main_model", "ocr_branch_no_image"),
                 "OCR loss does not update LoRA": ("main_model", "ocr_loss_no_lora_update"),
                 "Fusion loss only": ("main_model", "fusion_loss_only"), "No LoRA": ("main_model", "no_lora")}
    gated = {k: ("clip_baselines", v) for k, v in {
        "Interaction-gated model": "interaction_gated", "  no difference features": "interaction_gated_no_difference",
        "  no product features": "interaction_gated_no_product", "  no difference and product": "interaction_gated_no_both",
        "  no DINOv2": "interaction_gated_no_dino", "  no DINOv2, no difference and product": "interaction_gated_no_dino_no_both",
        "  bilinear view scoring": "interaction_gated_bilinear_view"}.items()}
    variant = {k: ("text_swap_variant", v) for k, v in {
        "Variant (swap + consistency + masking)": "variant", "  no consistency loss": "swap_masking_no_consistency",
        "  random instead of bias-guided swap": "random_swap_consistency_masking", "  no masking": "swap_consistency_no_masking",
        "  swap only": "swap_only", "  masking only": "masking_only"}.items()}
    doc = ["# Result tables", "", f"Generated by `scripts/make_tables.py` from `{base.name}/`. Official MMSD2.0 split; "
           "mean ± sample sd over seeds 42–44. The test split was used for earlier design decisions, so official-split test "
           "results are exploratory; the 5-fold CV (train+validation only) is the primary comparison.", "",
           "## 1. Official split: baselines vs. main model", "", table(rows, main_models), "",
           "Seeds 42–46:", "", table(rows, {"OCR-enhanced CLIP model (main)": ("main_model", "main"),
                                            "Variant": ("text_swap_variant", "variant")}, (42, 43, 44, 45, 46)),
           "", "## 2. 5-fold cross-validation (primary comparison)", "", CV_ARMS, "", cv_table(cv_files[0], "Pre-registered CV"), "",
           cv_table(cv_files[1], "CV supplement (separately trained fusion, main model without OCR text)"), "",
           cv_table(cv_files[2], "CV control: fusion with a text branch that reads no OCR text"), "",
           "Subgroups (images with / without text, OCR > 60 words, original-train rows only) are in the JSON files. Bootstrap "
           "seed 20261108 for the pre-registered CV and 20261109 for the supplement and the control, so the interval of the "
           "same comparison can differ by 0.01.", "",
           "## 3. Main-model ablations (official split)", "", table(rows, ablations), "",
           "## 4. Interaction-gated model ablations (official split)", "", table(rows, gated), "",
           "## 5. Text-swap + masking variant components (official split)", "", table(rows, variant), ""]
    cost = REPO / "results/analysis/cost_model.json"
    if cost.exists():
        c = json.loads(cost.read_text()); o = json.loads((REPO / "results/analysis/cost_ocr.json").read_text())
        doc += ["## 6. Inference cost (RTX 2070 SUPER, 256 validation images)", "",
                "| Setting | Preprocess ms/img | Forward ms/img | OCR ms/img | End-to-end ms/img |", "|---|---:|---:|---:|---:|"]
        for k, v in c["timing"].items():
            ocr = o["ms_per_image_mean"] if k.startswith("ocr") else 0.0
            doc.append(f"| {k} | {v['preprocess_ms_per_image']} | {v['forward_ms_per_image']} | {ocr} | "
                       f"{round(v['preprocess_ms_per_image'] + v['forward_ms_per_image'] + ocr, 1)} |")
        p = c["params"]
        doc += ["", f"Parameters — main model: {p['ocr_enhanced_clip']['total']:,} total / {p['ocr_enhanced_clip']['trainable']:,} trainable; "
                f"CLIP + wide MLP: {p['image_text_branch_eq_wide_mlp']['total']:,} / {p['image_text_branch_eq_wide_mlp']['trainable']:,}.", ""]
    target = (base / "tables.md")
    target.write_text("\n".join(doc)); print(f"wrote {target}")


if __name__ == "__main__":
    main()
