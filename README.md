# Does In-Image Text Help? A Controlled Evaluation of OCR-Enhanced CLIP for Multimodal Sarcasm Detection

Code, configurations and results for the CPS5801 (Advanced Artificial Intelligence) research project by Ye Chen.
This repository is the reproducibility package of the final paper of the same title; the
[paper-to-repository map](#paper-to-repository-map) lists the result file and the command behind each table.

Many sarcastic tweets carry text inside the image (memes, screenshots, captions). This project asks whether reading
that text with OCR and training it jointly with CLIP improves multimodal sarcasm detection on **MMSD2.0**, when the
comparison is made under strict conditions: a strong CLIP baseline, several seeds, paired statistics, and a
5-fold cross-validation that never touches the test split.

**Main model — OCR-enhanced CLIP model.** An image–text branch (CLIP ViT-B/32 with LoRA and a 1024-5604-2 MLP) and an
OCR text branch (DeBERTa-v3-small over the post text plus the PaddleOCR text of the image, concatenated with the
shared CLIP image vector) are trained jointly; the output averages the two branches' probabilities.

**Baseline — CLIP + wide MLP.** The image–text branch of the main model trained alone (5.9 M trainable parameters,
against 147.9 M for the main model). Its head width was chosen so that its trainable parameters match those of the
earlier interaction-gated model; "capacity-matched" in `results/tables.md` and in code comments refers to that match.

**Main findings** (5-fold CV, Macro-F1; see [Results](#results)):
- The main model is +0.39 above CLIP + wide MLP (4/5 folds, paired t p = 0.11, bootstrap 95% CI [+0.02, +0.74]),
  and the same model without OCR text scores only 0.01 lower ([−0.29, +0.30]).
- Joint training is not better than late fusion: averaging a separately trained wide MLP and OCR text branch is 0.39
  above the main model (p = 0.11, [+0.09, +0.68]) and 0.77 above the wide MLP (5/5 folds, p = 0.015).
- The same fusion with a text branch that never sees OCR text keeps +0.61 of that gain (p = 0.07); the OCR contribution inside
  the fusion is +0.16 [−0.08, +0.41]. The gain comes mainly from ensembling a second text model, not from the
  in-image text.
- On the official test split the main model and the wide MLP differ by 0.00. End-to-end inference is 4–10× slower,
  mostly because of OCR.

## Contents

1. [Repository structure](#repository-structure)
2. [Hardware](#hardware)
3. [Software and installation](#software-and-installation)
4. [Data and pretrained models](#data-and-pretrained-models)
5. [Reproducing the experiments](#reproducing-the-experiments)
6. [Results](#results)
7. [Paper-to-repository map](#paper-to-repository-map)
8. [Checkpoints](#checkpoints)
9. [Code provenance, attribution and AI use](#code-provenance-attribution-and-ai-use)
10. [Citation](#citation)

## Repository structure

```
ocr-clip-sarcasm/
├── README.md
├── AI_DISCLOSURE.md               AI use and code provenance
├── LICENSE                        MIT
├── requirements.txt               main environment (training, evaluation, analysis)
├── requirements-ocr.txt           PaddleOCR environment (OCR extraction only)
├── pyproject.toml                 makes `sarcasm` importable (pip install -e .)
├── configs/                       every run reported in the paper (JSON; no machine paths)
│   ├── clip_baselines.json        CLIP + small/wide MLP, interaction-gated model, local patch-attention model
│   ├── ocr_text_branch.json       standalone OCR text branch
│   ├── main_model.json            main model (seeds 42–46), LR grid, five ablations
│   ├── text_swap_variant.json     text-swap + masking variant and its components
│   ├── cv5.json                   5-fold CV: wide MLP, main model, variant, main model without OCR text
│   └── cv5_ocr_branch.json        5-fold CV: standalone text branch with / without OCR text (for the fusions)
├── data/                          README.md (dataset, preprocessing) and cv5/folds.json
├── scripts/
│   ├── download_assets.py         dataset + pretrained models at pinned revisions
│   ├── prepare_data.sh            OCR, frozen CLIP cache, bias-line pools, (DINOv2)
│   ├── check_data.py              compares prepared data with the original run
│   ├── smoke_test.sh              10-minute pipeline check
│   ├── train_official.sh          train + test every official-split run
│   ├── run_cv5.sh                 5-fold CV (test split never read)
│   ├── analyze.sh                 bootstrap, subgroup, cost, shortcut diagnostics, tables
│   ├── restore_predictions.py     shipped predictions -> outputs/ (analysis without GPU)
│   ├── verify_initialization.py   rebuilds every model on CPU and compares its initial weights with the original run
│   └── make_tables.py             results/tables.md from per-run metrics
├── src/sarcasm/
│   ├── config.py                  config loading, ${DATA}/${MODELS}/${OUTPUTS} placeholders
│   ├── utils.py                   metrics, atomic writes, hashing, OCR word counts for the subgroup analyses
│   ├── models/                    lora.py, saga_head.py, interaction_gated.py, clip_variants.py,
│   │                              local_patch_attention.py (incl. CLIP + wide MLP), ocr_text_branch.py,
│   │                              ocr_clip.py (main model)
│   ├── data/                      ocr_paddle.py, cache_clip.py, cache_dino.py, prepare_labels.py,
│   │                              bias_lines.py, text_swap.py, ocr_region_dropout.py
│   ├── train/                     train_clip.py, train_ocr_branch.py, train_main.py, clip_common.py
│   ├── cv/                        make_folds.py, prep_pools.py, cv_train.py, cv_ocr_branch.py, fuse_cv.py,
│   │                              cv_stats.py, analyze_cv.py, analyze_cv_supplement.py, analyze_cv_textonly.py
│   ├── evaluation/fuse.py         separately trained fusion model (fixed equal-weight rule)
│   └── analysis/                  test_bootstrap.py, cost_model.py, cost_ocr.py, shortcut_diagnostics.py,
│                                  label_statistics.py
└── results/
    ├── official_split_runs.csv    every official-split run: validation and test metrics + source record
    ├── training_cost.csv          training time and peak GPU memory of the original runs
    ├── reproduction_check.json    re-training with this repository vs. the original predictions
    ├── tables.md                  tables of the paper (generated)
    ├── analysis/                  CV and test statistics, inference cost, label statistics (JSON)
    ├── diagnostics/               shortcut diagnostics on validation (masking / injecting in-image text)
    ├── predictions/               per-sample predictions behind every statistical comparison (CSV.gz)
    └── ocr_word_counts.csv.gz     words of OCR text per image (original OCR run); defines the text subgroups
```

## Hardware

All reported runs used one **NVIDIA GeForce RTX 2070 SUPER (8 GB)**, driver 595.99, Ubuntu 26.04 (x86-64),
Python 3.10. Training and OCR need Linux with one CUDA GPU; the statistical analyses run on any CPU.
Peak allocated training memory: 1.03 GiB (CLIP + wide MLP), 4.36 GiB (main model), 5.2–5.5 GiB (text-swap
variant), so any single GPU with ≥ 8 GB works, including the course-provided RTX A5000 (24 GB), which this project
did not use. Batch sizes are fixed by the configs (effective batch 16) and do not depend on GPU memory. Time and
memory of the original runs are in `results/training_cost.csv`.

| Run | Time per run (RTX 2070 SUPER) |
|---|---|
| CLIP + wide MLP | ~16 min |
| OCR text branch (standalone) | ~40 min |
| Main model | ~60 min |
| Text-swap + masking variant | ~70 min |
| Standalone text branch, one CV fold (with / without OCR) | ~35 / ~28 min |
| One CV fold (all six arms) | ~4 h |
| All official-split runs (`train_official.sh`) | ~2.5 days |
| Full 5-fold CV (`run_cv5.sh`) | ~20 h |

## Software and installation

```bash
git clone https://github.com/ye0124/ocr-clip-sarcasm.git
cd ocr-clip-sarcasm

# main environment
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .

# PaddleOCR environment (only for OCR extraction)
python3.10 -m venv .venv-ocr
# paddlepaddle-gpu is not on PyPI; install it from the PaddlePaddle index for your CUDA version (reported runs: cu126)
.venv-ocr/bin/pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.venv-ocr/bin/pip install -r requirements-ocr.txt
```

Only the statistical analyses (no training): `pip install numpy==2.2.6 scipy==1.15.3 && pip install -e .` is enough.

Paths: data, models and outputs default to `data/`, `models/` and `outputs/` inside the repository. Override with
`SARCASM_DATA`, `SARCASM_MODELS`, `SARCASM_OUTPUTS` (and `SARCASM_ARROW_ROOT`). Config files contain only
placeholders such as `${DATA}/evidence`.

## Data and pretrained models

```bash
python scripts/download_assets.py            # add --dino for the interaction-gated model
PY=.venv/bin/python OCR_PY=.venv-ocr/bin/python bash scripts/prepare_data.sh
python scripts/check_data.py
```

| Asset | Identifier | Revision | License |
|---|---|---|---|
| MMSD2.0 | `coderchen01/MMSD2.0` (`mmsd-v2`) | `5b39e5b8` | see the dataset release |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `3d74acf9` | MIT |
| DeBERTa-v3-small | `microsoft/deberta-v3-small` | `a36c7390` | MIT |
| DINOv2-small (optional) | `facebook/dinov2-small` | `ed25f3a3` | Apache-2.0 |
| PaddleOCR | `paddleocr==3.3.3` (PP-OCRv5_server_det, en_PP-OCRv5_mobile_rec) | — | Apache-2.0 |

Dataset details, preprocessing steps and the expected directory layout are in [data/README.md](data/README.md).

**Preprocessing.** CLIP inputs use the standard CLIP processor (224×224, text truncated to 77 tokens). The OCR text
branch reads `[POST] post [IMAGE_TEXT] OCR lines` with the native DeBERTa SentencePiece tokenizer (post ≤ 96 tokens,
OCR text ≤ 154 tokens). OCR lines are ordered top-to-bottom, left-to-right; empty strings are dropped and no
confidence filter is applied.

## Reproducing the experiments

All commands run from the repository root. Every trainer resumes after an interruption and skips finished runs.

**0. Pipeline check (~10 min).**
```bash
bash scripts/smoke_test.sh
```

**1. Main experiment: main model on the official split (seed 42).**
```bash
python -m sarcasm.train.train_main --config configs/main_model.json --run main_seed42 --stage train
python -m sarcasm.train.train_main --config configs/main_model.json --run main_seed42 --stage test
```
Output in `outputs/main_model/main_seed42/`: `run_config.json`, `history.json`, `result.json` (selected epoch and
validation metrics), `best.pt`, `validation_predictions.json`, `test_metrics.json`, `test_predictions.json`.
Expected: validation Macro-F1 85.35, test Macro-F1 85.48 (see `results/official_split_runs.csv`).

**2. Baseline: CLIP + wide MLP (seed 42).**
```bash
python -m sarcasm.train.train_clip --config configs/clip_baselines.json --run wide_mlp_seed42 --stage train
python -m sarcasm.train.train_clip --config configs/clip_baselines.json --run wide_mlp_seed42 --stage test
```
Expected: validation 84.58, test 84.97.

**3. All official-split runs** (baselines, OCR branch, fusion, main model, ablations, variant):
`bash scripts/train_official.sh` (or `bash scripts/train_official.sh main` for main-model seeds 42–44 only).

**4. 5-fold cross-validation** (primary comparison; MMSD2.0 test is never read): `bash scripts/run_cv5.sh`.

**5. Analyses and tables:** `bash scripts/analyze.sh`, then `python scripts/make_tables.py --from-outputs`, which also
writes `outputs/official_split_runs.csv` for a row-by-row comparison with `results/official_split_runs.csv`.

**Statistics without a GPU or the dataset.** The per-sample predictions of the original runs are shipped; this
reproduces the paired bootstrap, the with/without-text analysis and all CV comparisons exactly (about 2 minutes
on a laptop CPU):
```bash
python scripts/restore_predictions.py
python -m sarcasm.analysis.test_bootstrap        # -> outputs/analysis/test_bootstrap.json  (= results/analysis/)
python -m sarcasm.cv.analyze_cv                  # -> outputs/cv5/results.json               (= results/analysis/cv5.json)
python -m sarcasm.cv.analyze_cv_supplement       # -> outputs/cv5/results_supplement.json    (= cv5_supplement.json)
python -m sarcasm.cv.analyze_cv_textonly         # -> outputs/cv5/results_textonly.json      (= cv5_textonly.json)
```
The text subgroups (images with / without text, OCR > 60 words) are defined by the number of OCR words per image.
The analyses read it from `data/evidence/*/ocr.jsonl` when you have run `prepare_data.sh`, and otherwise from
`results/ocr_word_counts.csv.gz`, the counts of the original OCR run.

**Determinism.** Seeds fix initialization, data order (`seed + 1000003·epoch`), dropout and augmentation; cuDNN
benchmark and TF32 are off and deterministic algorithms are on. Each config pins the SHA-256 of the initial weights
of the original run, and training stops if a rebuilt model differs. On the original machine, re-training the
CLIP + wide MLP gave bit-identical validation scores for all three seeds, and re-training `main_seed42` and
`wide_mlp_seed42` with this repository's code reproduced every validation and test prediction of the original runs
(largest probability difference < 1e-6). On other GPUs or library versions, small numerical differences are expected.
`python scripts/verify_initialization.py` rebuilds all 50 CLIP-only models and image-text branches on CPU and
compares them with the recorded hashes (no data or GPU needed). The initial weights are tied to x86-64: PyTorch draws
normal samples (the LoRA initialization) differently on ARM CPUs, so on e.g. Apple Silicon this check and the
initialization check of the trainers fail.

**Configuration and seeds.** Optimization settings (learning rates, batch sizes, epochs, patience, precision) and
the seed of every run are in `configs/*.json`; architecture constants (LoRA r = 8, α = 8 on q/v of the last 4
layers, head widths, token limits) are in `src/sarcasm/models/`. Official-split runs use seeds 42–44 (42–46 for the
main model and the variant), CV fold k uses seed 42 + k, the folds were drawn with seed 20261108, and the bootstrap
seeds are 20261108 (test, first CV analysis) and 20261109 (CV supplement and control).

**Protocol.** Hyperparameters and checkpoints are selected on validation Macro-F1 only (strictly greater, earliest
epoch on ties). Each run is evaluated on test once. Because test results were inspected while choosing among model
designs, official-split test numbers are reported as exploratory; the 5-fold CV over train + validation is the
primary comparison.

## Results

Macro-F1 (%). Full tables: [results/tables.md](results/tables.md); every run with its source record:
[results/official_split_runs.csv](results/official_split_runs.csv).

**5-fold CV (train + validation, 22,226 rows; one seed per fold, 42 + k).**

| Model | Mean ± sd over folds |
|---|---:|
| CLIP + wide MLP | 86.48 ± 0.47 |
| OCR-enhanced CLIP model (main) | 86.87 ± 0.26 |
| Variant: text-swap + masking training | 86.97 ± 0.63 |

| Paired comparison | Mean difference | Positive folds | Paired t (df 4), p | Bootstrap 95% CI |
|---|---:|---:|---:|---|
| Main − wide MLP | +0.39 | 4/5 | 0.11 | [+0.02, +0.74] |
| Variant − wide MLP | +0.49 | 4/5 | 0.07 | [+0.12, +0.85] |
| Variant − main | +0.10 | 3/5 | 0.62 | [−0.20, +0.40] |

**Where does the gain come from? (CV, same folds and seeds.)** Three further arms isolate the in-image text and the
joint training: the main model without OCR text, the separately trained fusion model (wide MLP and standalone OCR
text branch averaged with equal weights), and the same fusion whose text branch reads only the post. Results are in
`results/analysis/cv5_supplement.json` and `cv5_textonly.json`; the decision rules were fixed before training.

| Model | All rows | Images with text (17,387) | Without text (4,839) |
|---|---:|---:|---:|
| CLIP + wide MLP | 86.48 ± 0.47 | 86.61 ± 0.64 | 83.88 ± 1.07 |
| OCR-enhanced CLIP model (main) | 86.87 ± 0.26 | 87.05 ± 0.44 | 84.20 ± 1.70 |
| Main model, no OCR text | 86.86 ± 0.54 | 86.96 ± 0.81 | 84.47 ± 1.86 |
| Text branch with OCR text (standalone) | 85.61 ± 0.76 | 85.86 ± 0.89 | 82.45 ± 2.35 |
| Text branch without OCR text (standalone) | 85.22 ± 0.72 | 85.27 ± 0.95 | 82.85 ± 2.45 |
| Separately trained fusion model (wide + text branch with OCR) | 87.25 ± 0.60 | 87.52 ± 0.58 | 84.27 ± 1.91 |
| Same fusion, text branch without OCR | 87.09 ± 0.71 | 87.16 ± 0.85 | 84.83 ± 1.85 |

| Paired comparison (all rows) | Mean difference | Positive folds | Paired t (df 4), p | Bootstrap 95% CI |
|---|---:|---:|---:|---|
| Main − main without OCR text | +0.01 | 2/5 | 0.98 | [−0.29, +0.30] |
| Main − separately trained fusion | −0.39 | 1/5 | 0.11 | [−0.68, −0.09] |
| Separately trained fusion − wide MLP | +0.77 | 5/5 | 0.015 | [+0.47, +1.07] |
| Fusion without OCR − wide MLP | +0.61 | 4/5 | 0.07 | [+0.32, +0.88] |
| Fusion with OCR − fusion without OCR | +0.16 | 2/5 | 0.45 | [−0.08, +0.41] |
| Text branch with OCR − without OCR | +0.39 | 4/5 | 0.22 | [+0.02, +0.75] |

On images with text, fusion with OCR − fusion without OCR is +0.36 (5/5 folds, [+0.07, +0.65]); on images without
text it is −0.56 ([−1.17, +0.04]). These subgroup results are secondary and not corrected for multiple comparisons.
Restricting the held-out rows to the original MMSD2.0 train split (removing validation rows used for earlier model
selection) leaves every direction unchanged.

**Official split (seeds 42–44, mean ± sd; test exploratory).**

| Model | Validation | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| CLIP + small MLP | 84.30 ± 0.22 | 84.27 ± 0.33 | 84.17 ± 0.29 |
| CLIP + wide MLP | 84.71 ± 0.11 | 84.96 ± 0.12 | 84.86 ± 0.09 |
| Interaction-gated model | 84.66 ± 0.05 | 84.13 ± 0.52 | 83.97 ± 0.48 |
| Separately trained fusion model | 85.36 ± 0.16 | 85.36 ± 0.31 | 85.25 ± 0.29 |
| **OCR-enhanced CLIP model (main)** | 85.58 ± 0.22 | 84.98 ± 0.67 | 84.87 ± 0.67 |
| Main model: no OCR text | 85.12 ± 0.23 | 84.97 ± 0.11 | 84.87 ± 0.09 |

On test, all paired 95% bootstrap intervals between these models include 0 (`results/analysis/test_bootstrap.json`).

**Cost** (RTX 2070 SUPER, batch 16): CLIP + wide MLP 13.9 ms/image end to end; main model 140.8 ms/image, of which
PaddleOCR takes 125.4 ms. Trainable parameters 5.9 M vs. 147.9 M.

## Paper-to-repository map

| Paper | Files in `results/` | Regenerate with |
|---|---|---|
| Table 1, Figure 2, per-fold and original-train tables (5-fold CV) | `analysis/cv5.json`, `cv5_supplement.json`, `cv5_textonly.json`, `predictions/cv5/` | `scripts/run_cv5.sh`; statistics only: `sarcasm.cv.analyze_cv*` |
| Official-split table, ablations, variant and interaction-gated tables | `official_split_runs.csv`, `tables.md` | `scripts/train_official.sh`, `scripts/make_tables.py` |
| Test bootstrap intervals, subgroup table | `analysis/test_bootstrap.json`, CV files above (keys `with_text`, `no_text`, `ocr_gt60_words`) | `sarcasm.analysis.test_bootstrap`, `sarcasm.cv.analyze_cv*` |
| Error rates by subgroup (error analysis) | CV files above (`fpr_mean` / `fnr_mean` per arm and subgroup) | `sarcasm.cv.analyze_cv*` |
| Label statistics by amount of in-image text (error analysis) | `analysis/label_statistics.json` | `sarcasm.analysis.label_statistics` |
| Not in the paper: shortcut diagnostics (masking / injecting in-image text) | `diagnostics/` | `sarcasm.analysis.shortcut_diagnostics` |
| Cost table | `analysis/cost_model.json`, `cost_ocr.json`, `training_cost.csv` | `sarcasm.analysis.cost_model`, `cost_ocr.py` |
| Reproduction test (reproducibility statement) | `reproduction_check.json` | experiments 1 and 2 above |

Reported in the paper from the project's earlier experiment records and not regenerated by this repository: the
failure analysis of the text-swap variant (shares of false positives / false negatives by OCR length), the bootstrap
interval for removing DINOv2 from the interaction-gated model, the validation screening of earlier candidates
(including the LoRA-tuned Qwen2.5-VL-3B) and the count of development runs. Numbers of other systems are quoted
from their papers.

## Checkpoints

Model weights are not distributed: the main model's trainable weights are 565 MB per seed. Every result in `results/`
can be regenerated with the training commands above, and all per-run metrics and the per-sample predictions behind
the statistical comparisons are in `results/`.

## Code provenance, attribution and AI use

- **Written for this project** (by the author with AI coding assistants, see [AI_DISCLOSURE.md](AI_DISCLOSURE.md)):
  all code in `src/sarcasm`, `scripts/` and `configs/`. The repository consolidates the research code of the
  individual experiments; model code was moved without changes to the computation, training loops were kept line by
  line and only protocol bookkeeping (file locks, source-hash freeze checks, hard-coded paths) was removed.
- **No third-party repository code is copied or modified.** LoRA (`models/lora.py`) is an own implementation of the
  method of Hu et al. (2022).
- **Libraries:** PyTorch, Hugging Face Transformers / Datasets / Hub, SentencePiece, PaddleOCR / PaddlePaddle,
  NumPy, SciPy, scikit-learn, PyArrow, Pillow.
- **Pretrained models and data:** CLIP (Radford et al., 2021), DeBERTa-v3 (He et al., 2023), DINOv2 (Oquab et al.,
  2024), PaddleOCR PP-OCRv5, MMSD2.0 (Qin et al., 2023).

## Citation

```bibtex
@misc{chen2026ocrclipsarcasm,
  title  = {Does In-Image Text Help? A Controlled Evaluation of OCR-Enhanced CLIP for Multimodal Sarcasm Detection},
  author = {Chen, Ye},
  year   = {2026},
  note   = {CPS5801 course project}
}
```

Please also cite MMSD2.0: L. Qin et al., "MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System,"
Findings of ACL 2023, pp. 10834–10845.
