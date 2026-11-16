"""Rebuild every model on CPU and compare its initial weights with the SHA-256 recorded in the original runs.

  python scripts/verify_initialization.py            # all CLIP-only runs + main-model image-text branch
Needs only the pretrained models (scripts/download_assets.py); no data, no GPU. A match shows that this code
constructs exactly the models that produced the reported results. Run it on x86-64: PyTorch draws normal samples
(the LoRA initialization) differently on ARM CPUs, so the hashes do not match there.
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sarcasm.config import load_config, run_config  # noqa: E402
from sarcasm.train.clip_common import seed_all  # noqa: E402
from sarcasm.train.train_clip import MODELS  # noqa: E402
from sarcasm.models.local_patch_attention import LERModel  # noqa: E402


def main():
    bad = 0
    proto = load_config(REPO / "configs/clip_baselines.json")
    for r in proto["runs"]:
        cfg = run_config(proto, r["name"])
        seed_all(cfg["seed"])
        cls, variant = MODELS[cfg["model"]]
        got = cls(variant, cfg["clip_path"], cfg["seed"]).initialization_hashes()["full"]
        ok = got == cfg["expected_initialization_full"]; bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {r['name']}", flush=True)
    main = load_config(REPO / "configs/main_model.json")
    for seed, expected in main["b_initialization_full"].items():
        # the main model builds its image-text branch as LERModel("B"); see models/ocr_clip.py
        got = LERModel("B", main["common"]["clip_path"], int(seed)).initialization_hashes()["full"]
        ok = got == expected; bad += not ok
        print(f"{'ok ' if ok else 'BAD'} main-model image-text branch seed {seed}", flush=True)
    print("all initializations match the original runs" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
