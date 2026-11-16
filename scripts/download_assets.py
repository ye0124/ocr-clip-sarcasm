"""Download the dataset and pretrained models at the exact revisions used in the paper.

  python scripts/download_assets.py            # MMSD2.0 + CLIP ViT-B/32 + DeBERTa-v3-small
  python scripts/download_assets.py --dino     # also DINOv2-small (only for the interaction-gated model)

Writes the dataset to ${DATA}/hf_datasets, the arrow-file directory to ${DATA}/arrow_root.txt, and the
models to ${MODELS}/<name>. Large weight files are checked against the SHA-256 recorded on the original run.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sarcasm.config import roots  # noqa: E402

DATASET = dict(repo="coderchen01/MMSD2.0", data_dir="mmsd-v2", revision="5b39e5b844accf2331d1738a3d91d4cb59005a50")
MODELS = {
    "clip-vit-base-patch32": dict(repo="openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        files=["config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin", "special_tokens_map.json",
               "tokenizer.json", "tokenizer_config.json", "vocab.json"],
        sha256={"pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"}),
    "deberta-v3-small": dict(repo="microsoft/deberta-v3-small", revision="a36c739020e01763fe789b4b85e2df55d6180012",
        files=["config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json"],
        sha256={"pytorch_model.bin": "d4b0ebcc7799a2c2dde1a39199cb0ed709f2c6b63a48b8c05133b77955f271a9",
                "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"}),
    "dinov2-small": dict(repo="facebook/dinov2-small", revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        files=["config.json", "preprocessor_config.json", "model.safetensors"], sha256={}),
}
SPLITS = {"train": 19816, "validation": 2410, "test": 2409}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dino", action="store_true", help="also download DINOv2-small")
    ap.add_argument("--skip-dataset", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    r = roots(); data, models = Path(r["DATA"]), Path(r["MODELS"])
    for name, spec in MODELS.items():
        if name == "dinov2-small" and not a.dino:
            continue
        target = models / name
        for f in spec["files"]:
            path = hf_hub_download(spec["repo"], f, revision=spec["revision"], local_dir=target)
            if f in spec["sha256"]:
                assert sha256(path) == spec["sha256"][f], f"{name}/{f}: checksum differs from the original run"
        print(f"model {name}: {target}")
    if not a.skip_dataset:
        from datasets import load_dataset
        ds = load_dataset(DATASET["repo"], data_dir=DATASET["data_dir"], revision=DATASET["revision"],
                          cache_dir=str(data / "hf_datasets"))
        for split, n in SPLITS.items():
            assert len(ds[split]) == n, (split, len(ds[split]))
        arrow_root = Path(ds["train"].cache_files[0]["filename"]).parent
        assert sorted(p.name for p in arrow_root.glob("mmsd2.0-*.arrow")), arrow_root
        data.mkdir(parents=True, exist_ok=True)
        (data / "arrow_root.txt").write_text(str(arrow_root) + "\n")
        print(f"MMSD2.0 arrow files: {arrow_root} (written to {data / 'arrow_root.txt'})")


if __name__ == "__main__":
    main()
