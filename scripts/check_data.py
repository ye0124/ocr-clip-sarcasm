"""Check prepared data against the original experiment (run after scripts/prepare_data.sh).

  python scripts/check_data.py                       # report
  python scripts/check_data.py --update-pool-hashes  # accept locally regenerated bias pools (see below)

PaddleOCR output can differ slightly across GPUs / driver versions. The trainers only require the OCR files to be
internally consistent (ocr_summary.json), but the bias-line pools of the text-swap variant are derived from the OCR
text and are pinned by SHA-256 in configs/text_swap_variant.json and configs/cv5.json. If your OCR differs from the
original, training of the variant stops at that check; --update-pool-hashes writes your pool hashes into the two
configs. The main model, baselines and ablations do not use the pools.
"""
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sarcasm.config import roots  # noqa: E402
from sarcasm.utils import ORIGINAL_OCR_SHA256  # noqa: E402

ORIGINAL_WITH_TEXT = {"train": 15500, "validation": 1887, "test": 1863}  # images with OCR text, original run
ROWS = {"train": 19816, "validation": 2410, "test": 2409}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-pool-hashes", action="store_true")
    a = ap.parse_args()
    R = roots(); data = Path(R["DATA"]); ok = True
    arrow = Path(R["ARROW_ROOT"])
    print(f"arrow root: {arrow} ({len(list(arrow.glob('mmsd2.0-*.arrow')))} files)")
    same_ocr = True
    for split, n in ROWS.items():
        s = data / "evidence" / split / "ocr_summary.json"
        if not s.exists():
            print(f"[missing] OCR {split}"); ok = False; continue
        summ = json.loads(s.read_text())
        assert summ["count"] == n and sha(s.parent / "ocr.jsonl") == summ["output_sha256"], split
        same = summ["output_sha256"] == ORIGINAL_OCR_SHA256[split]; same_ocr &= same
        print(f"[ok] OCR {split}: {summ['with_text']} images with text (original {ORIGINAL_WITH_TEXT[split]}); "
              f"{'identical to' if same else 'DIFFERS from'} the original OCR")
        m = data / "evidence" / "clip" / f"{split}_manifest.json"
        print(f"[{'ok' if m.exists() else 'missing'}] frozen CLIP cache {split}"); ok &= m.exists()
    cv = json.loads((REPO / "configs/cv5.json").read_text())
    assert sha(data / "cv5/folds.json") == cv["common"]["cv_folds_sha256"]; print("[ok] CV folds (shipped)")
    variant = json.loads((REPO / "configs/text_swap_variant.json").read_text())
    pools = {"official": (data / "bias_pools/train_bias_pools.json", variant["common"]["bias_pools_sha256"])}
    for r in cv["runs"]:
        if r.get("bias_pools"):
            pools[f"fold{r['fold']}"] = (data / f"cv5/fold{r['fold']}/bias_pools.json", r["bias_pools_sha256"])
    new = {}
    for name, (path, expected) in pools.items():
        if not path.exists():
            print(f"[missing] bias pools {name}"); ok = False; continue
        got = sha(path); new[name] = got
        print(f"[{'ok' if got == expected else 'DIFFERS'}] bias pools {name}")
    if a.update_pool_hashes and new:
        variant["common"]["bias_pools_sha256"] = new.get("official", variant["common"]["bias_pools_sha256"])
        for r in cv["runs"]:
            if r.get("bias_pools"):
                r["bias_pools_sha256"] = new.get(f"fold{r['fold']}", r["bias_pools_sha256"])
        (REPO / "configs/text_swap_variant.json").write_text(json.dumps(variant, indent=1) + "\n")
        (REPO / "configs/cv5.json").write_text(json.dumps(cv, indent=1) + "\n")
        print("updated bias_pools_sha256 in configs/text_swap_variant.json and configs/cv5.json")
    if not same_ocr:
        print("note: OCR differs from the original run; OCR-dependent results will be close to, not identical with, "
              "results/official_split_runs.csv")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
