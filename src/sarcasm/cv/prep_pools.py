"""Per-fold bias-line pools for the text-swap variant, built from each fold's inner-train rows only.

Same bias_lines.py and quantile (0.02) as the official-split variant. Requires labels of train+validation:
  python -m sarcasm.data.prepare_labels --splits train validation --output data/cv5/pool_labels.json
  python -m sarcasm.cv.prep_pools
Writes ${DATA}/cv5/fold{k}/bias_pools.json; the sha256 values must match configs/cv5.json.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

from ..config import roots


def main():
    R = roots()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-root", type=Path, default=Path(R["DATA"]) / "evidence")
    ap.add_argument("--cv-dir", type=Path, default=Path(R["DATA"]) / "cv5")
    a = ap.parse_args()
    folds = json.loads((a.cv_dir / "folds.json").read_text())["folds"]
    rows = {}
    for split in ("train", "validation"):
        for line in (a.evidence_root / split / "ocr.jsonl").read_text().splitlines():
            r = json.loads(line); rows[r["id"]] = line
    lab = json.loads((a.cv_dir / "pool_labels.json").read_text())
    out = {}
    for k, f in folds.items():
        d = a.cv_dir / f"fold{k}"; d.mkdir(parents=True, exist_ok=True)
        (d / "train_ocr.jsonl").write_text("\n".join(rows[i] for i in f["train"]) + "\n")
        (d / "train_labels.json").write_text(json.dumps({i: lab[i] for i in f["train"]}))
        subprocess.run([sys.executable, "-m", "sarcasm.data.bias_lines", "--ocr", str(d / "train_ocr.jsonl"),
                        "--labels", str(d / "train_labels.json"), "--output", str(d / "bias_pools.json"),
                        "--quantile", "0.02"], check=True, stdout=open(d / "bias_lines.log", "w"))
        pools = json.loads((d / "bias_pools.json").read_text())
        used = {i for pool in ("positive_pool", "negative_pool", "uniform_pool") for i, _ in pools[pool]}
        assert used <= set(f["train"]), "pool references rows outside inner train"
        out[k] = {"sha256": hashlib.sha256((d / "bias_pools.json").read_bytes()).hexdigest(),
                  "n_eligible": pools["n_eligible"], "pool_size": len(pools["positive_pool"])}
    print(json.dumps(out, indent=1))
    (a.cv_dir / "pools_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
