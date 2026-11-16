"""Write {id: label} for the given MMSD2.0 splits (input of bias_lines.py and the CV bias pools).

  python -m sarcasm.data.prepare_labels --splits train --output data/bias_pools/train_labels.json
"""
import argparse, json
from pathlib import Path

from datasets import Dataset, concatenate_datasets

from ..config import roots

N = {"train": 19816, "validation": 2410}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrow-root", type=Path, default=Path(roots()["ARROW_ROOT"]))
    ap.add_argument("--splits", nargs="+", choices=tuple(N), required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for split in a.splits:
        parts = [Dataset.from_file(str(p)) for p in sorted(a.arrow_root.glob(f"mmsd2.0-{split}*.arrow"))]
        d = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        assert len(d) == N[split]
        d = d.remove_columns([c for c in d.column_names if c not in ("id", "label")])
        res.update({str(i): int(y) for i, y in zip(d["id"], d["label"])})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res)); print(len(res))


if __name__ == "__main__":
    main()
