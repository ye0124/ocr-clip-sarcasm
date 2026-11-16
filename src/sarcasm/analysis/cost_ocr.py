"""PaddleOCR per-image latency with the exact configuration used for the cached OCR (validation images only).

Runs in the PaddleOCR environment:
  python src/sarcasm/analysis/cost_ocr.py $(cat data/arrow_root.txt) outputs/analysis/cost_ocr.json
"""
import io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from ocr_paddle import load_table, recognise
from paddleocr import PaddleOCR
import paddle
ARROW = Path(sys.argv[1]); OUT = Path(sys.argv[2]); N, WARM = 256, 10
table = load_table(ARROW, "validation")
raws = [table.column("image")[i].as_py()["bytes"] for i in range(N + WARM)]
t0 = time.perf_counter()
ocr = PaddleOCR(lang="en", device="gpu:0", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
init_s = time.perf_counter() - t0
for r in raws[:WARM]: recognise(ocr, r)
times, nlines = [], []
for r in raws[WARM:]:
    t = time.perf_counter(); lines = recognise(ocr, r); times.append(time.perf_counter() - t); nlines.append(len(lines))
import numpy as np
times = np.array(times); has = np.array(nlines) > 0
res = dict(gpu=paddle.device.cuda.get_device_name(0), paddle=paddle.__version__, n_images=N, warmup=WARM, model_init_s=round(init_s, 1),
           ms_per_image_mean=round(1000 * times.mean(), 1), ms_per_image_median=round(1000 * float(np.median(times)), 1),
           ms_p90=round(1000 * float(np.percentile(times, 90)), 1),
           ms_mean_with_text=round(1000 * times[has].mean(), 1), ms_mean_without_text=round(1000 * times[~has].mean(), 1),
           frac_with_text=round(float(has.mean()), 3),
           peak_gpu_mib=round(paddle.device.cuda.max_memory_allocated() / 2**20, 1))
OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
