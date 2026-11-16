"""OCR region dropout: label-independent, lexicon-free masking of in-image text (training only).

For each training sample, the masker decides which OCR lines to hide. Each chosen line is
removed from BOTH input paths:
  * pixels: the line's box (with padding) is filled with the median colour of a thin ring
    around it, or heavily Gaussian-blurred, so CLIP can no longer read it;
  * OCR text: the line is dropped from the string given to the OCR text branch.

Masking decisions depend only on (seed, epoch, sample id, line index). They never look at
the label or the line content, so the augmentation cannot introduce a new label correlation.
Validation and test inputs are never masked.
"""
from dataclasses import dataclass, asdict
import hashlib
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class DropoutConfig:
    p_sample: float = 0.5   # probability that a training sample is augmented at all
    p_all: float = 0.2      # given augmented: hide every line (otherwise hide each line with p_line)
    p_line: float = 0.5     # per-line hide probability in partial mode
    pad_x: float = 0.20     # horizontal padding, as a fraction of the box height
    pad_y: float = 0.30     # vertical padding, as a fraction of the box height
    p_blur: float = 0.5     # per-line: Gaussian blur instead of ring-median fill
    ring: int = 3           # ring width in pixels for the fill colour
    seed: int = 0

    def to_dict(self):
        return asdict(self)


class OcrRegionDropout:
    def __init__(self, cfg: DropoutConfig):
        assert 0 <= cfg.p_sample <= 1 and 0 <= cfg.p_all <= 1 and 0 <= cfg.p_line <= 1
        self.cfg = cfg

    def rng(self, epoch: int, sample_id: str) -> random.Random:
        key = f"{self.cfg.seed}:{epoch}:{sample_id}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def choose(self, n_lines: int, epoch: int, sample_id: str):
        """Return (hidden flags per line, rng). Content-blind and label-blind."""
        rng = self.rng(epoch, sample_id)
        if n_lines == 0 or rng.random() >= self.cfg.p_sample:
            return [False] * n_lines, rng
        if rng.random() < self.cfg.p_all:
            return [True] * n_lines, rng
        hidden = [rng.random() < self.cfg.p_line for _ in range(n_lines)]
        if not any(hidden):  # an augmented sample hides at least one line
            hidden[rng.randrange(n_lines)] = True
        return hidden, rng

    def _box(self, box_normalized, width, height):
        x0, y0, x1, y1 = box_normalized
        h = max(1.0, (y1 - y0) * height)
        px, py = self.cfg.pad_x * h, self.cfg.pad_y * h
        return (max(0, int(x0 * width - px)), max(0, int(y0 * height - py)),
                min(width, int(np.ceil(x1 * width + px))), min(height, int(np.ceil(y1 * height + py))))

    def mask_image(self, image: Image.Image, lines, hidden, rng: random.Random) -> Image.Image:
        if not any(hidden):
            return image
        image = image.convert("RGB").copy()
        width, height = image.size
        array = np.asarray(image)
        draw = ImageDraw.Draw(image)
        for line, hide in zip(lines, hidden):
            if not hide:
                continue
            x0, y0, x1, y1 = self._box(line["box_normalized"], width, height)
            if x1 <= x0 or y1 <= y0:
                continue
            if rng.random() < self.cfg.p_blur:
                region = image.crop((x0, y0, x1, y1))
                radius = max(4.0, 0.6 * (y1 - y0))
                image.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x0, y0))
            else:
                r = self.cfg.ring
                outer = array[max(0, y0 - r):min(height, y1 + r), max(0, x0 - r):min(width, x1 + r)]
                inner_mask = np.ones(outer.shape[:2], dtype=bool)
                oy, ox = y0 - max(0, y0 - r), x0 - max(0, x0 - r)
                inner_mask[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = False
                ring_pixels = outer[inner_mask] if inner_mask.any() else outer.reshape(-1, 3)
                fill = tuple(int(v) for v in np.median(ring_pixels.reshape(-1, 3), axis=0))
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
        return image

    @staticmethod
    def mask_text(lines, hidden) -> str:
        """Same join as train_joint_v2.Split: newline-separated line texts, hidden lines dropped."""
        return "\n".join(line["text"] for line, hide in zip(lines, hidden) if not hide)
