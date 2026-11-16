"""Bias-guided cross-sample text swap (S4): label-preserving counterfactuals for in-image text.

For a training sample x (image, OCR lines, label y), build x' by pasting ONE text line cropped from
another training image (the donor) into x's image and inserting its text into x's OCR lines. The label
of x' is y.

Donor choice (decided by (seed, epoch, sample id) only):
  * with p_guided: from the bias pool of the OPPOSITE class -- lines that a cross-fitted, OCR-only
    Naive Bayes model (train split only, see bias_lines.py) finds most indicative of the other label;
  * otherwise: uniformly from all eligible train lines.
This puts text that predicts class c onto samples of class not-c, so no in-image text feature keeps a
strong label correlation. No lexicon is used.

Placement: with p_replace (if x has OCR lines) the donor covers one existing line of x (that line is
first filled with its ring-median colour and removed from x's OCR text); otherwise it is inserted at a
random position avoiding existing text boxes. Edges are feathered.
"""
from dataclasses import dataclass, asdict
import hashlib
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class SwapConfig:
    p_cf: float = 0.5          # fraction of training samples that get a counterfactual copy each epoch
    p_guided: float = 0.5      # donor from the opposite-class bias pool (else uniform pool)
    p_replace: float = 0.5     # cover an existing line (if any) instead of inserting
    scale_low: float = 0.8
    scale_high: float = 1.25
    pad_x: float = 0.20        # padding used to erase a covered line of the target
    pad_y: float = 0.30
    crop_pad_x: float = 0.15   # tighter padding for the donor crop (avoids fragments of neighbouring lines)
    crop_pad_y: float = 0.12
    feather: float = 2.0
    ring: int = 3
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _padded(box, width, height, pad_x, pad_y):
    x0, y0, x1, y1 = box
    h = max(1.0, (y1 - y0) * height)
    return (max(0, int(x0 * width - pad_x * h)), max(0, int(y0 * height - pad_y * h)),
            min(width, int(np.ceil(x1 * width + pad_x * h))), min(height, int(np.ceil(y1 * height + pad_y * h))))


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class TextSwap:
    def __init__(self, cfg: SwapConfig, uniform_pool, positive_pool, negative_pool):
        """Pools are lists of (donor key, line index); donor key is resolved by the caller."""
        assert uniform_pool and positive_pool and negative_pool
        self.cfg, self.uniform, self.pos, self.neg = cfg, uniform_pool, positive_pool, negative_pool

    def rng(self, epoch, sample_id):
        key = f"swap:{self.cfg.seed}:{epoch}:{sample_id}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def plan(self, epoch, sample_id, label):
        """Return None (no counterfactual) or (rng, donor_key, donor_line, guided)."""
        rng = self.rng(epoch, sample_id)
        if rng.random() >= self.cfg.p_cf:
            return None
        guided = rng.random() < self.cfg.p_guided
        pool = (self.pos if label == 0 else self.neg) if guided else self.uniform
        for _ in range(10):
            key, line = pool[rng.randrange(len(pool))]
            if key != sample_id:
                return rng, key, line, guided
        return None

    def apply(self, image, lines, donor_image, donor_line, rng):
        """Return (new image, new OCR line dicts) for the counterfactual x'."""
        image = image.convert("RGB").copy()
        donor_image = donor_image.convert("RGB")
        W, H = image.size
        dW, dH = donor_image.size
        c = self.cfg
        crop_box = _padded(donor_line["box_normalized"], dW, dH, c.crop_pad_x, c.crop_pad_y)
        patch = donor_image.crop(crop_box)
        pw, ph = patch.size
        # keep the donor's relative text height, randomly rescaled
        target_h = (crop_box[3] - crop_box[1]) / dH * H * rng.uniform(c.scale_low, c.scale_high)
        target_h = max(8.0, min(0.5 * H, target_h))
        scale = target_h / ph
        if pw * scale > 0.95 * W:
            scale = 0.95 * W / pw
        nw, nh = max(4, int(round(pw * scale))), max(4, int(round(ph * scale)))
        patch = patch.resize((nw, nh), Image.BICUBIC)
        own = [_padded(l["box_normalized"], W, H, c.pad_x, c.pad_y) for l in lines]
        new_lines = [dict(l) for l in lines]
        replaced = None
        if lines and rng.random() < c.p_replace:
            k = rng.randrange(len(lines))
            bx = own[k]
            self._fill(image, bx)
            x0 = min(max(0, bx[0]), max(0, W - nw))
            y0 = min(max(0, bx[1]), max(0, H - nh))
            del new_lines[k]
            replaced = k
        else:
            x0 = y0 = 0
            for _ in range(30):
                x0, y0 = rng.randint(0, max(0, W - nw)), rng.randint(0, max(0, H - nh))
                if not any(_overlap((x0, y0, x0 + nw, y0 + nh), b) for b in own):
                    break
        inset = min(int(c.feather), (min(nw, nh) - 1) // 2 - 1)
        if inset >= 1:  # feathered edges
            mask = Image.new("L", (nw, nh), 0)
            ImageDraw.Draw(mask).rectangle((inset, inset, nw - 1 - inset, nh - 1 - inset), fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(min(c.feather, inset)))
        else:  # patch too small to feather
            mask = Image.new("L", (nw, nh), 255)
        image.paste(patch, (x0, y0), mask)
        # the inserted line, expressed in the target's normalized coordinates (text box without padding)
        dx0, dy0, dx1, dy1 = donor_line["box_normalized"]
        sx = scale * dW / W; sy = scale * dH / H
        ox, oy = x0 / W - (crop_box[0] / dW) * sx, y0 / H - (crop_box[1] / dH) * sy
        inserted = dict(text=donor_line["text"], score=donor_line.get("score", 1.0),
                        box_normalized=[ox + dx0 * sx, oy + dy0 * sy, ox + dx1 * sx, oy + dy1 * sy], donor=True)
        if replaced is not None:
            new_lines.insert(replaced, inserted)  # takes the place of the covered line
        else:  # keep the original reading order; insert by vertical centre
            cy = (inserted["box_normalized"][1] + inserted["box_normalized"][3]) / 2
            at = next((i for i, l in enumerate(new_lines)
                       if (l["box_normalized"][1] + l["box_normalized"][3]) / 2 > cy), len(new_lines))
            new_lines.insert(at, inserted)
        return image, new_lines

    def _fill(self, image, box):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        arr = np.asarray(image)
        W, H = image.size
        r = self.cfg.ring
        outer = arr[max(0, y0 - r):min(H, y1 + r), max(0, x0 - r):min(W, x1 + r)]
        m = np.ones(outer.shape[:2], dtype=bool)
        oy, ox = y0 - max(0, y0 - r), x0 - max(0, x0 - r)
        m[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = False
        px = outer[m] if m.any() else outer.reshape(-1, 3)
        fill = tuple(int(v) for v in np.median(px.reshape(-1, 3), axis=0))
        ImageDraw.Draw(image).rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
