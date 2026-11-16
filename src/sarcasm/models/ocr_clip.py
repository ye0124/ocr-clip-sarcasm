"""OCR-enhanced CLIP model: image-text branch and OCR text branch inside one network, trained jointly.

Image-text branch: CLIP ViT-B/32 with LoRA (top-4 q/v, r = 8) and a 1024-5604-2 MLP. It is built by the
same code as the standalone CLIP + wide MLP baseline (``LERModel("B")``), so its initialization is
identical to the baseline for the same seed.
OCR text branch: DeBERTa-v3-small over "[POST] post [IMAGE_TEXT] OCR text", concatenated with the image-text
branch's L2-normalized CLIP image embedding (shared encoder, one CLIP forward), 2818-256-2 head.
Output: P = 0.5 * softmax(logits_image_text) + 0.5 * softmax(logits_ocr); sarcasm iff P[1] > 0.5.

Ablation switches (defaults = main model):
  to_image: "shared" | "detach" (OCR loss does not update the CLIP encoder) | "zero" (OCR branch sees no image)
  lora:     True | False (LoRA adapters frozen at their zero-initialized state, i.e. pretrained CLIP)
"""
import torch
from torch import nn

from .local_patch_attention import LERModel
from .ocr_text_branch import SERModel


class JointModel(nn.Module):
    def __init__(self, clip_path, encoder_path, vocab_size, seed):
        super().__init__()
        self.b = LERModel("B", clip_path, seed)
        self.t = SERModel(encoder_path, vocab_size, mode="joint", seed=seed, gradient_checkpointing=False)

    def forward(self, clip_inputs, tokens, presence):
        # Image-text branch in fp32, exactly as the standalone baseline; OCR branch under fp16 autocast.
        logits_b, aux = self.b(clip_inputs, return_aux=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits_t = self.t(tokens, aux["image_global"], presence)
        return logits_b, logits_t.float()

    @staticmethod
    def fuse(logits_b, logits_t):
        return 0.5 * logits_b.float().softmax(-1) + 0.5 * logits_t.float().softmax(-1)


class JointModelV2(JointModel):
    def __init__(self, clip_path, encoder_path, vocab_size, seed, *, to_image="shared", lora=True):
        super().__init__(clip_path, encoder_path, vocab_size, seed)
        assert to_image in ("shared", "detach", "zero")
        self.to_image = to_image
        if not lora:
            frozen = 0
            for name, p in self.b.named_parameters():
                if name.endswith(".down.weight") or name.endswith(".up.weight"):
                    p.requires_grad_(False); frozen += 1
            assert frozen == 32, frozen  # 4 layers x (q, v) x (text, vision)

    def forward(self, clip_inputs, tokens, presence):
        logits_b, aux = self.b(clip_inputs, return_aux=True)
        image = aux["image_global"]
        if self.to_image == "detach":
            image = image.detach()
        elif self.to_image == "zero":
            image = torch.zeros_like(image)
        with torch.autocast("cuda", dtype=torch.float16):
            logits_t = self.t(tokens, image, presence)
        return logits_b, logits_t.float()
