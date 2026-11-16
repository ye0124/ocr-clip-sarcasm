"""LoRA adapters for the q/v projections of the top CLIP encoder layers.

Moved unchanged from the original training script (``train_lora_e2e.py``). Parameter creation order
is part of the seeded initialization, so do not reorder this code.
"""
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x): return self.base(x) + self.scale * self.up(self.down(x))


def inject_lora(model, top=4, rank=8, alpha=8):
    paths = []
    for branch in ("text_model", "vision_model"):
        layers = getattr(getattr(model, branch), "encoder").layers
        for i in range(len(layers) - top, len(layers)):
            for name in ("q_proj", "v_proj"):
                attn = layers[i].self_attn; old = getattr(attn, name)
                setattr(attn, name, LoRALinear(old, rank, alpha))
                paths.append(f"{branch}.encoder.layers.{i}.self_attn.{name}")
    return paths


