"""Fusion head of the interaction-gated model (paper: "Interaction-gated model").

Bidirectional cross-attention between CLIP text tokens and image patches, difference / product
features, optional DINOv2 patch tokens, and a gated fusion of the resulting views.
``SAGAClipF`` and ``SAGAClipLF`` are the working names of the original research code; they are kept
because ``interaction_gated.py`` subclasses them and the seeded initialization depends on the
order in which the parameters are created. Do not reorder this code.
"""
import torch
from torch import nn


class _CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv, key_padding_mask=None):
        x, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        x = self.norm1(q + x)
        return self.norm2(x + self.ffn(x))


class SAGAClipF(nn.Module):
    """Structure-Aware Gated Alignment and incongruity CLIP, frozen variant.

    Inputs follow the existing cache: text_hidden [B,L,512], image_hidden
    [B,S,768] (CLS followed by patch tokens), and text_mask [B,L].
    """
    def __init__(self, hidden=256, heads=4, dropout=0.1, layers=2):
        super().__init__()
        self.text_proj = nn.Linear(512, hidden)
        self.image_proj = nn.Linear(768, hidden)
        self.text_adapter = nn.Identity()
        self.image_adapter = nn.Identity()
        self.view_proj = nn.Linear(768, hidden)
        self.view_score = nn.Linear(hidden * 2, 1)
        self.text_norm = nn.LayerNorm(hidden)
        self.image_norm = nn.LayerNorm(hidden)
        self.t2v = nn.ModuleList([_CrossBlock(hidden, heads, dropout)
                                  for _ in range(layers)])
        self.v2t = nn.ModuleList([_CrossBlock(hidden, heads, dropout)
                                  for _ in range(layers)])
        # [x, cross(x), |x-cross(x)|, x*cross(x)]
        self.text_diff = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU()
        )
        self.image_diff = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU()
        )
        self.text_score = nn.Linear(hidden, 1)
        self.image_score = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(),
            nn.Linear(hidden, 3)
        )
        self.diff_proj = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    @staticmethod
    def _masked_mean(x, mask):
        den = mask.sum(1, keepdim=True).clamp_min(1).to(x.dtype)
        return (x * mask.unsqueeze(-1).to(x.dtype)).sum(1) / den

    @staticmethod
    def _attn_pool(x, score, mask):
        a = score(x).squeeze(-1).masked_fill(~mask, torch.finfo(x.dtype).min)
        a = torch.softmax(a, dim=1)
        return (a.unsqueeze(-1) * x).sum(1)

    def forward(self, text_hidden, image_hidden, text_mask, image_views=None,
                return_aux=False):
        t = self.text_adapter(self.text_norm(self.text_proj(text_hidden.float())))
        v = self.image_adapter(self.image_norm(self.image_proj(image_hidden.float())))
        tm = text_mask.bool().clone()
        # CLS/BOS is not a lexical token for pooling, matching Model 1.
        tm[:, 0] = False
        vm = torch.ones(v.shape[:2], dtype=torch.bool, device=v.device)

        ht, hv = t, v
        for t_block, v_block in zip(self.t2v, self.v2t):
            ht = t_block(ht, v, key_padding_mask=~vm)
            hv = v_block(hv, t, key_padding_mask=~tm)

        td = self.text_diff(torch.cat([t, ht, torch.abs(t - ht), t * ht], -1))
        vd = self.image_diff(torch.cat([v, hv, torch.abs(v - hv), v * hv], -1))
        tg = self._masked_mean(t, tm)
        vg_patch = v.mean(1)
        if image_views is None:
            vg = vg_patch
            view_alpha = None
        else:
            views = self.view_proj(image_views.float())
            q = tg.unsqueeze(1).expand(-1, views.size(1), -1)
            view_score = self.view_score(torch.cat([views, q], -1)).squeeze(-1)
            view_alpha = torch.softmax(view_score, dim=1)
            vg_views = (view_alpha.unsqueeze(-1) * views).sum(1)
            vg = 0.5 * vg_patch + 0.5 * vg_views
        tdg = self._attn_pool(td, self.text_score, tm)
        vdg = self._attn_pool(vd, self.image_score, vm)
        gate_logits = self.gate(torch.cat([tg, vg, tdg, vdg], -1))
        gates = torch.softmax(gate_logits, dim=-1)
        dg = self.diff_proj(torch.cat([tdg, vdg], -1))
        fused = gates[:, 0:1] * tg + gates[:, 1:2] * vg + gates[:, 2:3] * dg
        logits = self.classifier(fused)
        if return_aux:
            return logits, {"gates": gates, "view_alpha": view_alpha,
                            "text_global": tg,
                            "image_global": vg, "text_diff": tdg,
                            "image_diff": vdg, "fused": fused}
        return logits


class LowRankAdapter(nn.Module):
    """LoRA-form residual adapter for cached CLIP hidden states."""
    def __init__(self, dim, rank=8, alpha=8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return x + self.scale * self.up(self.down(x))


class SAGAClipLF(SAGAClipF):
    """SAGA-F with low-rank adapters on cached text/image CLIP features.

    This is a cache-compatible parameter-efficient adaptation baseline. It is
    not equivalent to backpropagating through the original CLIP Transformer.
    """
    def __init__(self, hidden=256, heads=4, dropout=0.1, layers=2, rank=8):
        super().__init__(hidden, heads, dropout, layers)
        self.text_adapter = LowRankAdapter(hidden, rank=rank)
        self.image_adapter = LowRankAdapter(hidden, rank=rank)
