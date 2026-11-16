"""CLIP + wide MLP and the local patch-attention model (``LERModel``).

Variant strings (mapping to paper names in ``train/train_clip.py``):
  B     CLIP + wide MLP (capacity-matched baseline; also the image-text branch of the main model)
  LER   local patch-attention model: B plus a residual that attends from text tokens to image patches
  N, R, A, U, W   its ablations: no null key, no reference, neither, uniform attention, and a capacity
                  control (wide MLP with a 5845-unit hidden layer and no attention residual)

All variants share the initialization of the wide MLP (``clip_variants.SupplementModel``). Each input
uses one text encoder and one image encoder call. No external features or auxiliary loss.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .interaction_gated import AblationModel, _initialize, state_hash
from .clip_variants import SupplementModel


VARIANTS = ('B', 'LER', 'W', 'A', 'U', 'R', 'N')
CORE = ('B', 'LER', 'W', 'A')
BRANCH_NAMESPACE = 7000


def content_mask(input_ids, attention_mask):
    """CLIP B/32 tokenizer: BOS=49406, EOS/PAD=49407.

    Position checks also cover empty/padded rows; attention, never raw token
    IDs alone, determines validity. The tokenizer IDs are audited on host.
    """
    valid = attention_mask.bool()
    positions = valid.long().cumsum(-1)
    return (valid & (positions > 1) & (positions < valid.sum(-1, keepdim=True))
            & (input_ids != 49406) & (input_ids != 49407))


class LocalEvidenceResidual(nn.Module):
    def __init__(self, variant='LER'):
        super().__init__()
        assert variant in ('LER', 'U', 'R', 'N', 'A')
        self.variant = variant
        self.text = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128))
        self.image = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128))
        self.q = nn.Linear(128, 128, bias=False)
        self.k = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.null_key = nn.Parameter(torch.zeros(128))
        self.phi = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.word_score = nn.Linear(128, 1, bias=False)
        self.output = nn.Linear(128, 2, bias=False)
        nn.init.zeros_(self.output.weight)
        # Construct first, then remove unused tensors: retained parameters
        # have exactly the same initial values across structural ablations.
        if variant == 'U':
            del self.q, self.k, self.null_key
        elif variant in ('N', 'A'):
            del self.null_key

    def forward(self, text_hidden, patches, mask, *, zero_context=False,
                return_aux=False):
        assert patches.shape[1:] == (49, 768)
        t, v = self.text(text_hidden), self.image(patches)
        values = self.value(v)
        if self.variant == 'U':
            # Exactly 1/50 per patch, with the fiftieth (null) value zero.
            alpha = t.new_full((t.shape[0], t.shape[1], 50), 1 / 50)
            context = (values.sum(1) / 50).unsqueeze(1).expand(-1, t.shape[1], -1)
        else:
            keys = self.k(v)
            if hasattr(self, 'null_key'):
                keys = torch.cat((keys, self.null_key.view(1, 1, 128)
                                  .expand(t.shape[0], 1, -1)), dim=1)
            alpha = torch.softmax(self.q(t) @ keys.transpose(-1, -2) / (128 ** .5), -1)
            context = alpha[:, :, :49] @ values
        if zero_context:
            context = torch.zeros_like(context)
        evidence = self.phi(torch.cat((t, context), -1))
        if self.variant in ('LER', 'U', 'N'):
            evidence = evidence - self.phi(torch.cat((t, torch.zeros_like(context)), -1))
        scores = self.word_score(t).squeeze(-1)
        # Finite fill plus explicit normalization gives exactly zero weights
        # on all-empty rows, without a softmax over negative infinities.
        weights = torch.softmax(scores.masked_fill(~mask, torch.finfo(scores.dtype).min), -1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        pooled = (weights.unsqueeze(-1) * evidence).sum(1)
        correction = self.output(pooled)
        if return_aux:
            return correction, {'attention': alpha, 'word_weights': weights,
                                'context': context, 'evidence': evidence,
                                'pooled': pooled, 'content_mask': mask}
        return correction


class LERModel(SupplementModel):
    def __init__(self, variant: str, clip_path: str | Path, seed: int):
        assert variant in VARIANTS
        super().__init__('clip_lora_mlp_capacity_matched', clip_path, seed)
        self.variant = variant
        if variant == 'W':
            self.mlp = _initialize(seed, 'mlp', lambda: nn.Sequential(
                nn.Linear(1024, 5845), nn.GELU(), nn.Dropout(.1), nn.Linear(5845, 2)))
        elif variant != 'B':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed) + BRANCH_NAMESPACE)
                self.local = LocalEvidenceResidual(variant)

    def forward(self, x, return_aux=False, *, zero_context=False):
        text_output = self.backbone.text_model(
            input_ids=x['input_ids'], attention_mask=x['attention_mask'])
        image_output = self.backbone.vision_model(pixel_values=x['pixel_values'])
        text_global = F.normalize(self.backbone.text_projection(text_output.pooler_output), dim=-1)
        image_global = F.normalize(self.backbone.visual_projection(image_output.pooler_output), dim=-1)
        fused = torch.cat((text_global, image_global), -1)
        base_logits = self.mlp(fused)
        aux = {'text_global': text_global, 'image_global': image_global,
               'fused': fused, 'base_logits': base_logits}
        if hasattr(self, 'local'):
            mask = content_mask(x['input_ids'], x['attention_mask'])
            correction, local_aux = self.local(text_output.last_hidden_state,
                image_output.last_hidden_state[:, 1:], mask,
                zero_context=zero_context, return_aux=True)
            logits = base_logits + correction
            if return_aux:
                aux.update(local_aux, correction=correction)
        else:
            logits = base_logits
        return (logits, aux) if return_aux else logits

    def initialization_hashes(self):
        result = AblationModel.initialization_hashes(self)
        result['shared_baseline'] = state_hash({k: v for k, v in self.state_dict().items()
                                              if not k.startswith('local.')})
        result['branch_namespace'] = BRANCH_NAMESPACE
        return result

    def parameter_statistics(self):
        params = dict(self.named_parameters())
        trainable = {k: p for k, p in params.items() if p.requires_grad}
        count = sum(p.numel() for p in trainable.values())
        return {'total': sum(p.numel() for p in params.values()),
                'requires_grad': count, 'ce_graph_trainable': count,
                'ce_effective_trainable': count, 'unused_trainable': 0,
                'local_branch': sum(p.numel() for k, p in trainable.items() if k.startswith('local.')),
                'lora': sum(p.numel() for k, p in trainable.items() if k.startswith('backbone.')),
                'variant': self.variant,
                'note': 'U removes Q/K/null-key; N/A remove null-key. Counts exclude structurally unused tensors; '
                        'not a claim of identifiable parameters, nonzero initial gradients or equal statistical capacity.'}
