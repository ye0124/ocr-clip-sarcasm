"""Interaction-gated model, its ablations, and CLIP + small MLP (``AblationModel``).

Paper names and the variant strings used here (mapping in ``train/train_clip.py``; "E6" is the
working name of the interaction-gated model in the original research code):
  CLIP + small MLP                      clip_lora_mlp
  Interaction-gated model               e6_full
  no difference / product / both        e6_no_difference, e6_no_product, e6_no_both
  no DINOv2                             e6_no_dino

All modules are constructed on CPU with independent, fixed RNG namespaces, so a component has the
same initial weights in every variant that contains it. Construction preserves the caller's CPU
random state and never accesses a training sampler or CUDA random stream.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, TypeVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel

from .saga_head import SAGAClipF
from .lora import inject_lora


ABLATION_ORDER = (
    "e6_full",
    "clip_lora_mlp",
    "e6_no_difference",
    "e6_no_product",
    "e6_no_both",
    "e6_no_dino",
)

# Persisted in experiment configuration; changing these starts a new protocol.
INITIALIZATION_NAMESPACES = {
    "backbone": 0,
    "lora": 1000,
    "head": 2000,
    "global_proj": 3000,
    "relation": 4000,
    "incongruity": 5000,
    "mlp": 6000,
}

_T = TypeVar("_T")


def _initialize(seed: int, namespace: str, factory: Callable[[], _T]) -> _T:
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also reseeds CUDA; all constructors here are CPU
        # modules, so seed only the default CPU generator inside the fork.
        torch.random.default_generator.manual_seed(int(seed) + INITIALIZATION_NAMESPACES[namespace])
        return factory()


class AblationSAGAHead(SAGAClipF):
    """Historical SAGA head with only the designated feature blocks disabled.

    The flags are plain attributes, so all five E6 variants have identical
    state_dict keys, shapes and initial values for a given seed.
    """

    def __init__(self, use_difference: bool = True, use_product: bool = True):
        super().__init__(hidden=256, heads=4, dropout=0.1, layers=2)
        self.use_difference = bool(use_difference)
        self.use_product = bool(use_product)

    def interaction_input(self, original, cross):
        difference = torch.abs(original - cross) if self.use_difference else torch.zeros_like(original)
        product = original * cross if self.use_product else torch.zeros_like(original)
        return torch.cat([original, cross, difference, product], -1)

    def forward(self, text_hidden, image_hidden, text_mask, image_views=None,
                return_aux=False):
        # Keep casts, mask, pooling, residuals, LayerNorm and dropout identical
        # to SAGAClipF. In particular, K/V remain the original t/v each layer.
        t = self.text_adapter(self.text_norm(self.text_proj(text_hidden.float())))
        v = self.image_adapter(self.image_norm(self.image_proj(image_hidden.float())))
        tm = text_mask.bool().clone()
        tm[:, 0] = False
        vm = torch.ones(v.shape[:2], dtype=torch.bool, device=v.device)

        ht, hv = t, v
        for t_block, v_block in zip(self.t2v, self.v2t):
            ht = t_block(ht, v, key_padding_mask=~vm)
            hv = v_block(hv, t, key_padding_mask=~tm)

        td = self.text_diff(self.interaction_input(t, ht))
        vd = self.image_diff(self.interaction_input(v, hv))
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


class AblationModel(nn.Module):
    """Raw-input CLIP LoRA model for the six preregistered comparisons.

    ``x`` contains input_ids, attention_mask and pixel_values. DINO variants
    additionally require the already mean-pooled frozen DINO patch cache
    ``x['dino']`` with shape [batch, 384]. Neither the MLP nor no-DINO variant
    reads this key. Forward returns logits by default, or (logits, aux).
    """

    def __init__(self, variant: str, clip_path: str | Path, seed: int):
        super().__init__()
        if variant not in ABLATION_ORDER:
            raise ValueError(f"Unknown ablation {variant!r}; expected {ABLATION_ORDER}")
        self.variant = variant
        self.seed = int(seed)
        self.clip_path = str(clip_path)
        self.backbone = _initialize(
            self.seed, "backbone",
            lambda: CLIPModel.from_pretrained(
                self.clip_path, local_files_only=True, attn_implementation="eager"),
        )
        if (self.backbone.config.text_config.hidden_size != 512 or
                self.backbone.config.vision_config.hidden_size != 768 or
                self.backbone.config.projection_dim != 512 or
                self.backbone.config.text_config.num_hidden_layers != 12 or
                self.backbone.config.vision_config.num_hidden_layers != 12):
            raise ValueError("This protocol requires CLIP ViT-B/32 dimensions and 12-layer encoders")
        if self.backbone.config.vision_config.patch_size != 32:
            raise ValueError("This protocol requires CLIP ViT-B/32 patch_size=32")
        self.backbone.requires_grad_(False)
        self.paths = _initialize(
            self.seed, "lora", lambda: inject_lora(self.backbone, top=4, rank=8, alpha=8))

        if variant == "clip_lora_mlp":
            self.mlp = _initialize(self.seed, "mlp", lambda: nn.Sequential(
                nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2)))
        else:
            self.head = _initialize(self.seed, "head", lambda: AblationSAGAHead(
                use_difference=variant not in ("e6_no_difference", "e6_no_both"),
                use_product=variant not in ("e6_no_product", "e6_no_both")))
            self.global_proj = _initialize(self.seed, "global_proj", lambda: nn.Sequential(
                nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 768)))
            # Preserve historical auxiliary modules and their state names.
            # They are never invoked by this CE-only model.
            self.relation = _initialize(self.seed, "relation", lambda: nn.Sequential(
                nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1)))
            self.incongruity = _initialize(self.seed, "incongruity", lambda: nn.Sequential(
                nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid()))

    @property
    def uses_dino(self) -> bool:
        return self.variant not in ("clip_lora_mlp", "e6_no_dino")

    def forward(self, x, return_aux=False):
        text_output = self.backbone.text_model(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        image_output = self.backbone.vision_model(pixel_values=x["pixel_values"])
        if self.variant == "clip_lora_mlp":
            # Frozen projections remain differentiable with respect to their
            # inputs, so the top-layer q/v LoRA adapters receive CE gradients.
            text_global = F.normalize(
                self.backbone.text_projection(text_output.pooler_output), dim=-1)
            image_global = F.normalize(
                self.backbone.visual_projection(image_output.pooler_output), dim=-1)
            fused = torch.cat([text_global, image_global], dim=-1)
            logits = self.mlp(fused)
            if return_aux:
                return logits, {"text_global": text_global, "image_global": image_global,
                                "fused": fused}
            return logits

        t, v = text_output.last_hidden_state, image_output.last_hidden_state
        if self.uses_dino:
            dino = x["dino"]
            if dino.ndim != 2 or dino.shape[-1] != 384:
                raise ValueError("Expected mean-pooled DINO cache with shape [batch, 384]")
            image_views = torch.stack([v[:, 0], self.global_proj(dino)], dim=1)
        else:
            # A real single view preserves the historical CLIP CLS path and
            # the 0.5 patch + 0.5 view combination; None would remove CLS.
            image_views = v[:, 0:1]
        return self.head(t, v[:, 1:], x["attention_mask"],
                         image_views=image_views, return_aux=return_aux)

    def initialization_hashes(self) -> dict[str, object]:
        state = self.state_dict()
        modules = sorted({name.split(".", 1)[0] for name in state})
        return {
            "algorithm": "sha256_sorted_name_dtype_shape_raw_bytes_v1",
            "full": state_hash(state),
            "modules": {name: state_hash(state, prefix=name + ".") for name in modules},
            "backbone": state_hash(state, prefix="backbone."),
            "lora": state_hash({n: p for n, p in state.items()
                                if n.startswith("backbone.") and
                                (n.endswith(".down.weight") or n.endswith(".up.weight"))}),
            "namespaces": dict(INITIALIZATION_NAMESPACES),
        }

    def parameter_statistics(self) -> dict[str, object]:
        """Count nominal tensors and structurally active CE coefficients.

        ``ce_graph_trainable`` excludes entirely uncalled modules.
        ``ce_effective_trainable`` also excludes input-zeroed weight columns
        and view-score coefficients that cancel algebraically in softmax.
        These counts describe structure, not empirical nonzero gradients or
        statistical capacity. They do not exclude LoRA-down at initialization.
        """
        parameters = dict(self.named_parameters())
        trainable = {n: p for n, p in parameters.items() if p.requires_grad}
        unused_prefixes = []
        if self.variant != "clip_lora_mlp":
            unused_prefixes += ["relation.", "incongruity."]
        if self.variant == "e6_no_dino":
            unused_prefixes += ["global_proj."]
        unused = {n: p.numel() for n, p in trainable.items()
                  if any(n.startswith(prefix) for prefix in unused_prefixes)}
        zero_columns = {}
        scorer_cancelled = 0
        pooling_bias_cancelled = 0
        if self.variant != "clip_lora_mlp":
            disabled_blocks = int(not self.head.use_difference) + int(not self.head.use_product)
            zero_columns = {f"head.{stream}_diff.0.weight":
                            getattr(self.head, f"{stream}_diff")[0].out_features * 256 * disabled_blocks
                            for stream in ("text", "image")}
            scorer_cancelled = (self.head.view_score.weight.numel() +
                                self.head.view_score.bias.numel()) if not self.uses_dino else 257
            pooling_bias_cancelled = self.head.text_score.bias.numel() + self.head.image_score.bias.numel()
        nominal = sum(p.numel() for p in trainable.values())
        graph = nominal - sum(unused.values())
        return {
            "total": sum(p.numel() for p in parameters.values()),
            "requires_grad": nominal,
            "ce_graph_trainable": graph,
            "ce_effective_trainable": graph - sum(zero_columns.values()) - scorer_cancelled - pooling_bias_cancelled,
            "unused_trainable": sum(unused.values()),
            "unused_trainable_by_tensor": unused,
            "zero_input_weight_coefficients": sum(zero_columns.values()),
            "zero_input_weight_coefficients_by_tensor": zero_columns,
            "view_score_cancelled_coefficients": scorer_cancelled,
            "pooling_score_cancelled_coefficients": pooling_bias_cancelled,
            "note": "Structural coefficient counts; not a claim of matched effective capacity. "
                    "Unused parameters have grad=None; zero columns may still undergo AdamW decay. "
                    "Shared text and bias scorer terms cancel across views; for one view the entire scorer cancels.",
        }


def state_hash(state: Mapping[str, torch.Tensor], prefix: str | None = None) -> str:
    """Stable CPU hash; includes names, dtype and shape to avoid ambiguity."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if prefix is not None and not name.startswith(prefix):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        # The byte view also handles bfloat16, unsupported by NumPy directly.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
