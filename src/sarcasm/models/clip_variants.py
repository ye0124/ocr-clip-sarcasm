"""Further CLIP-only variants built on ``interaction_gated.AblationModel`` (``SupplementModel``).

Variant strings (mapping to paper names in ``train/train_clip.py``):
  clip_lora_mlp_capacity_matched   CLIP + LoRA with the 1024-5604-2 head (the wide MLP; the reported
                                   CLIP + wide MLP runs build it through ``local_patch_attention.LERModel('B')``)
  e6_no_dino_no_both               interaction-gated model without DINOv2 and without difference/product features
  e6_full_bilinear_view            interaction-gated model with bilinear view scoring

"E6", "B1", "B2" and "C1" in the comments below are working names of the original research code:
E6 = interaction-gated model, B1 = wide MLP, B2 = no DINOv2 / no difference and product, C1 = bilinear
view scoring. Shared components keep the initialization of ``interaction_gated.py``. Model
construction preserves the caller's CPU/CUDA RNG streams. ``C1NoDINOAuditModel`` is a
verification-only class of the original code; no reported run uses it and it is deliberately absent
from ``SUPPLEMENT_ORDER``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .interaction_gated import (
    AblationModel,
    AblationSAGAHead,
    _initialize,
    state_hash,
)


SUPPLEMENT_ORDER = (
    "clip_lora_mlp_capacity_matched",
    "e6_no_dino_no_both",
    "e6_full_bilinear_view",
)

_BASE_VARIANTS = {
    "clip_lora_mlp_capacity_matched": "clip_lora_mlp",
    "e6_no_dino_no_both": "e6_no_dino",
    "e6_full_bilinear_view": "e6_full",
}


class BilinearSAGAHead(AblationSAGAHead):
    """Reuse an initialized historical head and add only a zero matrix.

    Transferring the existing child modules preserves their state names,
    values and construction order. No module is initialized a second time.
    ``view_bilinear`` uses the convention ``view.T @ W @ text / 16``.
    """

    def __init__(self, historical_head: AblationSAGAHead):
        # Calling AblationSAGAHead.__init__ would unnecessarily draw random
        # numbers and replace the already initialized historical modules.
        nn.Module.__init__(self)
        for name, module in historical_head.named_children():
            self.add_module(name, module)
        for name, parameter in historical_head.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in historical_head.named_buffers(recurse=False):
            self.register_buffer(
                name, buffer,
                persistent=name not in historical_head._non_persistent_buffers_set,
            )
        self.use_difference = historical_head.use_difference
        self.use_product = historical_head.use_product
        self.training = historical_head.training
        self.view_bilinear = nn.Parameter(self.view_proj.weight.new_zeros((256, 256)))

    def view_logits(self, views: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        """Score projected [B,V,256] views with a [B,256] text query.

        This also exposes the exact scoring computation for the fixed-view
        engineering check. It consumes no random numbers, including when
        there is only one view. The old shared text/bias terms are retained.
        """
        q = text_global.unsqueeze(1).expand(-1, views.size(1), -1)
        old_score = self.view_score(torch.cat([views, q], -1)).squeeze(-1)
        bilinear_score = ((views @ self.view_bilinear) * q).sum(-1) / 16.0
        return old_score + bilinear_score

    def forward(self, text_hidden, image_hidden, text_mask, image_views=None,
                return_aux=False):
        # This is the historical head forward with only view scoring changed.
        # Original t/v remain K/V in both cross-attention layers.
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
            view_score = self.view_logits(views, tg)
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


class SupplementModel(AblationModel):
    """Three frozen follow-up definitions, with the original runner API.

    ``variant`` is the public follow-up ID; ``base_variant`` identifies the
    historical construction. B1 and B2 never access the DINO input key. All
    E6 models retain the historical CLIP CLS view and patch/CLS combination.
    """

    def __init__(self, variant: str, clip_path: str | Path, seed: int):
        if variant not in SUPPLEMENT_ORDER:
            raise ValueError(f"Unknown supplement {variant!r}; expected {SUPPLEMENT_ORDER}")
        base_variant = _BASE_VARIANTS[variant]
        super().__init__(base_variant, clip_path, seed)
        self.base_variant = base_variant
        self.variant = variant
        if variant == "clip_lora_mlp_capacity_matched":
            self.mlp = _initialize(self.seed, "mlp", lambda: nn.Sequential(
                nn.Linear(1024, 5604), nn.GELU(), nn.Dropout(0.1), nn.Linear(5604, 2)))
        elif variant == "e6_no_dino_no_both":
            self.head.use_difference = False
            self.head.use_product = False
        else:
            self.head = BilinearSAGAHead(self.head)

    @property
    def uses_dino(self) -> bool:
        return self.base_variant not in ("clip_lora_mlp", "e6_no_dino")

    def forward(self, x, return_aux=False):
        if self.base_variant != "clip_lora_mlp":
            # The historical E6 dispatch uses our uses_dino property, and
            # forwards into the retained or bilinear head without mutation.
            return super().forward(x, return_aux=return_aux)

        text_output = self.backbone.text_model(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        image_output = self.backbone.vision_model(pixel_values=x["pixel_values"])
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

    def initialization_hashes(self) -> dict[str, object]:
        result = super().initialization_hashes()
        # Names are kept intact: this hash is directly comparable to the
        # historical full hash for B2/C1 and C1-noDINO verification models.
        shared = {name: value for name, value in self.state_dict().items()
                  if name != "head.view_bilinear"}
        modules = sorted({name.split(".", 1)[0] for name in shared})
        result.update({
            "base_variant": self.base_variant,
            "shared_without_bilinear": state_hash(shared),
            "shared_modules_without_bilinear": {
                name: state_hash(shared, prefix=name + ".") for name in modules},
            "bilinear_initialization": "zeros_no_rng" if hasattr(self, "head")
                and hasattr(self.head, "view_bilinear") else None,
        })
        return result

    def parameter_statistics(self) -> dict[str, object]:
        """Keep the original graph/effective coefficient counting convention.

        The bilinear matrix is called even with a single view. In that case
        all its coefficients cancel in softmax, so it contributes to the CE
        graph count but not the structurally effective coefficient count.
        """
        parameters = dict(self.named_parameters())
        trainable = {name: p for name, p in parameters.items() if p.requires_grad}
        is_mlp = self.base_variant == "clip_lora_mlp"
        unused_prefixes = [] if is_mlp else ["relation.", "incongruity."]
        if not is_mlp and not self.uses_dino:
            unused_prefixes.append("global_proj.")
        unused = {name: p.numel() for name, p in trainable.items()
                  if any(name.startswith(prefix) for prefix in unused_prefixes)}
        zero_columns = {}
        scorer_cancelled = 0
        pooling_bias_cancelled = 0
        bilinear_cancelled = 0
        if not is_mlp:
            disabled_blocks = int(not self.head.use_difference) + int(not self.head.use_product)
            zero_columns = {
                f"head.{stream}_diff.0.weight":
                getattr(self.head, f"{stream}_diff")[0].out_features * 256 * disabled_blocks
                for stream in ("text", "image")}
            scorer_cancelled = (self.head.view_score.weight.numel() +
                                self.head.view_score.bias.numel()) if not self.uses_dino else 257
            pooling_bias_cancelled = self.head.text_score.bias.numel() + self.head.image_score.bias.numel()
            if hasattr(self.head, "view_bilinear") and not self.uses_dino:
                bilinear_cancelled = self.head.view_bilinear.numel()
        nominal = sum(p.numel() for p in trainable.values())
        graph = nominal - sum(unused.values())
        return {
            "total": sum(p.numel() for p in parameters.values()),
            "requires_grad": nominal,
            "ce_graph_trainable": graph,
            "ce_effective_trainable": graph - sum(zero_columns.values()) - scorer_cancelled
                - pooling_bias_cancelled - bilinear_cancelled,
            "unused_trainable": sum(unused.values()),
            "unused_trainable_by_tensor": unused,
            "zero_input_weight_coefficients": sum(zero_columns.values()),
            "zero_input_weight_coefficients_by_tensor": zero_columns,
            "view_score_cancelled_coefficients": scorer_cancelled,
            "pooling_score_cancelled_coefficients": pooling_bias_cancelled,
            "bilinear_view_score_cancelled_coefficients": bilinear_cancelled,
            "note": "Structural coefficient counts; not a claim of matched effective capacity. "
                    "Unused parameters have grad=None; zero columns may still undergo AdamW decay. "
                    "Shared text and bias scorer terms cancel across views; for one view the entire "
                    "linear scorer and bilinear matrix cancel. B1 matches the original full CE graph "
                    "budget to within 346 parameters.",
        }


class C1NoDINOAuditModel(SupplementModel):
    """Verification-only single-view C1 model for reusing old no-DINO runs.

    Full and no-DINO historically initialize the same modules and values.
    This changes only the explicit view selection, then retains the added
    finite-score computation for forward/gradient/optimizer/RNG checks.
    """

    def __init__(self, clip_path: str | Path, seed: int):
        super().__init__("e6_full_bilinear_view", clip_path, seed)
        self.base_variant = "e6_no_dino"
        self.variant = "e6_no_dino_bilinear_view_verification"
