"""OCR text branch: DeBERTa-v3-small over "[POST] post [IMAGE_TEXT] OCR text" plus a CLIP image vector.

Used standalone (``train/train_ocr_branch.py``, ``cv/cv_ocr_branch.py``) and as the second branch of
the main model (``models/ocr_clip.py``). The image vector and the OCR text are computed outside this
module. All variants use the same language encoder and a width-256 classifier. "SER" in class and
function names is the working name of this branch in the original research code; the [SCENE] source
token and its input slot exist in the architecture but are always empty in the reported runs.
"""
from pathlib import Path
import json

import sentencepiece as spm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

SOURCE_TOKENS = ("[POST]", "[IMAGE_TEXT]", "[SCENE]")


class NativeSERTokenizer:
    """Use the official SentencePiece normalization without backend conversion.

    Transformers 5.17's converted DeBERTa tokenizer changed the observed input
    'sorry ™' to a different token sequence. Source tokens are appended after
    all pretrained embedding rows, including the model's reserved rows.
    """
    def __init__(self, path):
        path = Path(path)
        self.sp = spm.SentencePieceProcessor(model_file=str(path / "spm.model"))
        self.base_vocab_size = json.loads((path / "config.json").read_text())["vocab_size"]
        assert self.base_vocab_size >= self.sp.get_piece_size()
        self.source_ids = {token: self.base_vocab_size + i for i, token in enumerate(SOURCE_TOKENS)}
        self.cls_token_id = self.sp.piece_to_id("[CLS]")
        self.sep_token_id = self.sp.piece_to_id("[SEP]")
        self.pad_token_id = self.sp.piece_to_id("[PAD]")
        assert (self.pad_token_id, self.cls_token_id, self.sep_token_id) == (0, 1, 2)

    def __len__(self):
        return self.base_vocab_size + len(SOURCE_TOKENS)

    def encode(self, text, add_special_tokens=False):
        ids = self.sp.encode(text, out_type=int)
        return [self.cls_token_id] + ids + [self.sep_token_id] if add_special_tokens else ids

    def convert_tokens_to_ids(self, tokens):
        return [self.source_ids[token] for token in tokens]

    def pad(self, data, *, padding=True, max_length=None, return_tensors="pt"):
        assert return_tensors == "pt"
        rows = data["input_ids"]
        width = max_length if padding == "max_length" else max(map(len, rows))
        assert all(len(row) <= width for row in rows)
        return dict(input_ids=torch.tensor([row + [self.pad_token_id]*(width-len(row)) for row in rows]),
                    attention_mask=torch.tensor([[1]*len(row) + [0]*(width-len(row)) for row in rows]))


def make_tokenizer(path: str | Path):
    return NativeSERTokenizer(path)


def tokenize_views(tokenizer, posts, ocrs, scenes, *, mode="paired", pad_to=256):
    """Preserve identical truncated evidence across paired/independent variants.

    Packed order is all posts, all OCR views, all scene views. Padding is masked.
    Source markers are ordinary learned embeddings, never label supervision.
    """
    assert mode in ("paired", "independent", "base", "joint")
    assert len(posts) == len(ocrs) == len(scenes)
    post_id, ocr_id, scene_id = tokenizer.convert_tokens_to_ids(list(SOURCE_TOKENS))
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    text_ids = [tokenizer.encode(text, add_special_tokens=False)[:96] for text in posts]
    ocr_ids = [tokenizer.encode(text, add_special_tokens=False)[:154] for text in ocrs]
    scene_ids = [tokenizer.encode(text, add_special_tokens=False)[:154] for text in scenes]
    post_views = [[cls, post_id] + text + [sep] for text in text_ids]
    views = [post_views]
    presence = torch.tensor([[bool(o.strip()), bool(c.strip())] for o, c in zip(ocrs, scenes)], dtype=torch.bool)
    if mode == "base":
        presence.zero_()
    if mode == "joint":
        flat = []
        for text, ocr, scene in zip(text_ids, ocr_ids, scene_ids):
            ids = [cls, post_id] + text + [sep]
            if ocr:
                ids += [ocr_id] + ocr + [sep]
            if scene:
                ids += [scene_id] + scene + [sep]
            flat.append(ids)
        assert max(map(len, flat), default=0) <= 411
        batch = tokenizer.pad({"input_ids": flat}, padding="max_length" if pad_to else True,
                              max_length=pad_to, return_tensors="pt")
        return batch, presence
    for marker, evidence in ((ocr_id, ocr_ids), (scene_id, scene_ids)):
        if mode == "paired":
            views.append([[cls, post_id] + text + [sep, marker] + ev + [sep]
                          for text, ev in zip(text_ids, evidence)])
        elif mode == "independent":
            views.append([[cls, marker] + ev + [sep] for ev in evidence])
    flat = [ids for group in views for ids in group]
    assert max(map(len, flat), default=0) <= 256
    batch = tokenizer.pad({"input_ids": flat}, padding="max_length" if pad_to else True,
                          max_length=pad_to, return_tensors="pt")
    batch.pop("token_type_ids", None)
    return batch, presence


class SERModel(nn.Module):
    def __init__(self, encoder_path: str | Path, vocab_size: int, *, mode="paired",
                 seed=42, gradient_checkpointing=True):
        super().__init__()
        assert mode in ("paired", "independent", "base", "joint")
        self.mode = mode
        torch.manual_seed(seed)
        self.encoder = AutoModel.from_pretrained(str(encoder_path), local_files_only=True,
                                                 dtype=torch.float32)
        self.encoder.resize_token_embeddings(vocab_size, mean_resizing=False)
        assert self.encoder.config.hidden_size == 768
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 17001)
            self.hidden = nn.Sequential(nn.Linear(2818, 256), nn.GELU(), nn.Dropout(0.1))
            self.classifier = nn.Linear(256, 2)
        # An auxiliary contrastive head will be added only for its controlled pilot.

    def forward(self, tokens, image_vectors, presence, *, return_aux=False):
        n = image_vectors.shape[0]
        assert image_vectors.shape == (n, 512) and presence.shape == (n, 2)
        states = self.encoder(**tokens).last_hidden_state[:, 0]
        if self.mode in ("base", "joint"):
            assert states.shape == (n, 768)
            if self.mode == "base":
                assert not presence.any()
            h, o, c = states, torch.zeros_like(states), torch.zeros_like(states)
        else:
            assert states.shape == (3*n, 768)
            h, o, c = states.split(n, dim=0)
        # CLIP vectors are L2-normalized while language states are not. Fixed
        # LayerNorm keeps a representation-scale difference from suppressing v.
        h = F.layer_norm(h, (768,))
        o = F.layer_norm(o, (768,)) * presence[:, 0:1]
        c = F.layer_norm(c, (768,)) * presence[:, 1:2]
        v = F.layer_norm(image_vectors, (512,))
        fused = torch.cat((h, o, c, v, presence.to(h.dtype)), dim=-1)
        hidden = self.hidden(fused)
        logits = self.classifier(hidden)
        if return_aux:
            return logits, dict(text=h, ocr=o, scene=c, fused=fused, hidden=hidden)
        return logits
