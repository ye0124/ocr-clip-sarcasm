"""Data loading, metrics and reproducibility helpers for the CLIP-only models (moved unchanged from the original experiment code of the interaction-gated model)."""
import contextlib
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import CLIPProcessor


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Explicitly recorded: warn-only preserves availability on the historical CUDA stack.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def environment():
    import datasets, transformers, PIL
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'transformers': transformers.__version__, 'datasets': datasets.__version__,
            'pillow': PIL.__version__, 'gpu': torch.cuda.get_device_name(),
            'cudnn': torch.backends.cudnn.version(), 'threads': torch.get_num_threads(),
            'deterministic_algorithms': 'warn_only', 'cudnn_benchmark': False,
            'cudnn_deterministic': True, 'tf32': False,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def load_split(cfg, split):
    root = Path(cfg['arrow_root'])
    files = sorted(root.glob(f'mmsd2.0-{split}*.arrow'))
    assert files, (root, split)
    parts = [Dataset.from_file(str(p)) for p in files]
    part = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    assert len(part) == {'train': 19816, 'validation': 2410, 'test': 2409}[split]
    return part


def load_dino(cfg, split):
    result = {}
    for path in sorted((Path(cfg['dino_root']) / split).glob('shard-*.pt')):
        for batch in torch.load(path, map_location='cpu', weights_only=False):
            pooled = batch['dino_hidden'].float().mean(1)
            for sample_id, value in zip(batch['id'], pooled):
                key = str(sample_id)
                assert key not in result
                result[key] = value
    assert result
    return result


class Collator:
    def __init__(self, processor, dino=None):
        self.processor, self.dino = processor, dino

    def __call__(self, rows):
        x = self.processor(text=[str(r['text']) for r in rows],
                           images=[r['image'].convert('RGB') for r in rows],
                           return_tensors='pt', padding='max_length',
                           truncation=True, max_length=77)
        x['labels'] = torch.tensor([int(r['label']) for r in rows])
        x['ids'] = [str(r['id']) for r in rows]
        if self.dino is not None:
            x['dino'] = torch.stack([self.dino[k] for k in x['ids']])
        return x


def make_processor(cfg):
    return CLIPProcessor.from_pretrained(cfg['clip_path'], local_files_only=True, use_fast=False)


def order_for_epoch(seed, epoch, count):
    # No DataLoader/validation/initialization operation can advance this stream.
    generator = torch.Generator().manual_seed(seed + 1_000_003 * epoch)
    return torch.randperm(count, generator=generator).tolist()


def make_loader(part, collator, cfg, order=None):
    # Worker seeds have a separate RNG, including validation loaders.
    return DataLoader(part, batch_size=cfg['micro_batch'],
                      sampler=order, shuffle=False, collate_fn=collator,
                      num_workers=cfg.get('num_workers', 2), pin_memory=True,
                      generator=torch.Generator().manual_seed(9_173),
                      persistent_workers=False)


def metrics(labels, predictions, loss=None):
    result = {'rows': len(labels), 'accuracy': accuracy_score(labels, predictions),
              'macro_f1': f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0),
              'sarcasm_f1': f1_score(labels, predictions, pos_label=1, zero_division=0),
              'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}
    if loss is not None:
        result['ce_loss'] = loss
    return result


def autocast(cfg):
    return torch.autocast('cuda', dtype=torch.float16) if cfg['precision'] == 'fp16' else contextlib.nullcontext()


def trainable_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    assert set(state) == expected, (expected - set(state), set(state) - expected)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in state:
                value.copy_(state[name])


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    rows, labels, predictions = [], [], []
    loss_sum = 0.
    torch.cuda.synchronize()
    start = time.monotonic()
    for batch in loader:
        ids = batch.pop('ids')
        y = batch.pop('labels').cuda(non_blocking=True)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with autocast(cfg):
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction='sum')
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        probs = logits.float().softmax(-1).cpu().tolist()
        scores = logits.float().cpu().tolist()
        pred = logits.argmax(-1).cpu().tolist()
        truth = y.cpu().tolist()
        loss_sum += loss.item()
        labels += truth
        predictions += pred
        rows.extend({'id': sid, 'label': lab, 'logits': log, 'probabilities': prob, 'prediction': pr}
                    for sid, lab, log, prob, pr in zip(ids, truth, scores, probs, pred))
    torch.cuda.synchronize()
    result = metrics(labels, predictions, loss_sum / len(labels))
    result['seconds_including_loader'] = time.monotonic() - start
    return result, rows
