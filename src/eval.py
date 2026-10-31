
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .dataset import preprocessing
from .tokenization import tokenize


def retrieval_metrics(images, texts, image_ids, text_image_ids, chunk_size=128):
    images, texts = F.normalize(images.float(), dim=-1), F.normalize(texts.float(), dim=-1)
    image_ids, text_image_ids = torch.as_tensor(image_ids), torch.as_tensor(text_image_ids)
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError('Evaluation requires one image embedding per unique image')
    result = {}
    for name, queries, candidates, qids, cids in (
        ('i2t', images, texts, image_ids, text_image_ids),
        ('t2i', texts, images, text_image_ids, image_ids)):
        ranks = []
        for start in range(0, len(queries), chunk_size):
            scores = queries[start:start + chunk_size] @ candidates.T
            positive = qids[start:start + chunk_size, None] == cids[None, :]
            if not positive.any(dim=1).all():
                raise ValueError('Every query must have a positive candidate')
            best_positive = scores.masked_fill(~positive, -torch.inf).max(dim=1).values
            # Deterministic pessimistic tie handling for random/untrained embeddings.
            rank = ((scores >= best_positive[:, None]) & ~positive).sum(dim=1) + 1
            ranks.extend(rank.tolist())
        r = np.asarray(ranks)
        result.update({f'{name}_recall@{k}': float((r <= k).mean()) for k in (1, 5, 10)})
        result[f'{name}_mean_rank'] = float(r.mean())
        result[f'{name}_median_rank'] = float(np.median(r))
    result['mean_recall'] = float(np.mean([result[f'{d}_recall@{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10)]))
    return result


@torch.inference_mode()
def evaluate_retrieval(model, manifest, device, batch_size=64, cache_dir=None):
    records = json.loads(Path(manifest).read_text())
    if not records:
        raise ValueError('Empty retrieval manifest')
    image_features, text_features, ids, text_ids = [], [], [], []
    transform = preprocessing(False)
    model.eval()
    # The declared validation split is evaluated in full; never truncate to a few batches.
    for start in range(0, len(records), batch_size):
        rows = records[start:start + batch_size]
        tensors, captions = [], []
        for i, row in enumerate(rows, start):
            if not row['captions']:
                raise ValueError('Each image needs at least one caption')
            path = Path(row['image'])
            if not path.is_absolute():
                path = Path(manifest).parent / path
            with Image.open(path) as image:
                tensors.append(transform(image.convert('RGB')))
            captions.extend(row['captions'])
            ids.append(i)
            text_ids.extend([i] * len(row['captions']))
        image_features.append(model.image_encoder(torch.stack(tensors).to(device)).float().cpu())
        for offset in range(0, len(captions), batch_size):
            tokens = tokenize(captions[offset:offset + batch_size], model.text_encoder.context_size, cache_dir)
            text_features.append(model.text_encoder(tokens.to(device)).float().cpu())
    return retrieval_metrics(torch.cat(image_features), torch.cat(text_features), ids, text_ids)
