
import torch
import torch.nn.functional as F
from .distributed import gather_features, gather_ids, rank


def clip_contrastive_loss(image_embeddings, text_embeddings, temperature=0.07,
                          image_ids=None, caption_ids=None, distributed=True):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Paired embedding batches must have identical shape")
    images = F.normalize(image_embeddings.float(), dim=-1)
    texts = F.normalize(text_embeddings.float(), dim=-1)
    all_images, all_texts = (gather_features(images), gather_features(texts)) if distributed else (images, texts)
    n = images.shape[0]
    if image_ids is None:
        image_ids = torch.arange(n, device=images.device) + rank() * n
    if caption_ids is None:
        caption_ids = image_ids
    all_i, all_t = (gather_ids(image_ids), gather_ids(caption_ids)) if distributed else (image_ids, caption_ids)
    # Repeated image OR identical caption constitutes a positive, not a false negative.
    positives = (image_ids[:, None] == all_i[None, :]) | (caption_ids[:, None] == all_t[None, :])

    def direction(logits):
        return (torch.logsumexp(logits, dim=1) -
                torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)).mean()

    return (direction(images @ all_texts.T / temperature) +
            direction(texts @ all_images.T / temperature)) / 2
