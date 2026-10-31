import torch
import torch.nn.functional as F


def clip_contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):

    # L2 normalization
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Similarity matrix
    logits = image_embeddings @ text_embeddings.T
    logits = logits / temperature

    # Correct image-text pair indices
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Image-to-text loss
    image_loss = F.cross_entropy(logits, labels)

    # Text-to-image loss
    text_loss = F.cross_entropy(logits.T, labels)

    # Symmetric contrastive loss
    loss = (image_loss + text_loss) / 2

    return loss