
import torch.nn as nn
from .model_arch import VisionTransformerCLIP, TextEncoder


class CLIP(nn.Module):
    def __init__(self, image_layers=12, text_layers=6, embedding_dimension=256,
                 context_size=77, vocab_size=49408, num_heads=4):
        super().__init__()
        self.image_encoder = VisionTransformerCLIP(embedding_dimension, image_layers)
        self.text_encoder = TextEncoder(vocab_size, embedding_dimension, context_size,
                                        num_heads, text_layers)

    def forward(self, images, token_ids):
        return self.image_encoder(images), self.text_encoder(token_ids)
