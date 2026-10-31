import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from .checkpoint import load_inference
from .dataset import preprocessing
from .tokenization import tokenize


@torch.inference_mode()
def match(weights, image, captions, device='cpu'):
    model = load_inference(weights, device)
    with Image.open(image) as im:
        tensor = preprocessing(False)(im.convert('RGB')).unsqueeze(0).to(device)
    tokens = tokenize(captions, model.text_encoder.context_size).to(device)
    a, b = model(tensor, tokens)
    return (F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T).squeeze(0).tolist()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--captions', nargs='+', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    for caption, score in zip(a.captions, match(a.weights, a.image, a.captions, a.device)):
        print(f'{score:.4f}\t{caption}')
