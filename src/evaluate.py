
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
import torch
from .checkpoint import load_inference
from .dataset import image_digest, atomic_json
from .eval import evaluate_retrieval


def make_exclusions(manifests, output):
    hashes, urls = set(), set()
    for manifest in manifests:
        for row in json.loads(Path(manifest).read_text()):
            path = Path(row['image'])
            if not path.is_absolute():
                path = Path(manifest).parent / path
            with Image.open(path) as image:
                hashes.add(image_digest(image))
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            urls.update(row.get('urls', []))
            if row.get('url'):
                urls.add(row['url'])
    atomic_json(output, {'image_hashes': sorted(hashes), 'urls': sorted(urls)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', nargs='+', required=True)
    p.add_argument('--weights')
    p.add_argument('--output', default='evaluation.json')
    p.add_argument('--make-exclusions', action='store_true')
    a = p.parse_args()
    if a.make_exclusions:
        make_exclusions(a.manifest, a.output)
    else:
        if not a.weights or len(a.manifest) != 1:
            p.error('Evaluation requires --weights and exactly one manifest')
        torch.set_num_threads(4)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = evaluate_retrieval(load_inference(a.weights, device), a.manifest[0], device)
        atomic_json(a.output, result)
        print(json.dumps(result, indent=2))
