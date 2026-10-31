"""Synthetic image–caption fixture for infrastructure tests."""
import io
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image
from .dataset import build_index, digest_id, image_digest


def create(root):
    root = Path(root)
    if (root / 'index/manifest.json').exists():
        return root / 'index'
    shards = root / 'shards'
    shards.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Generate enough identities to fill both sides of the hash-based split.
    train, val = [], []
    while len(train) < 320 or len(val) < 32:
        array = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        split = val if digest_id(image_digest(image)) % 1000 < 10 else train
        if len(split) < (32 if split is val else 320):
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            split.append(buf.getvalue())
    with tarfile.open(shards / '00000.tar', 'w') as tar:
        for i, blob in enumerate(train + val):
            for ext, payload in [('png', blob), ('txt', f'colorful random texture number {i}'.encode())]:
                member = tarfile.TarInfo(f'{i:06d}.{ext}')
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
    build_index(shards, root / 'index')
    return root / 'index'
