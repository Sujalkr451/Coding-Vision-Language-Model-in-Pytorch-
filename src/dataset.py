
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

DTYPE = np.dtype([(k, '<i8') for k in ('shard', 'image_offset', 'image_size',
                                      'text_offset', 'text_size', 'image_id', 'caption_id')])
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def digest_id(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') & ((1 << 63) - 1)


def image_digest(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def preprocessing(train=False, size=224):
    geometry = [T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC)] if train else [
        T.Resize(size, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(size)]
    # No horizontal flips: captions may specify left/right.
    return T.Compose(geometry + [T.ToTensor(), T.Normalize(MEAN, STD)])


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def build_index(shard_dir, output, exclusions=None, validation_permille=10):
    """Disk-backed pair deduplication and decoded-image validation before training.

    The sqlite transaction makes each shard resumable. Exclusions are RGB hashes
    and original URLs exported by `src.evaluate --make-exclusions`.
    """
    root, output = Path(shard_dir).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shards = sorted(root.rglob('*.tar'))
    if not shards:
        raise ValueError('No downloaded .tar shards found')
    exclude = json.loads(Path(exclusions).read_text()) if exclusions else {}
    blocked_hashes, blocked_urls = set(exclude.get('image_hashes', [])), set(exclude.get('urls', []))
    settings = {'root': str(root), 'shards': [str(p.relative_to(root)) for p in shards],
                'validation_permille': validation_permille,
                'exclusions_sha256': hashlib.sha256(json.dumps(exclude, sort_keys=True).encode()).hexdigest()}
    settings_file = output / 'index_settings.json'
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        raise ValueError('Index settings changed: use a new output directory')
    atomic_json(settings_file, settings)
    db = sqlite3.connect(output / 'audit.sqlite')
    db.executescript('''CREATE TABLE IF NOT EXISTS samples (
        image_hash TEXT, caption_hash TEXT, split TEXT, row BLOB,
        UNIQUE(image_hash, caption_hash));
        CREATE TABLE IF NOT EXISTS done (shard INTEGER PRIMARY KEY, size INTEGER, mtime INTEGER,
        valid INTEGER, corrupt INTEGER, duplicate INTEGER, excluded INTEGER);''')
    for sid, shard in enumerate(shards):
        stat = shard.stat()
        old = db.execute('SELECT size, mtime FROM done WHERE shard=?', (sid,)).fetchone()
        if old:
            if old != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f'Indexed shard changed: {shard}')
            continue
        counts = dict(valid=0, corrupt=0, duplicate=0, excluded=0)
        # One shard worth of metadata only, not decoded images.
        with tarfile.open(shard, 'r:') as tar, db:
            members = {}
            for m in tar:
                if m.isfile():
                    key, _, ext = m.name.rpartition('.')
                    members.setdefault(key, {})[ext.lower()] = m
            for parts in members.values():
                im = next((parts[x] for x in ('jpg', 'jpeg', 'png', 'webp') if x in parts), None)
                if im is None:
                    continue
                try:
                    txt = parts['txt']
                    caption = tar.extractfile(txt).read().decode('utf-8').strip()
                    if not caption:
                        raise ValueError('empty caption')
                    with Image.open(tar.extractfile(im)) as image:
                        image.load()
                        ih = image_digest(image)
                    metadata = json.load(tar.extractfile(parts['json'])) if 'json' in parts else {}
                    if ih in blocked_hashes or metadata.get('url') in blocked_urls or metadata.get('sha256') in blocked_hashes or metadata.get('source_sha256') in blocked_hashes:
                        counts['excluded'] += 1
                        continue
                    ch = hashlib.sha256(' '.join(caption.split()).casefold().encode()).hexdigest()
                    iid, cid = digest_id(ih), digest_id(ch)
                    split = 'val' if iid % 1000 < validation_permille else 'train'
                    row = np.array([(sid, im.offset_data, im.size, txt.offset_data, txt.size, iid, cid)], dtype=DTYPE)
                    cursor = db.execute('INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?)', (ih, ch, split, row.tobytes()))
                    counts['valid' if cursor.rowcount else 'duplicate'] += 1
                except (KeyError, ValueError, OSError, UnicodeError, Image.DecompressionBombError):
                    counts['corrupt'] += 1
            db.execute('INSERT INTO done VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (sid, stat.st_size, stat.st_mtime_ns, *counts.values()))
        print(json.dumps({'shard': sid, **counts}), flush=True)
    counts = {}
    for split in ('train', 'val'):
        count = db.execute('SELECT COUNT(*) FROM samples WHERE split=?', (split,)).fetchone()[0]
        tmp = output / f'{split}.npy.tmp'
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=DTYPE, shape=(count,))
        for i, (blob,) in enumerate(db.execute('SELECT row FROM samples WHERE split=? ORDER BY rowid', (split,))):
            arr[i] = np.frombuffer(blob, dtype=DTYPE)[0]
        arr.flush()
        del arr
        os.replace(tmp, output / f'{split}.npy')
        counts[split] = count
    audit = dict(zip(('downloaded_valid', 'corrupt', 'duplicate_pairs', 'excluded'),
                     db.execute('SELECT SUM(valid), SUM(corrupt), SUM(duplicate), SUM(excluded) FROM done').fetchone()))
    db.close()
    manifest = {**settings, 'counts': counts, 'audit': audit,
                'files': [{'size': p.stat().st_size, 'mtime': p.stat().st_mtime_ns} for p in shards]}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(output / 'manifest.json', manifest)
    return manifest


class ShardedPairs(Dataset):
    def __init__(self, index_dir, split='train', seed=42, context_length=77):
        self.directory = Path(index_dir)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.rows = np.load(self.directory / f'{split}.npy', mmap_mode='r')
        self.transform = preprocessing(split == 'train')
        self.seed, self.epoch = seed, 0
        self.handles = {}
        self.split = split
        self.token_rows = self.token_ids = None
        metadata = self.directory / f'{split}.tokens.json'
        cached = json.loads(metadata.read_text()) if metadata.exists() else {}
        self.cached_tokens = bool(cached.get('complete') and
                                  cached.get('fingerprint') == self.manifest['fingerprint'] and
                                  cached.get('context_length') == context_length)
        self._load_tokens()

    def _load_tokens(self):
        if self.cached_tokens:
            self.token_rows = np.load(self.directory / f'{self.split}.tokens.npy', mmap_mode='r')
            self.token_ids = np.load(self.directory / f'{self.split}.token_ids.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['handles'] = {}
        state['rows'] = None
        state['token_rows'] = state['token_ids'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rows = np.load(self.directory / f'{self.split}.npy', mmap_mode='r')
        self._load_tokens()

    @property
    def training(self):
        return isinstance(self.transform.transforms[0], T.RandomResizedCrop)

    def __getitem__(self, index):
        r = self.rows[index]
        sid = int(r['shard'])
        try:
            if sid not in self.handles:
                # Bound open descriptors even for many workers and thousands of shards.
                if len(self.handles) >= 16:
                    self.handles.pop(next(iter(self.handles))).close()
                path = Path(self.manifest['root']) / self.manifest['shards'][sid]
                self.handles[sid] = open(path, 'rb')
            f = self.handles[sid]
            f.seek(int(r['image_offset']))
            data = f.read(int(r['image_size']))
            if self.cached_tokens:
                caption = torch.from_numpy(self.token_rows[index].astype(np.int64))
                caption_id = int(self.token_ids[index])
            else:
                f.seek(int(r['text_offset']))
                caption = f.read(int(r['text_size'])).decode().strip()
                caption_id = int(r['caption_id'])
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert('RGB')
                # Reproducible augmentation for mid-epoch resume regardless of worker RNG.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.seed + self.epoch * len(self) + index)
                    tensor = self.transform(image)
            return tensor, caption, int(r['image_id']), caption_id
        except (OSError, ValueError, UnicodeError, Image.DecompressionBombError):
            return None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shards', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--exclusions')
    args = p.parse_args()
    print(json.dumps(build_index(args.shards, args.output, args.exclusions), indent=2))
