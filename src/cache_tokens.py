
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from .dataset import atomic_json, digest_id
from .tokenization import tokenize, TOKENIZER_NAME


def cache_tokens(index_dir, cache_dir=None, context_length=77, batch_size=1024):
    directory = Path(index_dir)
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {'fingerprint': manifest['fingerprint'], 'context_length': context_length,
                'tokenizer': TOKENIZER_NAME}
    for split in ('train', 'val'):
        metadata = directory / f'{split}.tokens.json'
        prior = json.loads(metadata.read_text()) if metadata.exists() else {}
        if prior and any(prior.get(k) != v for k, v in expected.items()):
            raise ValueError('Token cache belongs to a different dataset/tokenizer context')
        if prior.get('complete'):
            continue
        rows = np.load(directory / f'{split}.npy', mmap_mode='r')
        count = len(rows)
        mode = 'r+' if prior else 'w+'
        tokens = np.lib.format.open_memmap(directory / f'{split}.tokens.npy', mode=mode,
                                           dtype=np.uint16, shape=(count, context_length))
        ids = np.lib.format.open_memmap(directory / f'{split}.token_ids.npy', mode=mode,
                                        dtype=np.int64, shape=(count,))
        start = prior.get('next_row', 0)
        atomic_json(metadata, {**expected, 'next_row': start, 'complete': False})
        with ExitStack() as stack:
            handles = {}
            for offset in range(start, count, batch_size):
                captions = []
                for row in rows[offset:offset + batch_size]:
                    sid = int(row['shard'])
                    if sid not in handles:
                        if len(handles) >= 16:
                            handles.pop(next(iter(handles))).close()
                        handles[sid] = stack.enter_context(open(Path(manifest['root']) / manifest['shards'][sid], 'rb'))
                    f = handles[sid]
                    f.seek(int(row['text_offset']))
                    captions.append(f.read(int(row['text_size'])).decode().strip()[:8192])
                batch = tokenize(captions, context_length, cache_dir)
                end = offset + len(batch)
                tokens[offset:end] = batch.numpy().astype(np.uint16)
                ids[offset:end] = [digest_id(str(row.tolist())) for row in batch]
                if end == count or end - start >= 10000:
                    tokens.flush()
                    ids.flush()
                    atomic_json(metadata, {**expected, 'next_row': end, 'complete': end == count})
                    print(json.dumps({'cached_caption_split': split, 'rows': end, 'total': count}), flush=True)
                    start = end
        tokens.flush()
        ids.flush()
        atomic_json(metadata, {**expected, 'next_row': count, 'complete': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--tokenizer-cache')
    a = p.parse_args()
    cache_tokens(a.index, a.tokenizer_cache)
