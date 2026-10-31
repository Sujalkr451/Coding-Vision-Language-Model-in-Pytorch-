import random
import torch
from torch.utils.data import DataLoader, Sampler
from .dataset import ShardedPairs, digest_id
from .tokenization import tokenize


class BlockDistributedSampler(Sampler):

    def __init__(self, size, batch_size, rank=0, world_size=1, seed=42, block_size=10000, start_batch=0):
        self.size, self.batch, self.rank, self.world = size, batch_size, rank, world_size
        self.seed, self.block, self.epoch, self.start = seed, block_size, 0, start_batch
        self.usable = size // (batch_size * world_size) * batch_size * world_size

    def __len__(self):
        return max(0, self.usable // self.world - self.start * self.batch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks = list(range((self.size + self.block - 1) // self.block))
        rng.shuffle(blocks)
        position = 0
        for block in blocks:
            indices = list(range(block * self.block, min((block + 1) * self.block, self.size)))
            rng.shuffle(indices)
            for index in indices:
                if position >= self.usable:
                    return
                if position % self.world == self.rank and position // self.world >= self.start * self.batch:
                    yield index
                position += 1


class PairCollator:
    def __init__(self, context=77, cache_dir=None):
        self.context, self.cache = context, cache_dir

    def __call__(self, samples):
        skipped = sum(x is None for x in samples)
        if skipped:
            return {'valid': False, 'skipped': skipped}
        images, captions, ids, text_ids = zip(*samples)
        if isinstance(captions[0], torch.Tensor):
            tokens = torch.stack(captions)
        else:
            tokens = tokenize([text[:8192] for text in captions], self.context, self.cache)
            # Truncated-identical captions are positives too.
            text_ids = [digest_id(str(row.tolist())) for row in tokens]
        return dict(valid=True, skipped=0, images=torch.stack(images), tokens=tokens,
                    image_ids=torch.tensor(ids), caption_ids=torch.tensor(text_ids))


def training_loader(config, epoch, rank=0, world_size=1, start_batch=0):
    d, t = config['dataset'], config['training']
    dataset = ShardedPairs(d['index_dir'], seed=t['seed'], context_length=d['context_length'])
    dataset.epoch = epoch
    sampler = BlockDistributedSampler(len(dataset), t['batch_size_per_gpu'], rank, world_size,
                                      t['seed'], start_batch=start_batch)
    sampler.epoch = epoch
    workers = t['num_workers']
    return DataLoader(dataset, batch_size=t['batch_size_per_gpu'], sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(t['seed'] + epoch),
                      collate_fn=PairCollator(d['context_length'], d.get('tokenizer_cache')),
                      **({'prefetch_factor': 2} if workers else {}))
