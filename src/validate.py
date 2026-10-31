import torch
from torch.utils.data import DataLoader
from .dataset import ShardedPairs
from .dataloader import PairCollator
from .clip_loss import clip_contrastive_loss


@torch.inference_mode()
def validation_loss(model, config, device):
    dataset = ShardedPairs(config['dataset']['index_dir'], split='val', context_length=config['dataset']['context_length'])
    loader = DataLoader(dataset, batch_size=config['training']['batch_size_per_gpu'],
                        collate_fn=PairCollator(config['dataset']['context_length'],
                                               config['dataset'].get('tokenizer_cache')))
    total, count, skipped = 0., 0, 0
    model.eval()
    for batch in loader:
        if not batch['valid']:
            skipped += batch['skipped']
            continue
        n = len(batch['images'])
        if n < 2:
            continue
        image, text = model(batch['images'].to(device), batch['tokens'].to(device))
        loss = clip_contrastive_loss(image, text, config['training']['temperature'],
                                    batch['image_ids'].to(device), batch['caption_ids'].to(device), distributed=False)
        total += loss.item() * n
        count += n
    if not count:
        raise ValueError('No usable validation batches with >=2 examples')
    return {'val_loss': total / count, 'val_samples': count, 'val_corrupt': skipped}
