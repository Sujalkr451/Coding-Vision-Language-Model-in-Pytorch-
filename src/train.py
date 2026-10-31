
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

from . import distributed as du
from .checkpoint import load_checkpoint, save_checkpoint, unwrap
from .clip_loss import clip_contrastive_loss
from .dataloader import training_loader
from .dataset import atomic_json
from .eval import evaluate_retrieval
from .model import CLIP
from .tokenization import get_tokenizer
from .validate import validation_loss


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    t, d = config['training'], config['dataset']
    if d['image_size'] != 224 or d['context_length'] != config['model']['context_size']:
        raise ValueError('Dataset/model dimensions disagree with the custom architecture')
    if t['precision'] not in ('bf16', 'fp32'):
        raise ValueError('Use bf16 or fp32 (BF16 does not need a GradScaler)')
    for key in ('batch_size_per_gpu', 'epochs', 'gradient_accumulation', 'checkpoint_steps'):
        if t[key] < 1:
            raise ValueError(f'{key} must be positive')
    return config


def schedule(optimizer, total_steps, warmup):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1., progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run(config, resume=None):
    device = du.setup()
    writer = None
    try:
        t, d = config['training'], config['dataset']
        seed = t['seed'] + du.rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(t.get('cpu_threads', 4))
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.reset_peak_memory_stats(device)
            if t['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
                raise ValueError('Requested BF16 is not supported by this GPU')
        if not t.get('allow_small_dataset', False):
            manifest = config['evaluation'].get('validation_manifest')
            if not manifest or not Path(manifest).is_file():
                raise ValueError('Prepare held-out retrieval validation before allocating a large run')
        get_tokenizer(d.get('tokenizer_cache'))
        model = CLIP(**config['model']).to(device)
        if device.type == 'cuda' and t.get('channels_last', True):
            model.to(memory_format=torch.channels_last)
        if du.world_size() > 1:
            # Sync running buffers at each forward; per-rank BatchNorm retained.
            model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'],
                                      fused=device.type == 'cuda' and t.get('fused_optimizer', True))
        base_loader = training_loader(config, 0, du.rank(), du.world_size())
        batches = len(base_loader)
        if batches < 1:
            raise ValueError('Not enough audited data for one global batch')
        steps_per_epoch = math.ceil(batches / t['gradient_accumulation'])
        scheduler = schedule(optimizer, t['epochs'] * steps_per_epoch, t['warmup_steps'])
        fingerprint = base_loader.dataset.manifest['fingerprint']
        if not t.get('allow_small_dataset', False) and len(base_loader.dataset) < d['minimum_usable_pairs']:
            raise ValueError('Usable dataset below configured minimum; use smoke mode for small tests')
        epoch0, batch0, step, best, samples, history = 0, 0, 0, -math.inf, 0, []
        if resume:
            state = load_checkpoint(resume, model, optimizer, scheduler, fingerprint, config)
            epoch0, batch0, step = state['epoch'], state['next_batch'], state['global_step']
            best, samples, history = state['best_metric'], state['samples_processed'], state['history']
        output = Path(config['output_dir'])
        if du.rank() == 0:
            output.mkdir(parents=True, exist_ok=True)
            if not resume and (output / 'last_model.pth').exists():
                raise ValueError('Output has a checkpoint; resume or choose another output directory')
            atomic_json(output / 'run_config.json', config)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / 'tensorboard'), purge_step=step if resume else None)
        if du.active():
            dist.barrier()
        starting_step = step
        start = time.monotonic()
        skipped, dropped_batches = 0, 0
        nvml = None
        if device.type == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                nvml = pynvml
            except Exception:
                pass

        def emit(event):
            if du.rank() == 0:
                event = {'global_step': step, **event}
                print(json.dumps(event), flush=True)
                with open(output / 'events.jsonl', 'a') as f:
                    f.write(json.dumps(event) + '\n')
                for key, val in event.items():
                    if isinstance(val, (int, float)):
                        writer.add_scalar(key, val, step)
                writer.flush()

        def checkpoint(epoch, next_batch, extra=()):
            save_checkpoint(output / 'last_model.pth', model, optimizer, scheduler, config, epoch,
                            next_batch, step, best, samples, fingerprint, history, extra)
            if du.active():
                dist.barrier()
            if du.rank() == 0:
                # Explicit Volume commit after all files close; also has background commits.
                if os.environ.get('CLIP_MODAL_VOLUME'):
                    import modal
                    modal.Volume.from_name(os.environ['CLIP_MODAL_VOLUME']).commit()
                emit({'event': 'checkpoint', 'epoch': epoch, 'next_batch': next_batch})
            if du.active():
                dist.barrier()

        for epoch in range(epoch0, t['epochs']):
            offset = batch0 if epoch == epoch0 else 0
            loader = training_loader(config, epoch, du.rank(), du.world_size(), offset)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_start, group_start = time.monotonic(), time.monotonic()
            group_loss, group_samples, group_bad = 0., 0, False
            for batch_index, batch in enumerate(loader, offset):
                group_end = min((batch_index // t['gradient_accumulation'] + 1) * t['gradient_accumulation'], batches)
                group_begin = (batch_index // t['gradient_accumulation']) * t['gradient_accumulation']
                group_size = group_end - group_begin
                valid = torch.tensor(int(batch['valid']), device=device)
                bad_count = torch.tensor(batch['skipped'], device=device)
                if du.active():
                    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                    dist.all_reduce(bad_count)
                skipped += bad_count.item()
                if not valid.item():
                    group_bad = True
                    dropped_batches += 1
                elif not group_bad:
                    images = batch['images'].to(device, non_blocking=True,
                                                  memory_format=torch.channels_last if device.type == 'cuda' and t.get('channels_last', True) else torch.contiguous_format)
                    tokens = batch['tokens'].to(device, non_blocking=True)
                    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and t['precision'] == 'bf16' else contextlib.nullcontext()
                    with autocast:
                        image_features, text_features = model(images, tokens)
                    # Disable autocast for FP32 similarities, logsumexp and CE.
                    loss = clip_contrastive_loss(image_features, text_features, t['temperature'],
                                                batch['image_ids'].to(device), batch['caption_ids'].to(device))
                    finite = torch.tensor(int(torch.isfinite(loss)), device=device)
                    if du.active():
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError('Non-finite contrastive loss')
                    (loss / group_size).backward()
                    group_loss += du.mean(loss.detach(), device) / group_size
                    group_samples += len(images) * du.world_size()
                if batch_index + 1 != group_end:
                    continue
                if not group_bad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    samples += group_samples
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed = time.monotonic() - group_start
                    elapsed_tensor = torch.tensor(elapsed, device=device)
                    if du.active():
                        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                    event = dict(event='train', epoch=epoch, batch=batch_index + 1, train_loss=group_loss,
                                 learning_rate=optimizer.param_groups[0]['lr'], gradient_norm=float(grad_norm),
                                 images_per_second=group_samples / max(elapsed_tensor.item(), 1e-6),
                                 global_batch_size=t['batch_size_per_gpu'] * du.world_size(),
                                 optimizer_batch_size=group_samples, samples_processed=samples,
                                 skipped_corrupt=skipped, dropped_global_batches=dropped_batches,
                                 eta_seconds=(t['epochs'] * steps_per_epoch - step) * elapsed_tensor.item())
                    if device.type == 'cuda':
                        event['peak_vram_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
                        if nvml:
                            try:
                                event['gpu_utilization_percent'] = nvml.nvmlDeviceGetUtilizationRates(nvml.nvmlDeviceGetHandleByIndex(device.index)).gpu
                            except Exception:
                                pass
                    emit(event)
                if group_bad:
                    emit({'event': 'skipped_group', 'epoch': epoch, 'batch': batch_index + 1,
                          'skipped_corrupt': skipped, 'dropped_global_batches': dropped_batches})
                optimizer.zero_grad(set_to_none=True)
                group_loss, group_samples, group_bad = 0., 0, False
                group_start = time.monotonic()
                stop = torch.tensor(int((t.get('max_steps', 0) and step - starting_step >= t['max_steps']) or
                                        time.monotonic() - start >= t.get('max_runtime_seconds', 82800)), device=device)
                if du.active():
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if stop.item() or (step and step % t['checkpoint_steps'] == 0):
                    checkpoint(epoch, batch_index + 1)
                if stop.item():
                    emit({'event': 'stopped', 'elapsed_seconds': time.monotonic() - start})
                    return
            # Rank zero evaluates the unwrapped model; other ranks wait for the same metrics.
            metrics = [None]
            if du.rank() == 0:
                try:
                    metrics[0] = validation_loss(unwrap(model), config, device)
                    manifest = config['evaluation'].get('validation_manifest')
                    if manifest:
                        metrics[0].update(evaluate_retrieval(unwrap(model), manifest, device,
                                                            cache_dir=d.get('tokenizer_cache')))
                    elif not t.get('allow_small_dataset'):
                        raise ValueError('Large-scale runs require held-out retrieval validation')
                except Exception as exc:
                    metrics[0] = {'error': repr(exc)}
            if du.active():
                dist.broadcast_object_list(metrics, src=0)
            if 'error' in metrics[0]:
                raise RuntimeError(metrics[0]['error'])
            result = metrics[0]
            score = result.get('mean_recall', -result['val_loss'])
            improved = score > best
            best = max(best, score)
            history.append({'epoch': epoch + 1, 'global_step': step, **result})
            emit({'event': 'validation', 'epoch': epoch + 1, 'epoch_seconds': time.monotonic() - epoch_start, **result})
            extra = [output / 'best_model.pth'] if improved else []
            if (epoch + 1) % 5 == 0:
                extra.append(output / f'epoch_{epoch + 1:03d}.pth')
            checkpoint(epoch + 1, 0, extra)
        emit({'event': 'complete', 'elapsed_seconds': time.monotonic() - start})
    finally:
        if writer:
            writer.close()
        du.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--resume')
    args = p.parse_args()
    run(load_config(args.config), args.resume)
