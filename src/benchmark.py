
import gc
import time
import torch
from .model import CLIP
from .clip_loss import clip_contrastive_loss
from .dataset import atomic_json


def benchmark(output, batches=(32, 64, 128, 256)):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for GPU capacity benchmark')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    results = []
    for batch in batches:
        model, optimizer, images, tokens, loss, a, b = (None,) * 7
        try:
            torch.manual_seed(42)
            model = CLIP().cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.1)
            images = torch.randn(batch, 3, 224, 224, device='cuda')
            tokens = torch.randint(1, 49406, (batch, 77), device='cuda')
            tokens[:, 0], tokens[:, -1] = 49406, 49407
            torch.cuda.reset_peak_memory_stats()
            for step in range(5):
                if step == 2:
                    torch.cuda.synchronize()
                    start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a, b = model(images, tokens)
                loss = clip_contrastive_loss(a, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError('BF16 loss is non-finite')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            results.append({'batch_per_gpu': batch, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                            'compute_images_per_second': batch * 3 / (time.monotonic() - start),
                            'final_loss': float(loss), 'status': 'ok'})
        except torch.cuda.OutOfMemoryError:
            results.append({'batch_per_gpu': batch, 'status': 'oom'})
        finally:
            model = optimizer = images = tokens = loss = a = b = None
            gc.collect()
            torch.cuda.empty_cache()
        print(results[-1], flush=True)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'precision': 'bf16', 'warmup_steps': 2, 'timed_steps': 3,
              'includes': 'forward, global-single-GPU contrastive loss, backward, clipping, AdamW',
              'excludes': 'dataset loading, tokenization, DDP, checkpoints', 'results': results}
    atomic_json(output, report)
    return report
