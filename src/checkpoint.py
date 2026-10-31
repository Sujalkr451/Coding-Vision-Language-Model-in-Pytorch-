import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from .distributed import active, rank, world_size
from .dataset import atomic_json


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, next_batch,
                    global_step, best_metric, samples_processed, fingerprint, history,
                    extra_paths=()):
    states = [None] * world_size()
    if active():
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if rank() != 0:
        return
    net = unwrap(model)
    payload = dict(format_version=1, image_encoder=net.image_encoder.state_dict(),
                   text_encoder=net.text_encoder.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), epoch=epoch, next_batch=next_batch,
                   global_step=global_step, best_metric=best_metric, config=config,
                   rng_states=states, world_size=world_size(), samples_processed=samples_processed,
                   data_fingerprint=fingerprint, history=history, scaler=None,
                   resume_semantics='Same world size/config/index; next deterministic batch; checkpoints at optimizer boundaries')
    atomic_save(payload, path)
    for alias in extra_paths:
        atomic_save(payload, alias)
        if Path(alias).name == 'best_model.pth':
            export_weights(Path(alias).parent / 'best_inference.pth', net, config['model'])
    atomic_json(Path(path).parent / 'training_history.json', history)
    export_weights(Path(path).parent / 'inference.pth', net, config['model'])


def export_weights(path, model, model_config):
    net = unwrap(model)
    atomic_save({'model_config': model_config, 'image_encoder': net.image_encoder.state_dict(),
                 'text_encoder': net.text_encoder.state_dict(),
                 'tokenizer': 'openai/clip-vit-base-patch32'}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, fingerprint=None, config=None):
    # Only load trusted, locally produced resumable files (contains RNG Python objects).
    state = torch.load(path, map_location='cpu', weights_only=False)
    if fingerprint and state['data_fingerprint'] != fingerprint:
        raise ValueError('Dataset changed since checkpoint')
    if optimizer is not None:
        if state['world_size'] != world_size():
            raise ValueError('Exact resume requires the same GPU/process count')
        if config:
            for key in ('model', 'training'):
                if config[key] != state['config'][key]:
                    raise ValueError(f'Exact resume requires identical {key} config')
    net = unwrap(model)
    net.image_encoder.load_state_dict(state['image_encoder'])
    net.text_encoder.load_state_dict(state['text_encoder'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng_states'][rank()])
    return state


def load_inference(path, device='cpu'):
    from .model import CLIP
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = CLIP(**state['model_config']).to(device)
    model.image_encoder.load_state_dict(state['image_encoder'])
    model.text_encoder.load_state_dict(state['text_encoder'])
    return model.eval()
