import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(minutes=60))
    return device


def cleanup():
    if active():
        dist.destroy_process_group()


def gather_features(x):
    # Autograd backward sums contributions from every rank, before DDP averages.
    return torch.cat(all_gather(x), dim=0) if active() else x


def gather_ids(x):
    if not active():
        return x
    outputs = [torch.empty_like(x) for _ in range(world_size())]
    dist.all_gather(outputs, x)
    return torch.cat(outputs)


def mean(value, device):
    x = torch.as_tensor(value, dtype=torch.float64, device=device)
    if active():
        dist.all_reduce(x)
        x /= world_size()
    return x.item()
