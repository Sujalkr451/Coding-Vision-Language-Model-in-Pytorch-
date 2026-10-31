<p align="center">
  <img src="assets/vlm-banner.png" alt="Hand-drawn vision-language architecture with SpongeBob struggling at a school desk; image and caption branches connect to shared embeddings." width="100%" />
</p>

<h1 align="center">Building a Vision–Language Model in PyTorch</h1>
<p align="center">Learning how pixels and words connect, one layer at a time.</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/status-work_in_progress-e9b44c?style=flat-square" alt="Work in progress" />
</p>

## Large training dataset — PD12M (12.4M pairs)

- **Dataset:** [Spawning/PD12M](https://huggingface.co/datasets/Spawning/PD12M)
- **Caption metadata:** [metadata Parquet files](https://huggingface.co/datasets/Spawning/PD12M/tree/main/metadata)
- **Actual images:** [public AWS S3 bucket](https://registry.opendata.aws/pd12m/)
- **Publisher download instructions:** [PD12M images tutorial](https://huggingface.co/datasets/Spawning/PD12M/blob/main/tutorials/images.md)

PD12M contains real public-domain/CC0 images with machine-generated captions. The images are already hosted in a dedicated S3 bucket, rather than fetched from thousands of original websites. Images still need downloading, resizing, auditing and sharding; metadata alone is not the image dataset. Your original model architecture is unchanged.

A 512-record acquisition pilot completed in **35.6 seconds of downloader wall time** with **512 usable pairs** (509 train / 3 internal validation). The earlier DataComp pilot took about 121 seconds and yielded 320 usable pairs. This sample supports switching to PD12M; it is not a full-corpus speed guarantee. The original DataComp job was stopped and its downloaded shards retained separately.

```bash
# Prepare held-out evaluation data once:
modal run modal_train.py --stage prepare-eval
# Download, audit, and pre-tokenize the full real dataset (after approval):
modal run --detach modal_train.py --stage prepare --approve-full-run
# When preparation finishes, train your original model on 4 H100s:
modal run --detach modal_train.py --stage train --num-gpus 4 --approve-full-run
```

### Faster training path

- PyTorch scaled dot-product attention uses optimized CUDA kernels (including FlashAttention where supported) with your existing Q/K/V projections and causal masking.
- Fused CUDA AdamW, BF16, TF32, and channels-last CNN memory layout reduce GPU overhead.
- Captions are tokenized **once during preparation** and cached as compact memory-mapped arrays; training reuses them every epoch.
- Eight metadata downloads run concurrently; persistent tar shards, pinned-memory loading, prefetching and distributed sampling avoid repeated image downloads.
- Defaults use **256 pairs/GPU × 4 H100s = 1,024 contrastive pairs per step**. The full pool and H100 run are explicit commands, rather than a small test being the default.

The earlier measured throughput predates these optimizations. No additional GPU run or new speedup number is claimed for this update.

## Training the custom CLIP model

This project trains the original PyTorch encoders **from scratch**: a CNN feature extractor and 12-layer vision transformer, a 6-layer causal text transformer, 256-dimensional projections, 224 × 224 images, and 77-token captions. Only the `openai/clip-vit-base-patch32` **tokenizer** is downloaded; no pretrained CLIP model weights are used.

The pipeline supports local execution and single-node PyTorch DDP on one or four Modal H100 80GB GPUs. Full PD12M dataset preparation has been launched; large-scale GPU training has **not** started. Full preparation and training require `--approve-full-run`.

### Repository layout

| File | Responsibility |
| --- | --- |
| [`src/model_arch.py`](src/model_arch.py) | Original image/text encoders and transformer blocks, with numerical and configuration fixes. |
| [`src/model.py`](src/model.py) | Pairs both encoders into one module for DDP. |
| [`src/cache_tokens.py`](src/cache_tokens.py) | Restartable, one-time caption tokenization into compact on-disk arrays. |
| [`src/tokenization.py`](src/tokenization.py) | Lazy, cached CLIPTokenizer with truncation, padding, and vocabulary checks. |
| [`src/clip_loss.py`](src/clip_loss.py) | Symmetric CLIP loss, differentiable distributed gathering, duplicate positives. |
| [`src/prepare_data.py`](src/prepare_data.py) | Pinned PD12M/DataComp source adapters, bounded image downloading, retries and resume. |
| [`src/dataset.py`](src/dataset.py) | Tar audit, disk-backed deduplication, compact offset indices, preprocessing. |
| [`src/dataloader.py`](src/dataloader.py) | Disjoint distributed sampling, deterministic batches, cached tokenization. |
| [`src/train.py`](src/train.py) | AdamW, warmup/cosine schedule, BF16, accumulation, clipping, monitoring. |
| [`src/distributed.py`](src/distributed.py) | Process groups, device assignment, differentiable feature collectives. |
| [`src/checkpoint.py`](src/checkpoint.py) | Atomic resumable checkpoints, RNG state, inference exports. |
| [`src/prepare_evaluation.py`](src/prepare_evaluation.py) | Public COCO retrieval manifests and training exclusion hashes/URLs. |
| [`src/validate.py`](src/validate.py) | Deterministic held-out contrastive validation loss. |
| [`src/eval.py`](src/eval.py), [`src/evaluate.py`](src/evaluate.py) | Multiple-caption retrieval metrics and evaluation CLI. |
| [`src/infer.py`](src/infer.py) | Image-to-caption cosine similarity using exported encoders. |
| [`src/benchmark.py`](src/benchmark.py) | Bounded full-architecture BF16 batch-capacity probe. |
| [`src/plot_history.py`](src/plot_history.py) | Export loss and retrieval curves from persistent logs. |
| [`src/smoke_fixture.py`](src/smoke_fixture.py) | Synthetic infrastructure fixture; never the main training dataset. |
| [`modal_train.py`](modal_train.py), [`config.yaml`](config.yaml) | Modal stages and configurable training settings. |

The original notebook remains in [`experimentation/vlm.ipynb`](experimentation/vlm.ipynb). Its small word-level tokenizer and unfinished prototype text encoder are separate experiments; training uses `src/`.

### Architecture

```text
Images [B,3,224,224]
 → CNN [B,256,28,28] → 4×4 patches [B,49,256]
 → CLS + positions [B,50,256] → 12 pre-norm transformer blocks
 → LayerNorm → CLS pooling → projection [B,256]

CLIP token IDs [B,77]
 → token + positional embeddings → 6 pre-norm causal transformer blocks
 → LayerNorm → first EOS (49407) pooling → projection [B,256]

L2 normalization → global image/text similarity → symmetric multi-positive loss
```

<p align="center">
  <img src="assets/TextTransformer.svg" alt="Text transformer architecture diagram" width="360" />
</p>

Custom attention projections, CNN layers, residual ordering, encoder depths, and projections are retained. Fixes propagate text head/context/width configuration, reject captions without EOS, reject unsupported vision widths, and use stable scaled dot-product attention plus FP32 contrastive logits for BF16 training. The image branch still requires width 256 and 49 patch tokens. BatchNorm remains per GPU, with DDP buffer synchronization; changing GPU count changes training behavior.

## Installation and local verification

Use Python 3.11. Create a fresh virtual environment for the pinned runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest==8.3.5
python -m compileall -q src modal_train.py
python -c "from src.tokenization import tokenize; print(tokenize(['a red square'], cache_dir='data/tokenizer').shape)"
```

The model and data pipeline were previously exercised by local and Modal smoke tests. For this dataset switch, the real 512-image PD12M pilot exercised acquisition, audit and token caching; focused source-adapter checks cover both metadata schemas.

For dataset preparation outside Modal, install `requirements-prep.txt` too. Run modules from the repository root with `python -m src.<module>`.

## Dataset: PD12M

The configured primary source is [PD12M](https://huggingface.co/datasets/Spawning/PD12M), nominally **12,400,094 image–caption records**. Only `metadata/*.parquet` files are downloaded from Hugging Face; pretrained CLIP embedding files are excluded. The metadata revision is pinned for resume. Fields `caption`, `id`, and `hash` are normalized to the loader schema, while original source and image-license metadata are retained.

PD12M uses real images and automatically generated captions. The dataset is licensed under CDLA-Permissive-2.0, and its publisher identifies the images as public-domain/CC0. The imagery and captions differ from DataComp, so faster acquisition does not establish equal retrieval performance. Image downloads use the publisher's public S3 URLs; they are not prebuilt WebDataset tar downloads. Full usable counts are measured after acquisition and deduplication, not assumed from the source row count.

DataComp remains available explicitly with `--dataset datacomp_small`; its existing cache is preserved under `/data/datacomp_small`. PD12M uses `/data/pd12m`, and their shards/checkpoints are never mixed.

Preparation uses bounded parallel downloads, 15-second timeouts, two retries, and incremental shard recovery. Tar auditing decodes images, filters held-out hashes/URLs, deduplicates exact decoded-image/caption pairs using SQLite, and writes memory-mapped offset indices. No tar extraction or image download occurs in the training loop. Run preparation with a single writer; freeze the index/shards during training.

Image identities determine a fixed ~1% internal validation split, keeping alternative captions for the same decoded image together. Training uses random resized crops without left/right flips; validation uses resize + center crop and fixed normalization. Corrupt samples discovered later cause the whole distributed batch/accumulation group to be dropped consistently. Counts are recorded in preparation reports, index `manifest.json`, and training events.

PD12M includes high-resolution originals, so raw transfer volume and stored resized-shard size differ. Actual storage depends on source images and resize settings; no full-corpus size or completion-time estimate is claimed. Reserve room for metadata, SQLite audit, index, archives and checkpoints too. Full training refuses fewer than **3 million audited training pairs** by default. Actual steps per epoch derive from the index count, not the nominal pool size. Remainders smaller than one global batch are dropped, never padded with repeated images.

### Retrieval evaluation

`prepare-eval` downloads the public [COCO val2017 release](https://cocodataset.org/#download), retains original license metadata, and creates a deterministic **custom** split: first 1,000 image IDs for validation and remaining images for final test (4,000 before any exact-image deduplication). This is not the Karpathy split and should not be compared as that benchmark. Every available caption is a valid positive for its image.

Evaluation reports image-to-text and text-to-image Recall@1/5/10, mean/median ranks, and their mean recall. Similarity scores are computed in chunks. Best-model selection maximizes validation mean recall across both directions and all three K values. Exact decoded-image duplicates are merged before splitting. The final test split is never used for model selection. Internal validation contrastive loss uses batch negatives and is not a full-corpus retrieval metric.

Exclusions cover original file hashes, decoded RGB hashes, known source URLs and metadata source hashes before resize/blur. Near duplicates, re-encoded copies with changed pixels, and unknown URLs can still overlap; this is identifiable-overlap filtering, not a guarantee of zero web-scale leakage.

## Modal workflow

Current detached PD12M preparation: [Modal run](https://modal.com/apps/sujalkr451/main/ap-RPS83YL7VPiszxESHuuKdJ). This replaces the stopped DataComp job. The source-adapter checks passed for both datasets. Preparation does not automatically launch GPU training; run the training command after the full index is ready.

Authenticate with `modal setup` if needed. These commands use your configured Modal profile. Create v2 Volumes before first use (if the names already exist, keep the existing volumes):

```bash
modal volume create clip-data --version=2
modal volume create clip-runs --version=2
modal run modal_train.py --stage prepare-eval
modal run modal_train.py --stage prepare-pilot
modal run modal_train.py --stage benchmark
modal run modal_train.py --stage smoke --num-gpus 1
modal run modal_train.py --stage smoke --num-gpus 4
```

`prepare` downloads the full PD12M dataset configured in config.yaml. `prepare-pilot` is an explicitly named 512-record acquisition check; `smoke` performs 10 steps using those **real downloaded image–caption pairs**. Neither runs synthetic training. Synthetic fixtures are not part of the Modal training workflow. `--resume` continues a bounded real-data smoke segment. Use a new output directory for a new run; existing checkpoints require resume.

**Only after explicit approval**, prepare and launch the large-scale run:

```bash
modal run --detach modal_train.py --stage prepare --approve-full-run
modal run --detach modal_train.py --stage train --num-gpus 4 --approve-full-run
```

Wait for complete preparation, inspect `/data/pd12m/index/manifest.json`, and confirm usable counts before launching training. `config.yaml` proposes batch 256/GPU, LR 3e-4, AdamW weight decay 0.1, 2,000 warmup steps, 20 epochs and BF16. Tune against measured data throughput and learning curves. Capacity measurements alone do not establish a good learning rate.

Each invocation stops at a checkpoint boundary after ~23 hours, below Modal's 24-hour function limit. Resume until all configured epochs complete:

```bash
modal run --detach modal_train.py --stage train --num-gpus 4 --approve-full-run --resume
```

There is no automatic indefinite retry loop that can spend credits unattended. Periodic checkpoints limit lost work on interruption. One process owns writes; files close before explicit Volume commit, with a final commit on function exit.

### Local / other GPU servers

Edit paths in a copy of `config.yaml`, prepare the evaluation set, and run:

```bash
python -m src.prepare_evaluation --root data/evaluation
python -m src.prepare_data --dataset pd12m --root data/pd12m --exclusions data/evaluation/exclusions.json
python -m src.train --config local-config.yaml
# or multi-GPU:
torchrun --standalone --nproc_per_node=4 -m src.train --config local-config.yaml
# resume:
torchrun --standalone --nproc_per_node=4 -m src.train --config local-config.yaml --resume runs/clip/checkpoints/last_model.pth
```

Set `dataset.index_dir`, `dataset.tokenizer_cache`, `evaluation.validation_manifest` and `output_dir` to your local paths. CPU verification uses `precision: fp32` and small batches. `main.py` remains an unused placeholder; the executable training entrypoint is `src.train`.

## Distributed objective and resume semantics

Each GPU owns disjoint sample indices. Equal-sized feature batches are gathered with autograd; each rank computes local-anchor/global-candidate image-to-text and text-to-image losses. Autograd collectives sum remote contributions before DDP averages parameter gradients. Repeated image identities or identical truncated token sequences are positives rather than false negatives. Image/caption pair deduplication is exact; perceptual deduplication is not implemented.

Gradient accumulation averages independent microbatch contrastive objectives. It increases samples per optimizer update **without** increasing the negative pool beyond `batch_size_per_gpu × world_size`. Final partial accumulation groups are normalized by their actual size. Checkpoints are saved only at completed optimizer boundaries.

Resumable checkpoints include both encoder state dicts, optimizer, scheduler, epoch, next batch, global step, best metric, configuration, per-rank RNG states, processed samples, index fingerprint and history. BF16 needs no GradScaler (`scaler: null`). Exact resume requires the same training/model configuration, process count and immutable index. Sampler order and per-sample crops reproduce the next batch; recreating workers does not consume the model's dropout RNG. A changed world size requires a new training run rather than silent stream reuse. GPU kernels can still be nondeterministic.

## Checkpoints, logs, evaluation and inference

Persistent files are stored under `clip-runs:/pd12m/checkpoints/`:

```text
best_model.pth           # Resumable best validation model, after first validation
last_model.pth           # Most recent resumable optimizer boundary
epoch_005.pth            # Every fifth completed epoch
epoch_010.pth
best_inference.pth       # Lightweight best validation model
inference.pth            # Lightweight latest encoder weights + model configuration
training_history.json   # Epoch validation history
events.jsonl            # Per-step loss, LR, memory, throughput, counts and saves
tensorboard/            # Training curves and numeric metrics
```

Real-data smoke runs use separate `/pd12m-smoke-1/` and `/pd12m-smoke-4/` directories. Earlier synthetic verification outputs remain historical artifacts only. Smoke checkpoints contain untrained/test weights and are not useful pretrained models.

Download the full best checkpoint and lightweight inference export:

```bash
modal volume get clip-runs pd12m/checkpoints/best_model.pth ./best_model.pth
modal volume get clip-runs pd12m/checkpoints/last_model.pth ./last_model.pth
modal volume get clip-runs pd12m/checkpoints/best_inference.pth ./best_inference.pth
modal volume get clip-runs pd12m/checkpoints/inference.pth ./inference.pth
mkdir -p downloaded
modal volume get clip-runs pd12m/checkpoints downloaded
python -m src.plot_history downloaded/checkpoints
tensorboard --logdir downloaded/checkpoints/tensorboard
```

Only load trusted resumable checkpoints: optimizer/RNG state uses Python pickle. Lightweight inference exports load with `weights_only=True`. Cosine-similarity inference:

```bash
python -m src.infer --weights best_inference.pth --image example.jpg --captions "a red square" "a blue circle"
python -m src.evaluate --weights best_inference.pth --manifest data/evaluation/test.json --output test-results.json
```

Test retrieval should use the selected best model's lightweight export; see the validation report/runbook for artifacts from actual executed runs. Training logs include allocated peak VRAM, NVML utilization when available, images/sec, global contrastive batch, optimizer batch, processed/skipped counts, LR, gradient norms, epoch durations and estimated remaining time. Throughput includes the measured batch interval; initial warmup and checkpoint intervals distort it. Use steady-state medians, and inspect data-loader bottlenecks before scaling.

The PD12M pilot source revision, status counts, timing and index fingerprint are saved on `clip-data` in `/pd12m/preparation-first-1-limit-512.json`. Full preparation writes `/pd12m/preparation-full.json`.
