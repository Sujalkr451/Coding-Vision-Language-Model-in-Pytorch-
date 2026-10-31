<p align="center">
  <img src="assets/vlm-banner.png" alt="Vision–language model architecture" width="100%" />
</p>

# Vision–Language Model in PyTorch

A CLIP-style model trained from scratch with a CNN + vision transformer and a causal text transformer. Uses the `openai/clip-vit-base-patch32` tokenizer; no pretrained model weights.

## Architecture

```text
Images [B,3,224,224]
 → CNN [B,256,28,28] → 4×4 patches [B,49,256]
 → CLS + positions → 12 transformer blocks → CLS pooling → projection [B,256]

Token IDs [B,77]
 → token + positional embeddings → 6 causal transformer blocks
 → first EOS pooling → projection [B,256]

L2 normalization → image/text similarity → symmetric multi-positive loss
```

The vision encoder requires width 256 and 49 patch tokens. Text head count, depth and context length are configurable in `config.yaml`.

<p align="center">
  <img src="assets/TextTransformer.svg" alt="Text transformer architecture" width="360" />
</p>

## Setup

Use Python 3.11 and run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Local dataset preparation also requires `requirements-prep.txt`.

## Data

The default dataset is [PD12M](https://huggingface.co/datasets/Spawning/PD12M): about 12.4 million image–caption records. Metadata is pinned to a revision; images download from the publisher's public S3 bucket. The publisher identifies the images as public-domain/CC0, and the dataset license is CDLA-Permissive-2.0.

Preparation downloads and resizes images, audits tar shards, removes exact image/caption duplicates, filters evaluation overlaps, and caches caption tokens. A fixed image-based split reserves about 1% for internal validation. Training reads indexed tar offsets without extracting archives. Use one preparation writer and keep shards and indices fixed during training.

Full training requires at least 3 million audited training pairs by default. Usable counts are recorded in the index's `manifest.json`. DataComp is also available with `--dataset datacomp_small`; it uses a separate data directory.

Evaluation uses [COCO val2017](https://cocodataset.org/#download), deduplicated by decoded image and split by ascending image ID: 1,000 validation images and the remainder for test. This is a custom split, not the Karpathy benchmark. All captions for an image count as positives. Hash and URL exclusions remove identifiable training overlap; near duplicates can remain.

## Modal

Authenticate with `modal setup`. Create the volumes once:

```bash
modal volume create clip-data --version=2
modal volume create clip-runs --version=2
```

Prepare evaluation data, then check acquisition and training on a 512-record pilot:

```bash
modal run modal_train.py --stage prepare-eval
modal run modal_train.py --stage prepare-pilot
modal run modal_train.py --stage smoke --num-gpus 1
modal run modal_train.py --stage smoke --num-gpus 4
modal run modal_train.py --stage benchmark
```

Smoke runs use downloaded image–caption pairs and stop after 10 steps. Their outputs are stored separately from full training.

Prepare the full dataset, inspect `/data/pd12m/index/manifest.json`, then launch training:

```bash
modal run --detach modal_train.py --stage prepare --approve-full-run
modal run --detach modal_train.py --stage train --num-gpus 4 --approve-full-run
```

Defaults: four H100s, 256 pairs/GPU, BF16, AdamW at 3e-4, 2,000 warmup steps, and 20 epochs. Each invocation stops at a checkpoint boundary after about 23 hours. Resume with:

```bash
modal run --detach modal_train.py --stage train --num-gpus 4 --approve-full-run --resume
```

## Local training

Copy `config.yaml` to `local-config.yaml`. Set `dataset.index_dir`, `dataset.tokenizer_cache`, `evaluation.validation_manifest`, and `output_dir` to local paths.

```bash
python -m src.prepare_evaluation --root data/evaluation
python -m src.prepare_data --dataset pd12m --root data/pd12m --exclusions data/evaluation/exclusions.json
python -m src.train --config local-config.yaml
```

For multiple GPUs and checkpoint resume:

```bash
torchrun --standalone --nproc_per_node=4 -m src.train --config local-config.yaml
torchrun --standalone --nproc_per_node=4 -m src.train --config local-config.yaml --resume runs/clip/checkpoints/last_model.pth
```

Use `precision: fp32` and small batches on CPU.

## Training and checkpoints

DDP gathers features with autograd. Repeated image identities and identical truncated captions count as positives. Gradient accumulation averages microbatch losses; the negative pool remains `batch_size_per_gpu × world_size`. BatchNorm statistics remain per GPU.

Checkpoints include encoder weights, optimizer, scheduler, configuration, per-rank RNG states, sampler position, index fingerprint and history. Resume requires the same configuration, GPU count and immutable index. GPU kernels can still be nondeterministic.

Modal outputs live in `clip-runs:/pd12m/checkpoints/`:

| File | Contents |
| --- | --- |
| `last_model.pth` | Latest resumable checkpoint |
| `best_model.pth` | Best validation checkpoint |
| `epoch_005.pth`, etc. | Checkpoint every five epochs |
| `inference.pth` | Latest encoder weights and model configuration |
| `best_inference.pth` | Best validation model for inference |
| `training_history.json` | Epoch validation metrics |
| `events.jsonl`, `tensorboard/` | Loss, throughput, memory and other training metrics |

Best-model selection uses validation mean Recall@1/5/10 across both retrieval directions. The test split is reserved for final evaluation. Internal contrastive validation loss uses batch negatives rather than full-corpus retrieval.

```bash
mkdir -p downloaded
modal volume get clip-runs pd12m/checkpoints downloaded
python -m src.plot_history downloaded/checkpoints
tensorboard --logdir downloaded/checkpoints/tensorboard
```

Only load trusted resumable checkpoints: optimizer and RNG state use Python pickle. Inference exports load with `weights_only=True`.

## Inference and evaluation

```bash
python -m src.infer --weights downloaded/checkpoints/best_inference.pth --image example.jpg --captions "a red square" "a blue circle"
python -m src.evaluate --weights downloaded/checkpoints/best_inference.pth --manifest data/evaluation/test.json --output test-results.json
```

Retrieval reports include image-to-text and text-to-image Recall@1/5/10 and mean/median ranks.

## Source

- `src/model_arch.py`, `src/model.py`: encoders and paired model.
- `src/prepare_data.py`, `src/dataset.py`, `src/cache_tokens.py`: acquisition, indexing and token caching.
- `src/train.py`, `src/distributed.py`, `src/clip_loss.py`, `src/checkpoint.py`: training and resume.
- `src/prepare_evaluation.py`, `src/validate.py`, `src/eval.py`: evaluation data and metrics.
- `experimentation/vlm.ipynb`: original prototype notebook; training uses `src/`.
