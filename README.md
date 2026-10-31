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

## About

A learning project for implementing the pieces of a vision–language model in PyTorch. The current work includes convolutional image features, patch tokenization, custom multi-head self-attention, an MLP, and a transformer encoder block, alongside a synthetic dataset of colored shapes paired with captions.

The intended direction is a CLIP-style model that maps images and text into a shared embedding space. **The full model and training pipeline are not implemented yet.**

## What's in the repo

| File | Current content |
| --- | --- |
| [`src/model.py`](src/model.py) | `ImageFeatureMap`, `PatchEmbedding`, `MultiHeadAttention`, `MLP`, and a pre-norm `TransformerEncoder` block with residual connections. |
| [`experimentation/vlm.ipynb`](experimentation/vlm.ipynb) | Synthetic dataset, vocabulary and tokenization, train/validation loaders, a CNN image encoder prototype, a text encoder scaffold, and an image features → patches → attention experiment. |
| [`src/dataloader.py`](src/dataloader.py) | Initial PyTorch data-loading imports. |
| [`src/clip_loss.py`](src/clip_loss.py) | Empty placeholder for contrastive loss. |
| [`src/train.py`](src/train.py) | Empty placeholder for training. |
| [`src/eval.py`](src/eval.py) | Empty placeholder for evaluation. |
| [`main.py`](main.py) | Empty entry-point placeholder. |

## The model idea

```text
Image → Convolutional features → Patch tokens → Image encoder ─┐
                                                             ├→ Shared embeddings → Contrastive loss
Caption → Token IDs → Text encoder ───────────────────────────┘
```

This describes the intended pipeline. The image-side building blocks now have forward passes, but still need to be assembled into a complete encoder with positional information and a pooled image embedding. The text encoder, paired projection heads, contrastive loss, and training loop remain unfinished.

### Image encoder work

`ImageFeatureMap` starts with a 3 → 32 channel convolution, then uses depthwise and pointwise convolutions to reach 256 channels. Its forward pass returns the feature map. `PatchEmbedding` applies a 4 × 4 convolution with stride 4 and converts the result into `[batch, tokens, 256]` sequences.

`MultiHeadAttention` implements Q/K/V projections, scaled dot-product attention, attention dropout, head concatenation, and an output projection with dropout. The defaults are **256 embedding dimensions and 4 heads**.

`MLP` expands the embedding dimension by a factor of 4, applies GELU, and projects back to 256 dimensions. `TransformerEncoder` combines pre-attention and pre-MLP layer normalization with residual connections around both sublayers.

For a **224 × 224** input, the default components have these tensor shapes:

```text
[B, 3, 224, 224]
  → ImageFeatureMap    → [B, 256, 28, 28]
  → PatchEmbedding     → [B, 49, 256]
  → TransformerEncoder → [B, 49, 256]
```

The notebook also contains an earlier CNN image encoder prototype that produces normalized 64-dimensional embeddings. Its `TextEncoder` currently declares layers but leaves `forward` unimplemented.

### Synthetic image–caption data

The notebook generates **243 pairs** from every combination of:

- **9 colors:** red, green, blue, yellow, purple, orange, pink, brown, gray.
- **3 shapes:** square, circle, triangle.
- **9 positions:** left, center, right, top, bottom, and the four corners.

Images are drawn at **64 × 64** pixels using Pillow and converted to channel-first float tensors scaled to `[0, 1]`. Captions such as `red circle top-left` are split into words and prepended with a `[CLS]` token. No external dataset download is needed for this experiment.

The notebook splits the dataset into **194 training pairs and 49 validation pairs**, with a batch size of 12. Its separate attention experiment uses random 224 × 224 inputs.

## Explore locally

Use a Python environment compatible with PyTorch. There is no pinned dependency file yet.

```bash
git clone https://github.com/Sujalkr451/Coding-Vision-Language-Model-in-Pytorch-.git
cd Coding-Vision-Language-Model-in-Pytorch-
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pillow matplotlib jupyter
jupyter notebook experimentation/vlm.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Run the notebook cells in order to generate the dataset and inspect the image tensors, captions, and vocabulary. The notebook selects Apple MPS when available and otherwise uses the CPU.

There is no training command, trained checkpoint, or evaluation result yet; running `main.py` currently does nothing.

## Next steps

- [x] Generate synthetic shapes with matching captions.
- [x] Build a vocabulary and encode captions.
- [x] Create train/validation splits and data loaders in the notebook.
- [x] Implement the convolutional feature map and its forward pass.
- [x] Implement patch tokenization and custom multi-head self-attention.
- [x] Add a GELU MLP and a pre-norm transformer encoder block with residual connections.
- [ ] Add positional embeddings and assemble the complete image encoder.
- [ ] Implement a text encoder and shared embedding projections.
- [ ] Add contrastive loss and a training loop.
- [ ] Evaluate image–text matching and retrieval.

## Contributing

This is an evolving learning project. Issues explaining bugs, tensor shapes, or implementation ideas are welcome. For code changes, describe what changed and how you checked it.
