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

A learning project for implementing the pieces of a vision–language model in PyTorch. The current work explores a lightweight convolutional image encoder, the beginnings of multi-head attention, and a synthetic dataset of colored shapes paired with captions.

The intended direction is a CLIP-style model that maps images and text into a shared embedding space. **The full model and training pipeline are not implemented yet.**

## What's in the repo

| File | Current content |
| --- | --- |
| [`src/model.py`](src/model.py) | `ImageFeatureMap`: standard, depthwise, and pointwise convolutions. `MemeTransformer`: patch embedding and partial Q/K/V attention code. |
| [`experimentation/vlm.ipynb`](experimentation/vlm.ipynb) | Synthetic image generation, image–caption dataset, vocabulary construction, and text tokenization. |
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

This describes the intended pipeline. The source currently contains the convolutional layers and an attention scaffold; the text encoder, shared projections, loss, and training loop still need to be built.

### Image encoder work

`ImageFeatureMap` starts with a 3 → 32 channel convolution, then uses depthwise and pointwise convolutions to reach 256 channels. `MemeTransformer` defines a 4 × 4 patch convolution, Q/K/V linear projections, and head reshaping with defaults of 1,024 embedding dimensions and 16 heads.

These components are still under construction: `ImageFeatureMap.forward` does not return its output, and `MemeTransformer` does not yet compute attention or return a result. Its positional embedding also needs a floating-point initialization, and the patch features need to be connected to the attention input.

### Synthetic image–caption data

The notebook generates **243 pairs** from every combination of:

- **9 colors:** red, green, blue, yellow, purple, orange, pink, brown, gray.
- **3 shapes:** square, circle, triangle.
- **9 positions:** left, center, right, top, bottom, and the four corners.

Images are drawn at **64 × 64** pixels using Pillow and converted to channel-first float tensors scaled to `[0, 1]`. Captions such as `red circle top-left` are split into words and prepended with a `[CLS]` token. No external dataset download is needed for this experiment.

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
- [x] Define the convolutional image feature layers.
- [ ] Finish and validate the image encoder's forward pass.
- [ ] Complete patch tokenization, positional embeddings, and multi-head attention.
- [ ] Implement a text encoder and shared embedding projections.
- [ ] Add contrastive loss and a training loop.
- [ ] Evaluate image–text matching and retrieval.

## Contributing

This is an evolving learning project. Issues explaining bugs, tensor shapes, or implementation ideas are welcome. For code changes, describe what changed and how you checked it.
