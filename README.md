# Image Captioning with a CNN–RNN Architecture on MS COCO

A deep learning pipeline that generates natural-language captions for images. The model couples a **ResNet-50** visual encoder with an **LSTM** language decoder, trained end-to-end on the [MS COCO](https://cocodataset.org/) dataset. An auxiliary multi-label object classifier is attached to the visual features as additional supervision to sharpen caption relevance.

Built for **Advanced Machine Learning (Spring 2026), Assignment 2** — German International University of Applied Sciences (GIU), Informatics and Computer Science.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Task](https://img.shields.io/badge/Task-Image%20Captioning-success)
![Dataset](https://img.shields.io/badge/Dataset-MS%20COCO-orange)

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Results](#results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Methodology](#methodology)
- [Configuration](#configuration)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

Image captioning asks a model to solve two coupled problems at once: **recognizing** what is in a scene, and **describing** it in fluent language. This project tackles both with the classic encoder–decoder paradigm:

- **Encoder (the "eye"):** a pretrained, frozen ResNet-50 compresses each image into a 2048-dimensional feature vector that summarizes its visual content.
- **Decoder (the "voice"):** an LSTM generates a caption one word at a time, conditioned on the encoded image.
- **Auxiliary head:** a multi-label classifier predicts which of the 80 COCO object categories appear in the image. This extra objective grounds the visual features in real object semantics and acts as a regularizer.

The pipeline is designed to be trainable on modest hardware (a 4 GB laptop GPU) by **precomputing and caching CNN features to disk**, so the frozen ResNet only runs once instead of every epoch.

---

## Team

| Name | Student ID |
|------|------------|
| Ahmed Ramy | 13002184 |
| Saif Saad | 14004494 |
| Ahmed Amr | 13007323 |

---

## Results

Evaluated with **BLEU-1 through BLEU-4** using beam search (beam width = 5). The test set was held out entirely and scored only once.

| Metric | Validation (4,967 imgs) | Test (2,484 imgs) |
|--------|:-----------------------:|:-----------------:|
| BLEU-1 | 66.39 | **66.17** |
| BLEU-2 | 49.19 | **48.55** |
| BLEU-3 | 36.55 | **36.04** |
| BLEU-4 | 26.93 | **26.60** |

The close agreement between validation and test scores indicates the model generalizes well and is not overfit to the validation set. A BLEU-4 of ~27 is competitive for a single-layer LSTM decoder without an attention mechanism.

---

## Architecture

```
                  ┌──────────────────────────┐
   Input image ──▶│  ResNet-50 (frozen)      │──▶ 2048-d feature
                  │  ImageNet weights,        │        │
                  │  final FC layer removed   │        ├──────────────┐
                  └──────────────────────────┘        │              │
                                                       ▼              ▼
                                          ┌────────────────┐  ┌────────────────────┐
                                          │ Encoder        │  │ Auxiliary classifier│
                                          │ Linear 2048→256│  │ 2048→512→80 (BCE)   │
                                          │ BatchNorm+ReLU │  └────────────────────┘
                                          └────────────────┘   (object categories)
                                                       │
                       feature injected at timestep 0  ▼
   <start> w1 w2 ... ──▶ Embedding(256) ──▶ ┌──────────────────┐ ──▶ Linear 512→vocab ──▶ next word
                                            │ LSTM (hidden 512)│
                                            └──────────────────┘
```

| Component | Specification |
|-----------|---------------|
| **Encoder backbone** | ResNet-50, pretrained on ImageNet, frozen (FC layer removed) → 2048-d |
| **Feature projector** | Linear 2048 → 256, BatchNorm1d, ReLU, Dropout(0.5) |
| **Word embedding** | `nn.Embedding(6637, 256, padding_idx=0)` |
| **Decoder** | Single-layer LSTM, hidden dim 512, `batch_first=True` |
| **Output layer** | Linear 512 → 6637 (vocabulary size) |
| **Auxiliary head** | Linear 2048 → 512 → 80, Dropout(0.3), `BCEWithLogitsLoss` |
| **Combined loss** | `L = L_caption + 0.2 · L_auxiliary` |
| **Trainable parameters** | ~8.3 million (ResNet frozen) |

The image feature is fed to the LSTM as the **first input token** at timestep 0; words follow at subsequent timesteps. Training uses **teacher forcing** (ground-truth words are fed back rather than the model's own predictions), with `pack_padded_sequence` for efficiency and gradient clipping (max norm 5.0) for stability.

---

## Repository Structure

```
.
├── image_captioning_v3.ipynb     # Main notebook (end-to-end pipeline)
├── README.md
├── requirements.txt
│
├── data/                         # Dataset — NOT committed (see .gitignore)
│   ├── train2014/
│   │   └── train2014/            # Training images (.jpg)
│   └── captions/
│       └── annotations/
│           ├── captions_train2014.json
│           └── instances_train2014.json
│
├── features/                     # Precomputed CNN features — NOT committed
└── checkpoints/                  # Saved model weights — NOT committed
```

> **Note:** `data/`, `features/`, and `checkpoints/` are excluded from version control because they are large (the dataset alone is ~13 GB). Only the notebook, README, and requirements file are tracked. See the suggested `.gitignore` below.

<details>
<summary>Suggested <code>.gitignore</code></summary>

```gitignore
# Dataset & generated artifacts
data/
features/
checkpoints/
*.npy
*.pt
*.pth

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
venv/

# OS
.DS_Store
Thumbs.db
```
</details>

---

## Dataset Setup

1. Download the MS COCO dataset from Kaggle: **[hariwh0/ms-coco-dataset](https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset)**
2. Extract it so the folder layout matches the structure under `data/` shown above.
3. If your folder names differ, update the paths in the `Config` class at the top of the notebook (`TRAIN_IMG_DIR`, `CAPTIONS_FILE`, `INSTANCES_FILE`).

The full training set contains **82,783 images** with **414,113 captions** (~5 human-written captions per image). By default the notebook samples **60%** of this data — see [Methodology](#methodology).

---

## Installation

Requires Python 3.10+ and (recommended) a CUDA-capable GPU.

```bash
# Clone the repository
git clone https://github.com/AhmedRashed2024/<repo-name>.git
cd <repo-name>

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code> contents</summary>

```txt
torch
torchvision
nltk
pycocotools
matplotlib
tqdm
pillow
scikit-learn
numpy
```

For GPU acceleration, install the CUDA build of PyTorch that matches your driver — see the [official selector](https://pytorch.org/get-started/locally/).
</details>

---

## Running the Notebook

Open `image_captioning_v3.ipynb` in Jupyter or VS Code and run the cells top to bottom. The pipeline proceeds in these stages:

1. **Imports & configuration** — all hyperparameters live in the `Config` class.
2. **Data loading** — parse COCO annotations into lookup dictionaries.
3. **Subsetting & splitting** — sample 60% of the data, split into train / val / test.
4. **Caption preprocessing** — clean text, build vocabulary, numericalize and pad.
5. **CNN feature extraction** — run ResNet-50 once, cache 2048-d features to disk.
6. **Auxiliary labels** — build 80-class multi-hot object labels from instance annotations.
7. **Model definition** — encoder, decoder, auxiliary classifier.
8. **Training** — joint caption + auxiliary loss for 20 epochs.
9. **Qualitative inspection** — sample generated captions vs. ground truth.
10. **Evaluation** — BLEU-1 to BLEU-4 on validation and test sets.
11. **Inference utility** — `caption_new_image()` captions any image file.

> **First run is slower:** feature extraction over the subset takes time but only happens once. Subsequent runs load the cached features from `features/` and skip straight to training.

To caption your own image after training:

```python
caption_new_image("path/to/your/image.jpg", model, vocab, device=cfg.DEVICE)
```

---

## Methodology

### Data split strategy

The Kaggle distribution of MS COCO does not include the original `val2014` / `test2014` annotation files, so we construct our own three-way split from the training set:

| Split | % of subset | ~Images | Purpose |
|-------|:-----------:|:-------:|---------|
| **Train** | 85% | ~42,200 | Model training |
| **Validation** | 10% | ~4,960 | Tuning / transparent reporting |
| **Test** | 5% | ~2,480 | Final evaluation (scored once) |

Because the subset is 60% of the full data, training uses `0.60 × 0.85 ≈ 51%` of the full COCO training set — satisfying the assignment's **≥ 50% requirement** while still reserving genuine held-out sets.

### Preventing data leakage

- The **vocabulary is built from training captions only.** Including val/test words would let the model implicitly "know" vocabulary from data it should never have seen.
- The val and test splits are **disjoint** from training and are never used for gradient updates or model selection.

### Caption preprocessing

Captions are lowercased, stripped of punctuation (apostrophes kept), and collapsed to single spaces. The vocabulary keeps words appearing **≥ 5 times** (6,637 words from 18,685 unique tokens); rarer words map to `<unk>`. Four special tokens are reserved:

| Token | Index | Role |
|-------|:-----:|------|
| `<pad>` | 0 | Pads captions to equal length within a batch |
| `<start>` | 1 | Signals the decoder to begin |
| `<end>` | 2 | Signals the decoder to stop |
| `<unk>` | 3 | Replaces out-of-vocabulary words |

### Image preprocessing

Each image is resized (shorter edge → 256 px), center-cropped to 224×224, converted to a tensor, and normalized with ImageNet channel statistics (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`) to match the distribution ResNet-50 was pretrained on.

### Inference

Two decoding strategies are implemented:

- **Greedy decoding** — picks the highest-probability word at each step.
- **Beam search** (width 5) — keeps multiple candidate sequences and produces noticeably more fluent captions.

---

## Configuration

Key hyperparameters (all in the `Config` class):

| Setting | Value | Notes |
|---------|:-----:|-------|
| `SUBSET_FRACTION` | 0.60 | Fraction of full dataset used |
| `IMG_SIZE` | 224 | ResNet-50 input resolution |
| `MIN_WORD_FREQ` | 5 | Vocabulary frequency cutoff |
| `MAX_CAPTION_LEN` | 25 | Max tokens per caption (incl. special tokens) |
| `EMBED_DIM` | 256 | Word embedding dimension |
| `HIDDEN_DIM` | 512 | LSTM hidden state size |
| `NUM_LSTM_LAYERS` | 1 | LSTM depth |
| `DROPOUT` | 0.5 | Decoder dropout |
| `AUX_LOSS_WEIGHT` | 0.2 | Weight (λ) on auxiliary loss |
| `BATCH_SIZE` | 64 | Reduce to 32 if you hit OOM |
| `NUM_EPOCHS` | 20 | Training epochs |
| `LEARNING_RATE` | 3e-4 | Adam optimizer |
| `GRAD_CLIP` | 5.0 | Gradient clipping max norm |
| `BEAM_SIZE` | 5 | Beam search width |

**Training hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB VRAM), CUDA. Each epoch takes ≈ 1 min 24 s once features are cached. Mixed-precision (FP16) is used during feature extraction to save memory.

---

## Limitations & Future Work

- **No attention mechanism.** The decoder sees a single fixed feature vector, losing spatial information. Adding Bahdanau or Luong attention would let it focus on different image regions per word — the single biggest expected improvement.
- **Generic captions.** The model favors common phrasings (e.g. "a man on a skateboard") and struggles with unusual scenes.
- **Frozen encoder.** Fine-tuning the last few ResNet blocks could yield captioning-specific visual cues.
- **Teacher-forcing gap.** Scheduled sampling could reduce the train/inference mismatch.
- **Transformer decoder.** A Transformer-based decoder would likely outperform the LSTM, especially on longer captions.

---

## Acknowledgments

- **Course:** Advanced Machine Learning, Spring 2026 — German International University of Applied Sciences (GIU)
- **Instructor:** Dr. Caroline Sabty
- **Teaching Assistants:** Merna Said, Nouran Khaled
- **Dataset:** [MS COCO — Common Objects in Context](https://cocodataset.org/)
- **Backbone:** ResNet-50 (He et al., 2015), pretrained on ImageNet via `torchvision`

---

*This repository is coursework submitted for academic evaluation. Plagiarism is not tolerated.*
