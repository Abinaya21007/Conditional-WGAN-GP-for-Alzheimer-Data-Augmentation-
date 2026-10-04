# Conditional-WGAN-GP-for-Alzheimer-Data-Augmentation-

A conditional **WGAN-GP** (Wasserstein GAN with Gradient Penalty) that generates synthetic brain MRI images to balance imbalanced, multi-class Alzheimer's disease datasets — complete with an automated quality-validation suite and Grad-CAM–based explainability.

---

## Overview

Alzheimer's MRI datasets are typically imbalanced — far more "Non-Demented" scans than "Moderate Demented" ones. This project trains a class-conditional WGAN-GP to generate synthetic MRI images for each disease stage, so downstream classifiers can train on balanced data without needing more real patient scans.

**Classes:** Non-Demented · Very Mild Demented · Mild Demented · Moderate Demented

---

## Features

### 🧠 Core GAN
- Class-conditional **WGAN-GP** — Generator + Critic, trained on 64×64 grayscale MRI scans
- Critic uses **GroupNorm** (not BatchNorm) — required for a mathematically valid gradient penalty, and the fix that took training from diverging losses to stable convergence
- Trained on 10,496 real MRI images across 4 classes

### ⚡ On-demand generation
- Load the trained Generator once, generate any number of new images per class afterward — **no retraining required** per request
- Interactive UI (Colab `ipywidgets`) — enter a count, click generate, get saved output

### ✅ Quality validation
- **Critic-score realism %** — scores generated batches against two reference points: real images (100%) and pure random noise (0%), giving an interpretable confidence number per class
- **Auto-retry loop** — if a class scores below threshold, automatically regenerates with fresh noise (up to 3 attempts) before flagging it
- **Diversity / mode-collapse check** — confirms generated images vary from each other, not just repeating one pattern
- **Duplicate / memorization check** — nearest-neighbor pixel-distance comparison against real training images, to confirm the GAN is generating novel images rather than copying training data
- **Pixel statistics check** — compares brightness/contrast distributions between real and generated images per class

### 🔍 Explainability
- **Critic Grad-CAM** — visualizes which region of a generated MRI most influenced the Critic's "this looks real" judgment, the same underlying technique used for classifier explainability, applied here to the GAN's Critic

### 🔁 On-demand fine-tuning
- If a class consistently fails the quality threshold even after retries, the system asks the user whether to fine-tune the GAN further — confirming first before spending compute, then resuming training from the existing checkpoint rather than starting over

---

## Tech Stack

- **PyTorch** — model architecture, training loop, autograd-based gradient penalty
- **NumPy / PIL** — image preprocessing
- **Matplotlib** — visualization (sample grids, Grad-CAM overlays)
- **ipywidgets** — interactive Colab UI
- **Google Colab + Drive** — training environment and persistent checkpoint storage

---

## Architecture

**Generator:** noise vector (100-dim) + class label → `Linear` → 4 × `ConvTranspose2d` (upsampling, BatchNorm, ReLU) → `Tanh` → 64×64 image

**Critic:** image + class label → 4 × `Conv2d` (downsampling, GroupNorm, LeakyReLU) → `Linear` → realness score

**Loss:** WGAN-GP — `critic_loss = fake_score - real_score + λ·gradient_penalty`, `generator_loss = -fake_score`
`λ_gp = 10`, `n_critic = 5`, Adam (`lr=0.0001`, `betas=(0.5, 0.9)`)

---

## Results

| Check | Result |
|---|---|
| Training stability | Converged smoothly over 20 epochs after BatchNorm → GroupNorm fix (previously diverging) |
| Mode collapse | Not observed — fake/real diversity ratio 0.84–1.07 across all classes |
| Memorization | No meaningful duplication of training images detected |
| Realism (Critic score) | 50–80% of real-image confidence across classes, depending on batch |

---

## What's Included

- [x] Conditional WGAN-GP training pipeline
- [x] GAN quality validation suite (realism scoring, diversity, duplication checks)
- [x] On-demand generation with an interactive UI
- [x] Explainability (Critic Grad-CAM)
- [x] Automatic quality-gated retry + on-demand fine-tuning

This module is designed to plug into downstream classification or federated-learning pipelines as a drop-in data augmentation component — synthetic images are generated per class, on demand, from a single trained checkpoint.

---

## Setup

1. Mount Google Drive in Colab and place the dataset under:
   ```
   /Alzhiemer_GAN_federated/Alzhiemer's_disease/{class_name}/
   ```
2. Run the training script to train (or resume) the WGAN-GP — checkpoints save automatically to Drive
3. Run the generation script for on-demand synthetic image generation with built-in quality checks

---

## Acknowledgments

Dataset: Alzheimer's MRI (4-class) data, based on the structure used in ADNI-style Alzheimer's classification datasets.
