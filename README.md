# MixPro: Reproduction and Analysis
Based on: MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer (ICLR 2023)

## 1. Abstract
This project focuses on reproducing the MixPro data augmentation strategy for Vision Transformers (ViTs). MixPro addresses limitations in prior Mixup-based methods (like CutMix and TransMix) by introducing two novel components: MaskMix (image space) and Progressive Attention Labeling (PAL) (label space). This repository contains a PyTorch implementation, reproduction results on reduced subset of the imageNet1k dataset , a critique of the methodology, and a proposed improvement using Entropy-based confidence.

## 2. Background & Motivation
Vision Transformers (ViTs) generally require massive datasets to generalize well. While data augmentation techniques like CutMix and TransMix have helped, they suffer from specific issues when applied to ViTs:

  -  Image Space Deficit: Methods like CutMix use region-based cropping (a large rectangular block). This destroys the global context that ViTs rely on for self-attention.

  -  Label Space Noise: TransMix uses the model's attention map to determine label mixing ratios. However, early in training, attention maps are unreliable, leading to noisy label assignments.

## 3. Methodology

We adhered to the algorithmic structure defined in the original paper while refactoring the codebase for Distributed Data Parallel (DDP) support and robust logging.

### 3.1. MaskMix (Image Space)
Instead of a single large crop, MaskMix uses a grid-mask strategy.
* **Logic:** A binary mask $M$ mixes two images $x_i$ and $x_j$: $\tilde{x} = M \odot x_i + (1-M) \odot x_j$.
* **Constraint:** The mask patch size ($P_{mask}$) is a multiple (e.g., $4\times$) of the ViT input patch size. This ensures every token processed by the ViT comes from exactly one image.

### 3.2. Progressive Attention Labeling (PAL) (Label Space)
PAL solves the "unreliable attention" problem by introducing a dynamic weight $\alpha$.
* **The Formula:**
    $$\lambda = \alpha \cdot \lambda_{attn} + (1-\alpha) \cdot \lambda_{area}$$
* **Mechanism:** $\alpha$ measures model confidence.
    * **Low Confidence (Early Training):** $\alpha \to 0$. We trust the pixel area ($\lambda_{area}$).
    * **High Confidence (Late Training):** $\alpha \to 1$. We trust the attention map ($\lambda_{attn}$).

## 4. Experimental Setup: The "Stress Test"

To evaluate the robustness of MixPro, we devised a **Transfer Learning Stress Test**. We deviated from the paper's "from-scratch" setup to test if MixPro provides value when fine-tuning pre-trained models on smaller datasets.

| Feature | Original Paper Setup | Our Reproduction (Stress Test) | Justification |
| :--- | :--- | :--- | :--- |
| **Dataset** | ImageNet-1k (1.28M images) | ImageNet Subset | Resource constraints & Data Efficiency test. |
| **Epochs** | 300 Epochs | 150 Epochs | Sufficient for fine-tuning convergence. |
| **Initialization** | **Random (From Scratch)** | **Pre-trained (ImageNet)** | Investigating transfer learning robustness. |
| **Backbone** | DeiT-Small / DeiT-Tiny | DeiT-Small / DeiT-Tiny | Consistent architectural comparison. |

## 3. Implementation Details
We reproduced the method using PyTorch and timm.
Model: deit_tiny_patch16_224 / deit_small
Dataset: ImageNet1k (reduced)
Hyperparameters:
  -  Optimizer: AdamW
  -  Learning Rate: 0.001 (cosine decay)
  -  Epochs: 300 (with 20-epoch warm-up)
  -  MaskMix Scale: 4x
  -  Mixing Probabilities: MixUp (0.8), CutMix (1.0), MixPro (1.0)

## 4. Directory Structure
```
MIX-PRO-REPRO/
├── configs/
│   ├── deit_s_baseline.yaml
│   ├── deit_s_mixpro.yaml
│   ├── deit_t_baseline.yaml
│   ├── deit_t_mixpro.yaml
├── data/
│   ├── imagenet1k/
│   │   ├── train/
│   │   ├── val/
│   ├── imagenet1k_subset_100pc/
│   │   ├── train/
│   │   ├── val/
│   ├── raw/
│   │   ├── ILSVRC2012_devkit_t12.tar.gz
│   │   ├── ILSVRC2012_img_train.tar
│   │   ├── ILSVRC2012_img_val.tar
│   ├── tmp/
├── results/
├── scripts/
│   ├── prepare_imagenet.py     #extracts the images from the zipped tar files
│   ├── tmp/
├── src/
│   ├── data/
│   ├── methods/
│   │   ├── maskmix.py    # Mask generation logic
│   │   └── pal.py        # Progressive Attention Labeling loss
│   └── models/           # ViT wrappers to extract Attention Maps
│   │   ├── deit_s.py
│   │   ├── deit_t.py
├── train_baseline.py              # Main training loop
├── train_mixpro.py
└── README.md
```

## 2. Environment Setup

```bash
git clone github repo
cd mix-pro-repro

# (Optional) create conda env
conda create -n mixpro-repro python=3.10 -y
conda activate mixpro-repro

# Install requirements
pip install -r requirements.txt

#To run baseline:
CUDA_VISIBLE_DEVICES=<No of GPUs> torchrun --standalone --nproc_per_node= <No of GPUs>\
  -m src.train_baseline --config configs/deit_s_baseline.yaml --out results

#To run mixpro:
CUDA_VISIBLE_DEVICES=<No of GPUs> torchrun --standalone --nproc_per_node= <No of GPUs>\
  -m src.train_mixpro --config configs/deit_s_mixpro.yaml --out results

**To run deit_t, substitute with deit_t yaml file**

```
## 8. References
[1] Zhao, Q., et al. "MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer." ICLR 2023. arXiv:2304.12043

[2] Chen, J.-N., et al. "TransMix: Attend to Mix for Vision Transformers." CVPR 2022.

[3] Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021.
