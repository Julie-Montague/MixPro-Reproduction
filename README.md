# MixPro: Reproduction and Analysis
Base Paper: MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer (ICLR 2023)

## 1. Abstract
This project focuses on reproducing the MixPro data augmentation strategy for Vision Transformers (ViTs). We conduct a partial reproduction under limited compute using a deterministically constructed subset of the original dataset. Rather than aiming for exact headline scores, we test whether the relative performance trends hold under Transfer Learning conditions.

MixPro addresses limitations in prior Mixup-based methods by introducing two novel components: MaskMix (image space) and Progressive Attention Labeling (PAL) (label space). This repository contains a PyTorch implementation, reproduction results on a subset of ImageNet-1k, a scientific critique of the methodology, and a proposed improvement using Entropy-based confidence.

## 2. Background & Motivation
Vision Transformers (ViTs) generally require massive datasets to generalize well. While data augmentation techniques like CutMix and TransMix have helped, the authors identify specific issues when applied to ViTs:

  -  Image Space Deficit: Methods like CutMix use region-based cropping (a large rectangular block). This destroys the global context that ViTs naturally rely on for self-attention.

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

To evaluate the robustness of MixPro, we devised a Transfer Learning Stress Test. We deviated from the paper's "from-scratch" setup to test if MixPro provides value when fine-tuning pre-trained models on smaller datasets.

| Feature | Original Paper Setup | Our Reproduction (Stress Test) | Justification |
| :--- | :--- | :--- | :--- |
| **Dataset** | ImageNet-1k (1.28M images) | ImageNet Subset | Resource constraints & Data Efficiency test. |
| **Epochs** | 300 Epochs | 100 Epochs | Sufficient for fine-tuning convergence. |
| **Initialization** | **Random (From Scratch)** | **Pre-trained (ImageNet)** | Investigating transfer learning robustness. |
| **Backbone** | DeiT-Small / DeiT-Tiny | DeiT-Small / DeiT-Tiny | Consistent architectural comparison. |

## 5. Experimental Results: The "Stress Test"

We evaluated the models in a **Transfer Learning** regime (Pre-trained).To provide a comprehensive evaluation, we utilized a split-baseline strategy:

  -  DeiT-Small: Compared against a Standard Baseline (No Augmentation) to measure raw transfer capability.
  -  DeiT-Tiny: Compared against a Strong Baseline (CutMix+) to test competitiveness against SOTA.

### 5.1 Quantitative Results (Top-1 Accuracy)

| Model | Baseline Type | Baseline Acc | MixPro (Ours) | TransMix | Best Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeiT-Small** | Standard (No Aug) | 67.82% | 75.65% | **77.42%** | TransMix |
| **DeiT-Tiny** | Strong (CutMix+) | 68.00% | **70.77%** | 69.90% | **MixPro** |

### 5.2 Critical Analysis

Our results reveal a distinct interaction between Model Capacity and Augmentation Strategy, highlighting a limitation in the MixPro methodology when applied to Transfer Learning.

1.  **DeiT-Tiny Favors MixPro:**
    For the capacity-constrained Tiny model, **MixPro** achieved the highest accuracy (**70.77%**).
    *  Smaller models benefit significantly from the aggressive regularization provided by **MaskMix** (Grid Masking). The paper notes MixPro performs better on models with fewer parameters. By forcing the model to reconstruct global context from scattered patches, MixPro prevents overfitting, a benefit that outweighs the simpler CutMix strategy.

2.  **DeiT-Small Favors TransMix:**
    For the larger Small model, **TransMix** achieved the highest accuracy (**77.42%**).
    *  This contradicts the "unreliable attention" premise of the paper. Because we utilized a Pre-trained Backbone, the attention maps were reliable from Epoch 1.
    *  MixPro's PAL mechanism forces the model to ignore these high-quality attention maps (via $\alpha$) early in training. TransMix, which trusts attention immediately, leveraged the pre-trained priors more effectively.

## 6. Proposed Improvement: Entropy-Based PAL

The original PAL calculates confidence ($\alpha$) using Cosine Similarity between logits and the Ground Truth label13. This creates an artificial dependency where the model "peeks" at the label to determine confidence. We propose an Entropy-based Confidence measure.
A model should determine confidence based on the sharpness of its own predictions (Entropy), independent of the label.

**Formula:**
$$\alpha = 1 - \frac{Entropy(p)}{MaxEntropy}$$

## 7. Conclusion:
1. Verification of Model Capacity Claim (Successful Reproduction) The original paper states: "In particular, MixPro has better performance on models with fewer parameters".
  -  On the capacity-constrained DeiT-Tiny (5M params), MixPro (70.77%) outperformed both the Baseline and TransMix.
  -  This Confirmed the paper's claim that MaskMix's regularization is most critical for lightweight models.

2. Verification of the "Unreliable Attention" Mechanism (Negative Verification) The paper's central premise is that MixPro is necessary because "at the early stage of training, the model produces unreliable attention maps".
   -  By using a Pre-trained Backbone, we artificially removed the "unreliable attention" problem. When attention maps were reliable from the start (DeiT-Small), TransMix (which trusts attention immediately) outperformed MixPro.
   -  This validates the authors' mechanism by proving the inverse: MixPro's PAL mechanism is indeed designed specifically for scenarios where attention is noisy (training from scratch). When that condition is removed, the method's advantage disappears, exactly as the theory predicts.
  
**Our results confirm that MixPro is a highly effective "cold-start" optimizer, particularly for small models, but is functionally redundant when applying Transfer Learning to larger, pre-converged backbones.**


## 7. Directory Structure
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

## 8. Environment Setup

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
## 9. References
[1] Zhao, Q., et al. "MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer." ICLR 2023. arXiv:2304.12043

[2] Chen, J.-N., et al. "TransMix: Attend to Mix for Vision Transformers." CVPR 2022.

[3] Touvron, H., et al. "Training data-efficient image transformers & distillation through attention." ICML 2021.
