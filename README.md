# QR-Code-SR-GAN (v2.0): Robust QR Code Restoration via Adversarial Learning and Attention Mechanisms

**v2.0 New Features:** Integration of **ESRGAN** architecture, **SE (Squeeze-and-Excitation) Attention** mechanisms, and **Adversarial Training** (GAN) to fundamentally resolve edge blurring and artifacts present in v1.0.

This repository hosts the v2.0 implementation of the QR Code Super-Resolution restoration network. Moving beyond the standard residual learning of v1.0, v2.0 introduces a complete Generative Adversarial Network (GAN) pipeline aimed at reconstructing high-frequency details—such as **Finder Patterns** and **Alignment Patterns**—which are critical for successful decoding.

------

## v2.0 vs. v1.0: Key Improvements

The primary objective of v2.0 is to overcome the "smoothing effect" caused by the pixel-wise loss functions (L1/MSE) used in v1.0.

| **Feature**           | **v1.0 (Original)**              | **v2.0 (Proposed)**                  | **Improvement Analysis**                                     |
| --------------------- | -------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **Architecture**      | Standard ResNet (SRResNet)       | **RRDB + SE-Attention**              | RRDB increases network capacity; SE modules focus on texture channels over background noise. |
| **Training Paradigm** | Supervised (L1 Loss)             | **Adversarial (GAN)**                | The Discriminator forces the Generator to produce sharp edges unattainable by L1 loss alone. |
| **Loss Function**     | MSE / L1 Loss                    | **Perceptual + Edge + Binary + GAN** | Added **Binarization** and **Edge Losses** specifically constrained for QR code geometric properties. |
| **Visual Quality**    | Smooth but blurry (Low contrast) | **Sharp & High Contrast**            | Eliminates gray artifacts and restores clear black/white boundaries. |
| **Recognition Rate**  | ~45% (Unstable under noise)      | **~92% (Robust)**                    | Successfully recovers decodability even in heavily degraded samples. |

------

## Methodology (v2.0 Technical Details)

### 1. Enhanced Generator with Attention

We replaced the standard residual blocks from v1.0 with **Residual-in-Residual Dense Blocks (RRDB)** equipped with **Squeeze-and-Excitation (SE)** modules.

- **Motivation:** QR code decoding relies heavily on global structural connectivity. The SE module recalibrates channel-wise features, suppressing irrelevant background noise while enhancing the weights of geometric structures.

### 2. Domain-Specific Loss Design

General SR losses often result in pixel values clustering in the gray range. v2.0 enforces binary characteristics through:

- **$L_{edge}$ (Edge Loss):** Utilizes Sobel operators to calculate gradient maps, maximizing the gradient magnitude at pixel transitions.

- $L_{bin}$ (Binarization Loss): Penalizes gray pixels (near 0.5), forcing convergence toward 0 or 1:

  

  $$L_{bin} = \frac{1}{N} \sum (x \cdot (1-x))$$

### 3. Two-Stage Training Strategy

- **Stage 1 (Warm-up):** Pre-training with L1 Loss to obtain stable initial weights.
- **Stage 2 (GAN Fine-tuning):** Activates the Discriminator and Perceptual Loss to jump visual quality from "blurry" to "sharp".

------

## Project Structure

Plaintext

```
├── checkpoints_gan/      # [NEW] v2.0 GAN trained model weights
├── dataset_4/            # [NEW] Augmented dataset with complex noise
├── failed_cases_analysis/# [NEW] Automated reports for hard samples
├── model.py              # Generator code with SE-RRDB
├── discriminator.py      # [NEW] VGG-style Discriminator for GAN
├── train_gan.py          # [NEW] Core v2.0 training script
├── evaluate.py           # [NEW] Evaluation with TTA (Test-Time Augmentation)
└── ...
```

------

## Quick Start

### 1. Training Workflow

To ensure convergence stability, a two-stage approach is recommended.

**Step 1: Warm-up (Optional)**

```
python train.py --epochs 10
```

**Step 2: Adversarial Training (Core)**

Load pre-trained weights and begin GAN fine-tuning:

```
# Ensure CONFIG in train_gan.py points to the correct dataset path
python train_gan.py
```

### 2. Evaluation

Evaluate the recognition rate on the validation set and generate comparison plots:

```
python evaluate.py
```

### 3. Single Image Inference

```
python test_single.py --img inputs/blurred_qr.png --model checkpoints_gan/gan_generator_epoch70.pth
```

------

## Benchmark

**Test Environment:** `dataset_v4_realistic` (Synthetic blur, noise, and downsampling degradation).

| **Metric**            | **v1.0 Baseline** | **v2.0 (Proposed)** |
| --------------------- | ----------------- | ------------------- |
| **PSNR**              | 24.5 dB           | **22.1 dB***        |
| **SSIM**              | 0.82              | **0.89**            |
| **Decoding Accuracy** | 45.2%             | **92.6%**           |

**Note: GAN-based methods often yield slightly lower PSNR than L1 regression because pixel-level statistical alignment does not equate to perceptual sharpness. For QR codes, decoding success is the decisive metric.*

------

## 中文说明 (v2.0)

本项目是二维码超分辨率复原网络的 **v2.0** 版本。相比于基于标准残差网络的 **v1.0**，本版本引入了 **RRDB 密集残差块**、**SE 注意力机制**以及**生成对抗训练 (GAN)**，解决了边缘模糊和伪影问题。

------

## Citation

If v2.0 improvements help your research, please cite this project:

```
@article{QR-SR-GAN-v2,
  title={QR Code Restoration via Adversarial Learning and Attention Mechanisms},
  author={Li Zhongzheng},
  journal={GitHub Repository},
  year={2025}
}
```