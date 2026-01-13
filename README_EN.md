# QRSuperResolutionNet (v2.0)

**Robust QR Code Restoration via Adversarial Learning and Attention Mechanisms**

[中文](D:\github\SR_QRCODE\sr_qrcode_github\mine_model_v2\README.md)

This repository is the official PyTorch implementation of QRSuperResolutionNet v2.0.

Addressing the issue of decoding failures in complex degradation scenarios, this project proposes a restoration scheme based on an improved ESRGAN. By introducing an RRDB (Residual-in-Residual Dense Block) backbone, SE (Squeeze-and-Excitation) channel attention mechanisms, and hybrid loss function constraints, it aims to enhance the reconstruction quality of critical geometric features (such as Finder Patterns) in QR codes.

------

## ✨ Key Features

Based on the provided implementation, v2.0 includes the following core technical improvements:

- **Architecture Upgrade**: Replaces the standard ResNet with deep **RRDB** modules and integrates **SE Attention** modules after each residual block to enhance feature response to texture details.
- **Two-Stage Training Strategy**:
  - **Warm-up**: Pixel-wise pre-training based on L1 Loss to initialize generator weights.
  - **GAN Fine-tuning**: Introduces a VGG Discriminator for adversarial training to optimize perceptual quality.
- **Compound Loss Function**: Integrates **L1 Loss** (Content), **Perceptual Loss**, **Edge Loss** (Sobel constraint), and **Binarization Loss**.
- **Robust Decoding Post-processing**:
  - **TTA (Test-Time Augmentation)**: Automatically attempts to rotate the input ($90^{\circ}, 180^{\circ}, 270^{\circ}$) if decoding fails during inference.
  - **Morphological Repair**: Integrates traditional image processing techniques such as Otsu thresholding, sharpening, and opening/closing operations to eliminate artifacts.

------

## 📂 Directory Structure

Plaintext

```
QRSuperResolutionNet/
├── checkpoints_gan/      # Model weights storage (GAN stage)
├── dataset_4/            # Dataset directory (must contain train/val subdirs)
├── vis_gan/              # Visualization results during training
├── dataset.py            # Data loader (includes RAM pre-loading & random rotation)
├── discriminator.py      # VGG-Style Discriminator definition
├── model.py              # Generator core network (RRDB + SE-Attention)
├── train.py              # Stage 1: Warm-up training script
├── train_gan.py          # Stage 2: Adversarial training script
├── evaluate.py           # Evaluation script (Integrated TTA & Morphology)
└── test_single.py        # Single image inference demo
```

------

## 🚀 Usage

### 1. Data Preparation

Please organize your data according to the structure below. The code supports `.png` and `.jpg` formats by default.

Plaintext

```
dataset_4/
├── train/
│   ├── lr/  # Low-resolution / Blurry images
│   └── hr/  # High-resolution / Ground Truth (filenames must match lr)
└── val/
    ├── lr/
    └── hr/
```

> **Note**: `dataset.py` implements a `preload=True` option, which loads all data into RAM to accelerate training. If you have insufficient memory, please set this to `False` during instantiation.

### 2. Training

A **Two-Stage** training mode is recommended to ensure convergence stability.

Stage 1: Warm-up (Pixel-wise Pre-training)

Train the generator using only L1, Edge, and Perceptual losses to obtain basic restoration capabilities.

Bash

```
# Modify the CONFIG path in train.py before running
python train.py
```

Stage 2: Adversarial Fine-tuning (GAN)

Load the weights from Stage 1 and enable the discriminator for adversarial training to recover high-frequency textures.

Bash

```
# Modify 'pretrained_model' in train_gan.py to the path saved in Stage 1
python train_gan.py
```

### 3. Evaluation & Inference

Batch Evaluation

Evaluate model performance on the validation set. The script automatically calculates decoding rates for LR, Standard SR, and SR with TTA, and statistics for "Recovered" samples.

Bash

```
python evaluate.py
```

Single Image Inference

Restore a single image. The result will be saved as a side-by-side comparison.

Bash

```
python test_single.py \
    --img inputs/blurred_qr.png \
    --model checkpoints_gan/gan_generator_epoch70.pth \
    --out result.png
```

------

## 📸 Visual Demonstration

*(Place your comparison images here)*

------

## 📝 Methodology Details

### Network Architecture

The Generator consists of `Config['num_blocks']` (default 16) stacked **RRDBs**. Each Block uses `ResidualDenseBlock` for feature extraction, connected to an `SEBlock` for channel weighting at the end, and finally upsampled 4x via `PixelShuffle`.

### Loss Function Definition

The overall optimization objective is:



$$L_{total} = \lambda_{1}L_{pixel} + \lambda_{p}L_{percep} + \lambda_{e}L_{edge} + \lambda_{b}L_{bin} + \lambda_{adv}L_{GAN}$$

Where $L_{edge}$ utilizes the Sobel operator to calculate gradient differences, and $L_{bin}$ penalizes intermediate gray values ($x(1-x)$), forcing the network to output binarized images.

------

## 📍 Citation

If you find this project useful for your research, please cite:

代码段

```
@article{QR-SR-GAN-v2,
  title={QR Code Restoration via Adversarial Learning and Attention Mechanisms},
  author={Li, Zhongzheng},
  journal={GitHub Repository},
  year={2026}
}
```

------

**Contact**: @  [ZhongzhengLi](https://www.google.com/search?q=https://github.com/zhongzhengli13&authuser=1)  If you have any questions, please open an Issue!