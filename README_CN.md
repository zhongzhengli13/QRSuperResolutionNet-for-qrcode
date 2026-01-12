# QR-Code-SR-GAN (v2.0): 基于对抗学习与注意力机制的鲁棒二维码复原网络

> **v2.0 新特性：** 集成 **ESRGAN** 架构、**SE (Squeeze-and-Excitation) 注意力机制** 以及 **对抗训练 (Adversarial Training)**，彻底解决了 v1.0 版本中存在的边缘模糊与伪影问题。

本仓库托管了二维码超分辨率复原网络的 **v2.0** 实现。不同于仅依赖标准残差学习的 [v1.0-original](https://github.com/zhongzhengli13/QRSuperResolutionNet-for-qrcode/tree/v1.0-original) 版本，**v2.0** 引入了完整的生成对抗网络 (GAN) 流程，旨在重建对于扫码解码至关重要的高频细节（如探测图形 Finder Patterns 和校正图形 Alignment Patterns）。

------

## v2.0 vs. v1.0: 核心改进对比

v2.0 的主要目标是克服 v1.0 中像素级损失函数（Pixel-wise Loss）带来的“平滑效应”。

| **特性**       | **v1.0 (原始版)**      | **v2.0 (本文/改进版)**         | **改进原理分析**                                             |
| -------------- | ---------------------- | ------------------------------ | ------------------------------------------------------------ |
| **网络架构**   | 标准 ResNet (SRResNet) | **RRDB + SE-Attention**        | 引入残差密集块 (RRDB) 提升网络容量；SE 注意力模块使网络更关注纹理通道而非背景噪声。 |
| **训练范式**   | 监督学习 (像素级 L1)   | **对抗学习 (GAN)**             | 判别器 (Discriminator) 迫使生成器输出 L1 损失无法实现的锐利边缘。 |
| **损失函数**   | MSE / L1 Loss          | **感知 + 边缘 + 二值化 + GAN** | 新增 **二值化损失** 与 **边缘损失**，专门针对二维码的几何二值特性进行约束。 |
| **视觉效果**   | 平滑但模糊 (低对比度)  | **锐利且高对比度**             | 消除灰色伪影，恢复清晰的黑白边界。                           |
| **扫码识别率** | ~45% (噪声下不稳定)    | **~92% (强鲁棒性)**            | 即便在严重降质样本下也能成功恢复可解码性。                   |

### 视觉效果对比

<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20260112165407424.png" alt="image-20260112165407424" style="zoom:50%;" />

------

## 方法论 (v2.0 技术细节)

### 1. 融合注意力机制的增强生成器

我们将 v1.0 中的标准残差块替换为 **带有压缩-激励 (SE) 模块的残差密集块 (RRDB)**。

- **动机：** 二维码高度依赖全局结构的连通性。SE 模块通过显式建模通道间的相关性，帮助网络抑制无用的背景噪声特征，增强几何结构特征的权重。

### 2. 领域特定的损失函数设计

通用的超分损失（如 v1.0 所用）往往导致输出图像像素值集中在灰色区间。v2.0 强制约束二值特性：

- **$L_{edge}$ (边缘损失):** 利用 Sobel 算子计算梯度图，最大化像素跳变处的梯度幅值。

- $L_{bin}$ (二值化损失): 惩罚接近 0.5 的灰度像素，迫使像素值向 0 或 1 收敛。

  

  $$L_{bin} = \frac{1}{N} \sum (x \cdot (1-x))$$

### 3. 两阶段训练策略 (Two-Stage Training)

- **阶段一 (Warm-up):** 仅使用 L1 Loss 进行预训练，获得稳定的初始权重（性能接近 v1.0）。
- **阶段二 (GAN Fine-tuning):** 激活 **判别器** 和 **感知损失 (Perceptual Loss)**。这是视觉质量从“模糊”跃升至“锐利”的关键步骤。

------

## 项目结构说明

```bash
├── checkpoints_gan/      # [NEW] 存放 v2.0 GAN 训练的模型权重
├── dataset_4/            # [NEW] 包含更复杂噪声样本的增强数据集
├── failed_cases_analysis/# [NEW] 自动生成的困难样本分析报告
├── model.py              # 集成 SE-RRDB 的改进版生成器代码
├── discriminator.py      # [NEW] 用于 GAN 训练的 VGG 风格判别器
├── train_gan.py          # [NEW] v2.0 的核心训练脚本
├── evaluate.py           # [NEW] 包含 TTA (测试时增强) 的评估脚本
└── ...
```

------

## 🛠️ 快速开始

### 1. 训练 (v2.0 流程)

为了保证收敛稳定性，建议采用两阶段训练法。

**步骤 1: 预热 (Warm-up, 可选)**

```bash
python train.py --epochs 10
```

**步骤 2: 对抗训练 (核心步骤)**

加载预训练权重并开始 GAN 微调：

```bash
# 请确保 train_gan.py 中的配置指向正确的数据集路径
python train_gan.py
```

### 2. 模型评估

评估模型在验证集上的识别率，并生成对比图：

```bash
python evaluate.py
```

### 3. 单张图片推理

```bash
python test_single.py --img inputs/blurred_qr.png --model checkpoints_gan/gan_generator_epoch70.pth
```

------

## 📊 性能基准 (Benchmark)

测试环境：`dataset_v4_realistic` (包含合成的模糊、噪声、下采样降质)。

| **指标**       | **v1.0 基准 (Baseline)** | **v2.0 (本文方法)** |
| -------------- | ------------------------ | ------------------- |
| **PSNR**       | 24.5 dB                  | **22.1 dB***        |
| **SSIM**       | 0.82                     | **0.89**            |
| **解码成功率** | 45.2%                    | **92.6%**           |

**注意：GAN 方法的 PSNR 通常会略低于纯 L1 回归方法，因为像素级的统计对齐并不代表感知上的清晰度。对于二维码而言，解码成功率是决定性的指标。*

------

## 📜 引用 (Citation)

如果 v2.0 的改进对您的研究有所帮助，请引用本项目：

代码段

```
@article{QR-SR-GAN-v2,
  title={QR Code Restoration via Adversarial Learning and Attention Mechanisms},
  author={Li Zhongzheng},
  journal={GitHub Repository},
  year={2025}
}
```