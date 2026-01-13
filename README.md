# QRSuperResolutionNet (v2.0)

**Robust QR Code Restoration via Adversarial Learning and Attention Mechanisms**

[English](https://github.com/zhongzhengli13/QRSuperResolutionNet-for-qrcode/blob/main/README_EN.md) 

本仓库为 QRSuperResolutionNet v2.0 的官方 PyTorch 实现。

针对二维码在复杂降质场景下的解码失败问题，本项目提出了一种基于 ESRGAN 改进的生成对抗网络复原方案。通过引入 RRDB (Residual-in-Residual Dense Block) 主干、SE (Squeeze-and-Excitation) 通道注意力机制以及混合损失函数约束，旨在提升二维码关键几何特征（如寻像图形）的重建质量。

## 核心特性 (Key Features)

基于提供的代码实现，v2.0 包含以下核心技术改进：

- **架构升级**: 采用深层 RRDB 模块替代普通 ResNet，并在每个残差块后集成 SE Attention 模块，增强对纹理细节的特征响应。
- **两阶段训练策略**:
  1. **Warm-up**: 基于 L1 损失的像素级预训练，初始化生成器权重。
  2. **GAN Fine-tuning**: 引入 VGG 判别器进行对抗训练，优化感知质量。
- **复合损失函数**: 集成 `L1 Loss` (内容)、`Perceptual Loss` (感知)、`Edge Loss` (Sobel边缘约束) 与 `Binarization Loss` (二值化约束)。
- **鲁棒解码后处理**:
  - **TTA (Test-Time Augmentation)**: 推理时若解码失败，自动尝试旋转 (90°/180°/270°) 输入。
  - **形态学修复**: 集成 Otsu 阈值、锐化、开/闭运算等传统图像处理手段以消除伪影。

## 目录结构 (Directory Structure)

Plaintext

```
QRSuperResolutionNet/
├── checkpoints_gan/      # 模型权重保存目录 (GAN阶段)
├── dataset_4/            # 数据集目录 (需包含 train/val 子目录)
├── vis_gan/              # 训练过程可视化结果
├── dataset.py            # 数据加载器 (含内存预加载与随机旋转增强)
├── discriminator.py      # VGG-Style 判别器定义
├── model.py              # 生成器核心网络 (RRDB + SE-Attention)
├── train.py              # 阶段一：Warm-up 训练脚本
├── train_gan.py          # 阶段二：Adversarial 训练脚本
├── evaluate.py           # 评估脚本 (集成 TTA 与形态学后处理)
└── test_single.py        # 单张图像推理演示
```

## 使用指南 (Usage)

### 1. 数据准备

请按以下结构组织数据。代码默认支持 `.png`, `.jpg` 格式。

```
dataset_4/
├── train/
│   ├── lr/  # 低分辨率/模糊图像
│   └── hr/  # 高分辨率/原始图像 (文件名需与lr对应)
└── val/
    ├── lr/
    └── hr/
```

> **注意**: `dataset.py` 中实现了 `preload=True` 选项，会将所有数据加载至 RAM 以加速训练。如果内存不足，请在实例化时设为 `False`。

### 2. 模型训练 (Training)

推荐采用 **Two-Stage** 训练模式以保证收敛稳定性。

Stage 1: Warm-up (Pixel-wise Pre-training)

仅使用 L1、Edge 和感知损失训练生成器，使其获得基础的复原能力。

```bash
# 修改 train.py 中的 CONFIG 路径后运行
python train.py
```

Stage 2: Adversarial Fine-tuning (GAN)

加载 Stage 1 的权重，开启判别器进行对抗训练，恢复高频纹理。

```bash
# 修改 train_gan.py 中的 'pretrained_model' 为 Stage 1 保存的权重路径
python train_gan.py
```

### 3. 评估与测试 (Evaluation)

批量评估

在验证集上评估模型性能。脚本会自动计算 LR、标准 SR 以及开启 TTA 后的最终识别率，并统计“挽救成功 (Recovered)”的样本数。

```bash
python evaluate.py
```

单图推理

对单张图片进行修复测试，结果将保存为左右对比图。

```bash
python test_single.py \
    --img inputs/blurred_qr.png \
    --model checkpoints_gan/gan_generator_epoch70.pth \
    --out result.png
```

> 效果展示：<img src="https://lzz-1340752507.cos.ap-shanghai.myqcloud.com/lzz/image-20260113101155030.png" alt="image-20260113101155030" style="zoom:50%;" />

## 方法论细节 (Methodology)

### 网络架构

生成器由 `Config['num_blocks']` (默认16) 个 **RRDB** 堆叠而成。每个 Block 内部采用 `ResidualDenseBlock` 提取特征，末端连接 `SEBlock` 进行通道加权，最后通过 `PixelShuffle` 实现 4 倍上采样。

### 损失函数定义

总体优化目标为：



$$L_{total} = \lambda_{1}L_{pixel} + \lambda_{p}L_{percep} + \lambda_{e}L_{edge} + \lambda_{b}L_{bin} + \lambda_{adv}L_{GAN}$$

其中 $L_{edge}$ 利用 Sobel 算子计算梯度差，$L_{bin}$ 惩罚非 0/1 的中间灰度值 ($x(1-x)$)，强制网络输出二值化图像。

## 引用 (Citation)

如果您参考了本项目的代码或方法，请引用：

代码段

```
@article{QR-SR-GAN-v2,
  title={QR Code Restoration via Adversarial Learning and Attention Mechanisms},
  author={Zhongzheng Li},
  journal={GitHub Repository},
  year={202}
}
```

------

**Contact**: @ [ZhongzhengLi ](https://github.com/zhongzhengli13/QRSuperResolutionNet-for-qrcode/issues)如有问题，欢迎提 Issue 交流！