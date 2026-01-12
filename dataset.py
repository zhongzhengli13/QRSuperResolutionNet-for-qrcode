# @Author : LiZhongzheng
# 开发时间  ：2025-12-27 (Optimized)

import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

class QRSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, augment=True, preload=False):
        """
        :param lr_dir: 低分辨率图像目录
        :param hr_dir: 高分辨率图像目录
        :param augment: 是否开启数据增强（随机翻转、旋转），训练时建议开启，验证时关闭
        :param preload: 是否将所有图片预加载到 RAM，极大提升训练速度
        """
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.augment = augment
        self.preload = preload

        # 获取并过滤图片文件
        self.image_filenames = sorted([
            f for f in os.listdir(lr_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        # 预先检查配对是否完整
        self._check_pairing()

        # 如果开启预加载，一次性读取所有图片
        self.cache = []
        if self.preload:
            print(f"📦 Pre-loading {len(self.image_filenames)} images into RAM...")
            for filename in tqdm(self.image_filenames, unit="img"):
                lr, hr = self._load_file(filename)
                self.cache.append((lr, hr))
            print("✅ Pre-loading finished.")

    def _check_pairing(self):
        """初始化时检查所有 HR 图片是否存在，避免训练中途报错"""
        missing_files = []
        for lr_name in self.image_filenames:
            # 假设命名规则是 lr_0.png -> hr_0.png
            hr_name = lr_name.replace("lr_", "hr_")
            hr_path = os.path.join(self.hr_dir, hr_name)
            if not os.path.exists(hr_path):
                missing_files.append(hr_name)
        
        if missing_files:
            raise FileNotFoundError(f"❌ Missing {len(missing_files)} HR images. First missing: {missing_files[0]}")

    def _load_file(self, filename):
        """读取单对文件"""
        hr_filename = filename.replace("lr_", "hr_")
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, hr_filename)

        # 始终转换为单通道灰度图
        lr_img = Image.open(lr_path).convert("L")
        hr_img = Image.open(hr_path).convert("L")

        return lr_img, hr_img

    def _apply_augmentation(self, lr, hr):
        """
        对 LR 和 HR 应用完全相同的随机变换
        """
        # 1. 随机水平翻转
        if random.random() > 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        # 2. 随机垂直翻转
        if random.random() > 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)
            
        # 3. 随机旋转 (0, 90, 180, 270)
        # 这种离散旋转最适合二维码，不会引入插值模糊
        rotations = random.choice([0, 90, 180, 270])
        if rotations > 0:
            lr = TF.rotate(lr, rotations)
            hr = TF.rotate(hr, rotations)

        return lr, hr

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.preload:
            lr_img, hr_img = self.cache[idx]
        else:
            filename = self.image_filenames[idx]
            lr_img, hr_img = self._load_file(filename)

        # 数据增强 (仅在训练模式下)
        if self.augment:
            lr_img, hr_img = self._apply_augmentation(lr_img, hr_img)

        # 转换为 Tensor (自动归一化到 [0, 1])
        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)

        return lr_tensor, hr_tensor