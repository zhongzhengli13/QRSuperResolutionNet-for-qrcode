# @Author : LiZhongzheng
# 开发时间  ：2025-12-27 (Optimized)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 导入你的自定义模块
from dataset import QRSRDataset
from model import QRSuperResolutionNet 

# ================= 配置区域 =================
CONFIG = {
    "train_dir_lr": "/root/autodl-tmp/mine-qr-code/mine_model_v2/dataset_1/train/lr",
    "train_dir_hr": "/root/autodl-tmp/mine-qr-code/mine_model_v2/dataset_1/train/hr",
    "batch_size": 16,          # 显存允许的话尽量大，建议 16 或 32
    "num_workers": 6,
    "epochs": 20,
    "lr": 2e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_interval": 5,        # 每几轮保存一次固定权重
    "vis_dir": "vis_1",          # 可视化结果保存路径
    "ckpt_dir": "checkpoints_1", # 模型保存路径
    
    # === 断点续训设置 ===
    # 如果要重新开始训练，设为 "" (空字符串)
    # 如果要从断点恢复，填入路径，例如 "checkpoints/qr_sr_latest.pth"
    "resume_path": "" 
}
# ===========================================


# --- 1. 辅助 Loss 定义 ---

class VGG16Feature(nn.Module):
    def __init__(self):
        super(VGG16Feature, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # 取前 16 层作为特征提取器
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        # 冻结参数，不参与训练
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg_layers(x)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG16Feature().to(device).eval()
        self.criterion = nn.MSELoss()

    def forward(self, sr, hr):
        # 扩展单通道到 3 通道以适应 VGG
        sr_rgb = sr.repeat(1, 3, 1, 1)
        hr_rgb = hr.repeat(1, 3, 1, 1)
        sr_feat = self.vgg(sr_rgb)
        hr_feat = self.vgg(hr_rgb)
        return self.criterion(sr_feat, hr_feat)

class EdgeLoss(nn.Module):
    """边缘损失：使用 Sobel 算子计算梯度差异"""
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_grad_x = F.conv2d(sr, self.k_x, padding=1)
        sr_grad_y = F.conv2d(sr, self.k_y, padding=1)
        hr_grad_x = F.conv2d(hr, self.k_x, padding=1)
        hr_grad_y = F.conv2d(hr, self.k_y, padding=1)
        return self.criterion(sr_grad_x, hr_grad_x) + self.criterion(sr_grad_y, hr_grad_y)

class BinarizationLoss(nn.Module):
    """二值化损失：惩罚接近 0.5 的灰色像素"""
    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        return torch.mean(x * (1 - x))


# --- 2. 工具函数 ---

def calc_ssim(sr_imgs, hr_imgs):
    """计算一批图像的平均 SSIM"""
    sr_np = sr_imgs.detach().cpu().numpy()
    hr_np = hr_imgs.detach().cpu().numpy()
    ssim_score = 0.0
    batch_size = sr_np.shape[0]
    
    for i in range(batch_size):
        sr = sr_np[i, 0] # 取单通道
        hr = hr_np[i, 0]
        
        # 避免全黑全白导致的 NaN
        if np.std(sr) == 0 or np.std(hr) == 0:
            continue
            
        data_range = hr.max() - hr.min()
        if data_range == 0: data_range = 1.0
        
        win_size = min(7, min(sr.shape))
        if win_size % 2 == 0: win_size -= 1 # win_size 必须是奇数
        
        ssim_score += ssim(sr, hr, win_size=win_size, data_range=data_range)
        
    return ssim_score / batch_size

def evaluate_recognition_accuracy(model, dataloader, device):
    """评估识别率 (仅用于验证，不参与梯度计算)"""
    model.eval()
    total = 0
    recognized = 0

    # 为了速度，只测试一部分数据（比如前 5 个 batch）
    max_batches = 5
    
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            if i >= max_batches: break
            
            lr_imgs = lr_imgs.to(device)
            sr_imgs = model(lr_imgs).cpu()

            for img in sr_imgs:
                img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np, mode='L')
                # 尝试解码
                if len(decode(pil_img)) > 0:
                    recognized += 1
                total += 1

    model.train() # 切回训练模式
    acc = recognized / total if total > 0 else 0
    return round(acc * 100, 2)


# --- 3. 主训练流程 ---

def train():
    # 准备目录
    os.makedirs(CONFIG["vis_dir"], exist_ok=True)
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)

    # 1. 数据集
    print("Loading dataset...")
    train_dataset = QRSRDataset(CONFIG["train_dir_lr"], CONFIG["train_dir_hr"])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # 2. 模型与优化器
    model = QRSuperResolutionNet(num_blocks=16).to(CONFIG["device"]) # 确保这里调用的是深层网络
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    # 使用余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)

    # 3. 损失函数
    l1_loss = nn.L1Loss().to(CONFIG["device"])
    perc_loss = PerceptualLoss(CONFIG["device"]).to(CONFIG["device"])
    edge_loss = EdgeLoss(CONFIG["device"]).to(CONFIG["device"])
    bin_loss = BinarizationLoss().to(CONFIG["device"])

    # 4. 断点续训逻辑
    start_epoch = 1
    if CONFIG["resume_path"] and os.path.exists(CONFIG["resume_path"]):
        print(f"🔄 Resuming training from {CONFIG['resume_path']}...")
        checkpoint = torch.load(CONFIG["resume_path"], map_location=CONFIG["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed successfully from Epoch {start_epoch - 1}")
    else:
        print("🚀 Starting new training...")

    # 5. 训练循环
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", unit="batch")

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(CONFIG["device"])
            hr_imgs = hr_imgs.to(CONFIG["device"])

            # Forward
            sr_imgs = model(lr_imgs)

            # Calculate Losses
            loss_content = l1_loss(sr_imgs, hr_imgs)
            loss_p = perc_loss(sr_imgs, hr_imgs)
            loss_e = edge_loss(sr_imgs, hr_imgs)
            loss_b = bin_loss(sr_imgs)

            # 加权求和 (你可以根据情况微调权重)
            # 内容占主导，边缘负责清晰，二值化负责去灰
            total_loss = loss_content + 0.1 * loss_p + 0.5 * loss_e + 0.1 * loss_b

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪：对于深层网络非常重要，防止 Loss 突然飞升
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()

            # Update Progress Bar
            epoch_loss += total_loss.item()
            loop.set_postfix(
                loss=f"{total_loss.item():.4f}",
                edge=f"{loss_e.item():.4f}",
                ssim=f"{calc_ssim(sr_imgs, hr_imgs):.3f}"
            )

            # 每个 Epoch 的第一个 Batch 保存可视化图片
            # if batch_idx == 0:
            #     comparison = torch.cat([lr_imgs, sr_imgs, hr_imgs], dim=0)
            #     # 只取前 4 组展示
            #     save_image(comparison[:12], f"{CONFIG['vis_dir']}/epoch{epoch}_sample.png", nrow=4)
            if batch_idx == 0:
                # 1. 临时将 LR 放大到 HR 尺寸，仅用于拼接展示
                # 使用 'nearest' 插值可以保留 LR 的马赛克感，让你看清原本有多糊
                lr_resized = F.interpolate(lr_imgs, size=(256, 256), mode='nearest')
                
                # 2. 拼接 (LR放大版, SR预测版, HR真值)
                # 即使 batch_size=8，我们只取前4张展示，避免图片过大
                n_vis = min(CONFIG['batch_size'], 4)
                comparison = torch.cat([
                    lr_resized[:n_vis], 
                    sr_imgs[:n_vis], 
                    hr_imgs[:n_vis]
                ], dim=0)
                
                # 3. 保存，nrow=n_vis 表示每一行展示一组 (LR, SR, HR) 的对比
                save_image(comparison, f"{CONFIG['vis_dir']}/epoch{epoch}_sample.png", nrow=n_vis)

        # 打印 Epoch 总结
        avg_loss = epoch_loss / len(train_loader)
        
        # 验证识别率 (每 2 个 Epoch 测一次，节省时间)
        acc_msg = ""
        if epoch % 2 == 0:
            acc = evaluate_recognition_accuracy(model, train_loader, CONFIG["device"])
            acc_msg = f"| Recog Acc: {acc}%"
        
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.6f} {acc_msg}")

        # 6. 保存模型 (Checkpointing)
        
        # 保存 "Latest" (用于断点续训，每次覆盖)
        ckpt_content = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }
        torch.save(ckpt_content, f"{CONFIG['ckpt_dir']}/qr_sr_latest.pth")

        # 保存阶段性模型 (不覆盖)
        if epoch % CONFIG["save_interval"] == 0:
            torch.save(model.state_dict(), f"{CONFIG['ckpt_dir']}/qr_sr_epoch{epoch}.pth")
            print(f"💾 Checkpoint saved: epoch {epoch}")

        scheduler.step()

if __name__ == "__main__":
    train()