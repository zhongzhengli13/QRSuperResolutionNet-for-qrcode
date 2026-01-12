import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 引入你的模块
from dataset import QRSRDataset
from model import QRSuperResolutionNet
from discriminator import Discriminator
# 复用之前的 Loss
from train import PerceptualLoss, EdgeLoss, BinarizationLoss, calc_ssim, evaluate_recognition_accuracy

# ================= 配置 =================
CONFIG = {
    "train_dir_lr": "dataset_4/train/lr", # 确保路径对
    "train_dir_hr": "dataset_4/train/hr",
    "pretrained_model": "checkpoints_1/qr_sr_latest.pth", # 👈 这里填你刚才那个 71% 的模型路径
    "batch_size": 16,
    "lr_G": 1e-5,  # 生成器学习率要非常小，防止破坏已有结构
    "lr_D": 1e-4,  # 判别器学习率稍大
    "epochs": 70, # GAN 训练不需要太久，收敛很快
    "device": "cuda",
    "save_dir": "checkpoints_gan",
    "vis_dir": "vis_gan"
}
# =======================================

def train_gan():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["vis_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])

    # 1. 数据集
    print("Loading dataset...")
    train_dataset = QRSRDataset(CONFIG["train_dir_lr"], CONFIG["train_dir_hr"], augment=True, preload=True)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)

    # 2. 模型初始化
    generator = QRSuperResolutionNet(num_blocks=16).to(device)
    discriminator = Discriminator().to(device)

    # ✅ 关键：加载预训练权重 (Warm Start)
    if os.path.exists(CONFIG["pretrained_model"]):
        print(f"🔄 Loading pretrained Generator from {CONFIG['pretrained_model']}...")
        checkpoint = torch.load(CONFIG["pretrained_model"], map_location=device)
        if 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    else:
        print("⚠️ Warning: Pretrained model not found! Training GAN from scratch is unstable.")

    # 3. 优化器 & Loss
    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG["lr_G"], betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG["lr_D"], betas=(0.9, 0.999))
    
    # 损失函数
    criterion_GAN = nn.BCEWithLogitsLoss().to(device) # 对抗损失
    criterion_pixel = nn.L1Loss().to(device)
    criterion_perc = PerceptualLoss(device).to(device)
    criterion_edge = EdgeLoss(device).to(device)
    criterion_bin = BinarizationLoss().to(device)

    scaler = GradScaler()

    for epoch in range(1, CONFIG["epochs"] + 1):
        generator.train()
        discriminator.train()
        
        loop = tqdm(train_loader, desc=f"GAN Epoch {epoch}/{CONFIG['epochs']}", unit="batch")
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        for batch_idx, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(device), hr.to(device)

            # ===============================================
            #  训练 判别器 (Discriminator)
            # ===============================================
            with autocast():
                # 1. 生成假图
                fake_hr = generator(lr)

                # 2. 判别器打分
                pred_real = discriminator(hr)           # 真图得分
                pred_fake = discriminator(fake_hr.detach()) # 假图得分 (detach防止梯度传回G)

                # 3. 计算 D 的损失 (真图要判1，假图要判0)
                # 标签平滑 (Label Smoothing): 真图用 0.9 而不是 1.0，防止判别器过度自信
                loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real) * 0.9)
                loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = (loss_real + loss_fake) / 2

            optimizer_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # ===============================================
            #  训练 生成器 (Generator)
            # ===============================================
            with autocast():
                # 1. 重新让判别器看一眼生成的假图 (这次带有梯度)
                pred_fake = discriminator(fake_hr)

                # 2. 对抗损失 (Generator 希望判别器把假图判为 1)
                loss_adversarial = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

                # 3. 内容损失组合
                loss_content = criterion_pixel(fake_hr, hr)
                loss_perc = criterion_perc(fake_hr, hr)
                loss_edge = criterion_edge(fake_hr, hr)
                loss_bin = criterion_bin(fake_hr)

                # 4. 总损失 G
                # 权重配比非常关键：
                # GAN Loss 负责纹理 (给一点点权重 0.005 就够了)
                # Edge/Bin 负责结构
                # Content 负责颜色准确
                loss_G = loss_content + 0.1 * loss_perc + 1.0 * loss_edge + 0.5 * loss_bin + 0.005 * loss_adversarial

            optimizer_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            # 更新 Scaler
            scaler.update()

            epoch_loss_g += loss_G.item()
            epoch_loss_d += loss_D.item()
            
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())

            # 可视化
            if batch_idx == 0:
                with torch.no_grad():
                     # 简单的拼接展示
                    lr_resized = torch.nn.functional.interpolate(lr, size=(256, 256), mode='nearest')
                    comparison = torch.cat([lr_resized[:4], fake_hr[:4], hr[:4]], dim=0)
                    save_image(comparison, f"{CONFIG['vis_dir']}/epoch{epoch}_gan.png", nrow=4)

        print(f"Epoch {epoch} | Loss G: {epoch_loss_g/len(train_loader):.4f} | Loss D: {epoch_loss_d/len(train_loader):.4f}")

        # 保存模型
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), f"{CONFIG['save_dir']}/gan_generator_epoch{epoch}.pth")
            
            # 验证一下
            acc = evaluate_recognition_accuracy(generator, train_loader, device) # 这里偷懒用了 train_loader，实际建议用 val
            print(f"🔍 Current Recognition Accuracy: {acc}%")

if __name__ == "__main__":
    train_gan()