# # # import torch
# # # import torch.nn as nn
# # # from torch.utils.data import DataLoader
# # # from tqdm import tqdm
# # # from pyzbar.pyzbar import decode
# # # from PIL import Image
# # # import numpy as np
# # # import os

# # # # 引用你的项目模块
# # # from dataset import QRSRDataset
# # # from model import QRSuperResolutionNet

# # # # ================= 配置 =================
# # # CONFIG = {
# # #     "val_lr_dir": "dataset_4/val/lr",
# # #     "val_hr_dir": "dataset_4/val/hr",
# # #     "model_path": "/root/autodl-tmp/mine-qr-code/mine_model_v2/checkpoints_gan/gan_generator_epoch70.pth", # 或者选一个效果最好的 epoch
# # #     "device": "cuda" if torch.cuda.is_available() else "cpu",
# # #     "batch_size": 16,  # 评估时 Batch 可以大一点
# # #     "num_blocks": 16   # 必须与训练时保持一致！
# # # }
# # # # =======================================

# # # # def scan_image(tensor_img):
# # # #     """
# # # #     辅助函数：将 Tensor 转换为 PIL 图片并尝试扫码
# # # #     返回: (是否成功, 扫码内容)
# # # #     """
# # # #     # Tensor (C, H, W) -> Numpy (H, W) -> uint8
# # # #     img_np = tensor_img.squeeze().cpu().numpy()
# # # #     img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
# # # #     # 转换为 PIL
# # # #     pil_img = Image.fromarray(img_np, mode='L')
    
# # # #     # 尝试解码
# # # #     decoded_objects = decode(pil_img)
    
# # # #     if decoded_objects:
# # # #         # 返回第一个解码出的数据字符串
# # # #         return True, decoded_objects[0].data.decode("utf-8")
# # # #     else:
# # # #         # 如果直接扫失败，尝试二值化后再扫一次（增强鲁棒性）
# # # #         # 很多扫描器对二值化后的图像更敏感
# # # #         bw_img = pil_img.point(lambda x: 0 if x < 128 else 255, '1')
# # # #         decoded_bw = decode(bw_img)
# # # #         if decoded_bw:
# # # #              return True, decoded_bw[0].data.decode("utf-8")
             
# # # #         return False, None
# # # def scan_image(tensor_img):
# # #     """
# # #     辅助函数：将 Tensor 转换为 PIL 图片并尝试扫码
# # #     增加了【二值化】预处理，能显著提升识别率
# # #     """
# # #     # 1. 转为 Numpy uint8 [0, 255]
# # #     img_np = tensor_img.squeeze().cpu().numpy()
# # #     img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
# # #     # 2. 转换 PIL
# # #     pil_img = Image.fromarray(img_np, mode='L')
    
# # #     # === 策略 A: 直接扫描 (原图) ===
# # #     decoded_objects = decode(pil_img)
# # #     if decoded_objects:
# # #         return True, decoded_objects[0].data.decode("utf-8")
    
# # #     # === 策略 B: 强制二值化 (Thresholding) ===
# # #     # 将所有像素：<127 变黑(0), >127 变白(255)
# # #     # 这对去除模型输出的“灰色噪声”非常有效
# # #     bw_img = pil_img.point(lambda x: 0 if x < 127 else 255, '1')
# # #     decoded_bw = decode(bw_img)
# # #     if decoded_bw:
# # #         return True, decoded_bw[0].data.decode("utf-8")

# # #     # === 策略 C: 使用 OpenCV 的 Otsu 自动阈值 (更高级) ===
# # #     # 适用于光照不均匀的情况
# # #     # cv_img = np.array(pil_img)
# # #     # _, otsu = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #     # decoded_otsu = decode(Image.fromarray(otsu))
# # #     # if decoded_otsu:
# # #     #     return True, decoded_otsu[0].data.decode("utf-8")

# # #     return False, None

# # # def evaluate():
# # #     print(f"Loading model from {CONFIG['model_path']} ...")
# # #     device = torch.device(CONFIG['device'])
    
# # #     # 加载模型
# # #     model = QRSuperResolutionNet(num_blocks=CONFIG['num_blocks']).to(device)
# # #     checkpoint = torch.load(CONFIG['model_path'], map_location=device)
# # #     if 'model_state_dict' in checkpoint:
# # #         model.load_state_dict(checkpoint['model_state_dict'])
# # #     else:
# # #         model.load_state_dict(checkpoint)
# # #     model.eval()

# # #     # 加载数据 (验证集不需要 shuffle)
# # #     # 注意：这里 augment=False，评估时我们要看稳定的结果
# # #     val_dataset = QRSRDataset(CONFIG['val_lr_dir'], CONFIG['val_hr_dir'], augment=False, preload=False)
# # #     val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

# # #     # 统计指标
# # #     stats = {
# # #         "total": 0,
# # #         "valid_hr": 0,      # HR 本身能扫出来的数量 (有效数据基准)
# # #         "lr_success": 0,    # 低清图能扫出来的数量
# # #         "sr_success": 0,    # 修复图能扫出来的数量
# # #         "recovered": 0,     # LR扫不出 -> SR能扫出 (真正被模型救回来的)
# # #         "broken": 0         # LR能扫出 -> SR扫不出 (模型帮倒忙的)
# # #     }

# # #     print("🚀 Starting evaluation...")
    
# # #     with torch.no_grad():
# # #         for lr_imgs, hr_imgs in tqdm(val_loader, unit="batch"):
# # #             lr_imgs = lr_imgs.to(device)
            
# # #             # 模型推理
# # #             sr_imgs = model(lr_imgs)

# # #             # 遍历当前 Batch 的每一张图
# # #             batch_size = lr_imgs.size(0)
# # #             for i in range(batch_size):
# # #                 stats["total"] += 1
                
# # #                 # 1. 检查 Ground Truth (HR)
# # #                 hr_ok, hr_txt = scan_image(hr_imgs[i])
                
# # #                 # 如果连高清原图都扫不出来，这张图不具备测试价值（可能是数据集生成时截断了或者 pyzbar 能力有限）
# # #                 if not hr_ok:
# # #                     continue
                
# # #                 stats["valid_hr"] += 1
                
# # #                 # 2. 检查 LR (低清)
# # #                 lr_ok, lr_txt = scan_image(lr_imgs[i])
                
# # #                 # 3. 检查 SR (超分修复)
# # #                 sr_ok, sr_txt = scan_image(sr_imgs[i])
                
# # #                 # 验证内容一致性（防止扫出来了但是是乱码）
# # #                 if lr_ok and lr_txt != hr_txt: lr_ok = False
# # #                 if sr_ok and sr_txt != hr_txt: sr_ok = False

# # #                 # 统计
# # #                 if lr_ok: stats["lr_success"] += 1
# # #                 if sr_ok: stats["sr_success"] += 1
                
# # #                 if not lr_ok and sr_ok:
# # #                     stats["recovered"] += 1 # 救回来了！
# # #                 if lr_ok and not sr_ok:
# # #                     stats["broken"] += 1    # 搞坏了...

# # #     # === 生成报告 ===
# # #     if stats["valid_hr"] == 0:
# # #         print("❌ No valid HR QR codes found in dataset. Check your data.")
# # #         return

# # #     print("\n" + "="*40)
# # #     print("📊 EVALUATION REPORT")
# # #     print("="*40)
# # #     print(f"Total Images Processed:    {stats['total']}")
# # #     print(f"Valid HR Images (Base):    {stats['valid_hr']}")
# # #     print("-" * 40)
    
# # #     lr_acc = (stats['lr_success'] / stats['valid_hr']) * 100
# # #     sr_acc = (stats['sr_success'] / stats['valid_hr']) * 100
    
# # #     print(f"📉 LR Recognition Rate:    {lr_acc:.2f}%  ({stats['lr_success']}/{stats['valid_hr']})")
# # #     print(f"📈 SR Recognition Rate:    {sr_acc:.2f}%  ({stats['sr_success']}/{stats['valid_hr']})")
# # #     print("-" * 40)
# # #     print(f"🚑 Recovered (Fail->Pass): {stats['recovered']}")
# # #     print(f"💔 Broken (Pass->Fail):    {stats['broken']}")
# # #     print("="*40)

# # #     # 简短评价
# # #     gain = sr_acc - lr_acc
# # #     if gain > 0:
# # #         print(f"✅ Model improved recognition rate by +{gain:.2f}% points.")
# # #     else:
# # #         print(f"⚠️ Model performance is degraded or equal to input.")

# # # if __name__ == "__main__":
# # #     evaluate()

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm
# # from pyzbar.pyzbar import decode
# # from PIL import Image
# # import numpy as np
# # import os
# # import cv2

# # # 引用你的项目模块
# # from dataset import QRSRDataset
# # from model import QRSuperResolutionNet

# # # ================= 配置 =================
# # CONFIG = {
# #     "val_lr_dir": "dataset_4/val/lr",
# #     "val_hr_dir": "dataset_4/val/hr",
# #     # 记得改成你效果最好的那个 GAN 模型路径
# #     "model_path": "checkpoints_gan/gan_generator_epoch70.pth", 
# #     "device": "cuda" if torch.cuda.is_available() else "cpu",
# #     "batch_size": 16,  
# #     "num_blocks": 16
# # }
# # # =======================================

# # def try_scan(pil_img):
# #     """尝试多种预处理方式进行扫码"""
# #     # 1. 直接扫
# #     res = decode(pil_img)
# #     if res: return True, res[0].data.decode("utf-8")
    
# #     # 2. 二值化后扫 (Otsu)
# #     cv_img = np.array(pil_img)
# #     _, binary = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #     res = decode(Image.fromarray(binary))
# #     if res: return True, res[0].data.decode("utf-8")
    
# #     # 3. 锐化后扫 (Sharpening)
# #     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# #     sharpened = cv2.filter2D(cv_img, -1, kernel)
# #     res = decode(Image.fromarray(sharpened))
# #     if res: return True, res[0].data.decode("utf-8")

# #     return False, None

# # def evaluate():
# #     print(f"Loading model from {CONFIG['model_path']} ...")
# #     device = torch.device(CONFIG['device'])
    
# #     model = QRSuperResolutionNet(num_blocks=CONFIG['num_blocks']).to(device)
# #     checkpoint = torch.load(CONFIG['model_path'], map_location=device)
# #     if 'model_state_dict' in checkpoint:
# #         model.load_state_dict(checkpoint['model_state_dict'])
# #     else:
# #         model.load_state_dict(checkpoint)
# #     model.eval()

# #     val_dataset = QRSRDataset(CONFIG['val_lr_dir'], CONFIG['val_hr_dir'], augment=False, preload=False)
# #     val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

# #     stats = {
# #         "total": 0,
# #         "valid_hr": 0,
# #         "lr_success": 0,
# #         "sr_success": 0,     # 一次通过
# #         "sr_tta_success": 0, # TTA 救回来的
# #         "recovered": 0,
# #     }

# #     print("🚀 Starting evaluation with TTA (Test-Time Augmentation)...")
    
# #     with torch.no_grad():
# #         for lr_imgs, hr_imgs in tqdm(val_loader, unit="batch"):
# #             lr_imgs = lr_imgs.to(device)
# #             batch_size = lr_imgs.size(0)

# #             # ==========================================
# #             # 第一次推理 (原始方向)
# #             # ==========================================
# #             sr_imgs = model(lr_imgs)

# #             for i in range(batch_size):
# #                 stats["total"] += 1
                
# #                 # 1. 验证 HR (基准)
# #                 hr_pil = Image.fromarray((hr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
# #                 hr_ok, hr_txt = try_scan(hr_pil)
# #                 if not hr_ok: continue
# #                 stats["valid_hr"] += 1
                
# #                 # 2. 验证 LR
# #                 lr_pil = Image.fromarray((lr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
# #                 lr_ok, lr_txt = try_scan(lr_pil)
# #                 if lr_ok and lr_txt == hr_txt: stats["lr_success"] += 1

# #                 # 3. 验证 SR (常规)
# #                 sr_img_np = (sr_imgs[i].squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
# #                 sr_pil = Image.fromarray(sr_img_np, mode='L')
# #                 sr_ok, sr_txt = try_scan(sr_pil)

# #                 # ==========================================
# #                 # 🔥 TTA 救场机制
# #                 # 如果第一次没扫出来，尝试旋转 LR 图片再送入模型
# #                 # ==========================================
# #                 final_success = False
                
# #                 if sr_ok and sr_txt == hr_txt:
# #                     stats["sr_success"] += 1
# #                     final_success = True
# #                 else:
# #                     # 失败了？别急，旋转大法好！
# #                     # 依次尝试旋转 90, 180, 270 度
# #                     lr_tensor = lr_imgs[i].unsqueeze(0) # 取出单张并增加 batch 维度
                    
# #                     for k in [1, 2, 3]: # k=1(90度), k=2(180度), k=3(270度)
# #                         # 1. 旋转输入
# #                         lr_rot = torch.rot90(lr_tensor, k, [2, 3])
                        
# #                         # 2. 再次推理
# #                         sr_rot = model(lr_rot)
                        
# #                         # 3. 把结果转回来 (其实不转回来也能扫，二维码无视方向，但也转回来保险点)
# #                         sr_rot = torch.rot90(sr_rot, -k, [2, 3])
                        
# #                         # 4. 再次尝试扫描
# #                         sr_rot_np = (sr_rot.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
# #                         sr_rot_pil = Image.fromarray(sr_rot_np, mode='L')
# #                         tta_ok, tta_txt = try_scan(sr_rot_pil)
                        
# #                         if tta_ok and tta_txt == hr_txt:
# #                             stats["sr_tta_success"] += 1 # 记在 TTA 功劳簿上
# #                             final_success = True
# #                             break # 只要有一个角度成功就行

# #                 if final_success and not lr_ok:
# #                     stats["recovered"] += 1

# #     # === 报告 ===
# #     total_sr_success = stats['sr_success'] + stats['sr_tta_success']
# #     sr_acc = (total_sr_success / stats['valid_hr']) * 100
    
# #     print("\n" + "="*40)
# #     print("📊 TTA EVALUATION REPORT")
# #     print("="*40)
# #     print(f"Total Valid HR:            {stats['valid_hr']}")
# #     print(f"Standard SR Success:       {stats['sr_success']}")
# #     print(f"🔥 Bonus TTA Rescued:     {stats['sr_tta_success']} (Extra gains!)")
# #     print("-" * 40)
# #     print(f"🏆 Final Recognition Rate:  {sr_acc:.2f}%")
# #     print("="*40)

# # if __name__ == "__main__":
# #     evaluate()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from pyzbar.pyzbar import decode
# from PIL import Image
# import numpy as np
# import os
# import cv2

# # 引用你的项目模块
# from dataset import QRSRDataset
# from model import QRSuperResolutionNet

# # ================= 配置 =================
# CONFIG = {
#     "val_lr_dir": "dataset_4/val/lr",
#     "val_hr_dir": "dataset_4/val/hr",
#     # 确保这里是你训练得最好的那个 GAN 模型
#     "model_path": "checkpoints_gan/gan_generator_epoch70.pth", 
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "batch_size": 16,  
#     "num_blocks": 16
# }
# # =======================================

# def try_scan(pil_img):
#     """强力扫码函数：尝试原图、二值化、锐化三种方式"""
#     # 1. 直接扫
#     res = decode(pil_img)
#     if res: return True, res[0].data.decode("utf-8")
    
#     # 2. 二值化后扫 (Otsu)
#     cv_img = np.array(pil_img)
#     try:
#         _, binary = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         res = decode(Image.fromarray(binary))
#         if res: return True, res[0].data.decode("utf-8")
#     except:
#         pass # 防止空图报错
    
#     # 3. 锐化后扫
#     try:
#         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#         sharpened = cv2.filter2D(cv_img, -1, kernel)
#         res = decode(Image.fromarray(sharpened))
#         if res: return True, res[0].data.decode("utf-8")
#     except:
#         pass

#     return False, None

# def evaluate():
#     print(f"Loading model from {CONFIG['model_path']} ...")
#     device = torch.device(CONFIG['device'])
    
#     model = QRSuperResolutionNet(num_blocks=CONFIG['num_blocks']).to(device)
#     # 加载权重 (兼容不同的保存格式)
#     checkpoint = torch.load(CONFIG['model_path'], map_location=device)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model.load_state_dict(checkpoint)
#     model.eval()

#     # 加载数据
#     val_dataset = QRSRDataset(CONFIG['val_lr_dir'], CONFIG['val_hr_dir'], augment=False, preload=False)
#     val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

#     # === 统计计数器 ===
#     stats = {
#         "valid_hr": 0,         # 有效样本总数 (分母)
#         "lr_ok_count": 0,      # LR 原图就能扫出的
#         "sr_std_ok_count": 0,  # SR 标准推理能扫出的
#         "sr_tta_ok_count": 0,  # SR 标准不行，但 TTA 救回来的
#         "final_ok_count": 0,   # 最终通过数 (SR标准 + TTA救回)
        
#         "recovered": 0,        # 真正被救回的 (LR失败 -> Final成功)
#         "broken": 0            # 被搞坏的 (LR成功 -> Final失败)
#     }

#     print("🚀 Starting Comprehensive Evaluation (LR vs SR vs TTA)...")
    
#     with torch.no_grad():
#         for lr_imgs, hr_imgs in tqdm(val_loader, unit="batch"):
#             lr_imgs = lr_imgs.to(device)
#             batch_size = lr_imgs.size(0)

#             # 1. 模型推理 (标准方向)
#             sr_imgs = model(lr_imgs)

#             for i in range(batch_size):
#                 # --- A. 检查 Ground Truth (HR) ---
#                 hr_pil = Image.fromarray((hr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
#                 hr_ok, hr_txt = try_scan(hr_pil)
                
#                 if not hr_ok: 
#                     continue # 连原图都扫不出，跳过，不计入总数
                
#                 stats["valid_hr"] += 1
                
#                 # --- B. 检查 LR (低清原图) ---
#                 lr_pil = Image.fromarray((lr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
#                 lr_ok, lr_txt = try_scan(lr_pil)
#                 if lr_ok and lr_txt == hr_txt:
#                     stats["lr_ok_count"] += 1
#                     is_lr_pass = True
#                 else:
#                     is_lr_pass = False

#                 # --- C. 检查 SR (标准推理) ---
#                 sr_img_np = (sr_imgs[i].squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
#                 sr_pil = Image.fromarray(sr_img_np, mode='L')
#                 sr_ok, sr_txt = try_scan(sr_pil)

#                 is_final_pass = False # 最终是否通过

#                 if sr_ok and sr_txt == hr_txt:
#                     stats["sr_std_ok_count"] += 1
#                     is_final_pass = True
#                 else:
#                     # --- D. 激活 TTA (旋转救援) ---
#                     # 如果标准推理失败，尝试旋转 90/180/270 度
#                     tta_rescued = False
#                     lr_tensor = lr_imgs[i].unsqueeze(0)
                    
#                     for k in [1, 2, 3]:
#                         lr_rot = torch.rot90(lr_tensor, k, [2, 3])
#                         sr_rot = model(lr_rot) # 再次推理
#                         sr_rot_np = (sr_rot.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
#                         # 注意：这里不需要把图转回去也能扫，甚至转回去因为插值可能变糊，直接扫旋转图即可
#                         sr_rot_pil = Image.fromarray(sr_rot_np, mode='L')
#                         tta_ok, tta_txt = try_scan(sr_rot_pil)
                        
#                         if tta_ok and tta_txt == hr_txt:
#                             stats["sr_tta_ok_count"] += 1
#                             is_final_pass = True
#                             tta_rescued = True
#                             break # 救回一个就行
                
#                 # --- E. 统计 Recovered / Broken ---
#                 if is_final_pass:
#                     stats["final_ok_count"] += 1
                
#                 # 统计逻辑：
#                 # Recovered: LR 失败 -> Final 成功
#                 if not is_lr_pass and is_final_pass:
#                     stats["recovered"] += 1
                
#                 # Broken: LR 成功 -> Final 失败 (模型帮倒忙)
#                 if is_lr_pass and not is_final_pass:
#                     stats["broken"] += 1

#     # === 生成终极报表 ===
#     N = stats['valid_hr']
#     if N == 0:
#         print("❌ No valid HR images found.")
#         return

#     lr_rate = (stats['lr_ok_count'] / N) * 100
#     sr_std_rate = (stats['sr_std_ok_count'] / N) * 100
#     final_rate = (stats['final_ok_count'] / N) * 100
    
#     print("\n" + "="*50)
#     print("📊 COMPREHENSIVE EVALUATION REPORT")
#     print("="*50)
#     print(f"Total Valid HR Images (Base):    {N}")
#     print("-" * 50)
#     print(f"1️⃣  Baseline (Low Res):         {lr_rate:.2f}%  ({stats['lr_ok_count']}/{N})")
#     print(f"2️⃣  GAN Model (Standard):       {sr_std_rate:.2f}%  ({stats['sr_std_ok_count']}/{N})")
#     print(f"3️⃣  + TTA Rotation Boost:       +{stats['sr_tta_ok_count']} images rescued")
#     print("-" * 50)
#     print(f"🏆 Final Recognition Rate:      {final_rate:.2f}%  ({stats['final_ok_count']}/{N})")
#     print("="*50)
#     print("❤️  Survival Analysis:")
#     print(f"🚑 Recovered (Fail -> Pass):    {stats['recovered']} images")
#     print(f"💔 Broken (Pass -> Fail):       {stats['broken']} images")
#     print("="*50)

# if __name__ == "__main__":
#     evaluate()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
import os
import cv2

# 引用你的项目模块
from dataset import QRSRDataset
from model import QRSuperResolutionNet

# ================= 配置区域 =================
CONFIG = {
    # ⚠️ 请确保这里指向你生成的最新数据集路径
    "val_lr_dir": "dataset_4/val/lr",
    "val_hr_dir": "dataset_4/val/hr",
    
    # ⚠️ 请确保这里指向你效果最好的 GAN 模型
    "model_path": "checkpoints_gan/gan_generator_epoch70.pth", 
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,  
    "num_blocks": 16
}
# ===========================================

def try_scan_robust(pil_img):
    """
    终极扫码函数：尝试多种后处理手段
    解决了 GAN 生成图像中常见的“边缘粘连”和“微弱噪点”问题
    """
    # 1. 转为 Numpy 格式
    img_np = np.array(pil_img)
    
    # === 策略 A: 原始扫描 ===
    # 很多时候直接扫效果最好
    res = decode(pil_img)
    if res: return True, res[0].data.decode("utf-8")
    
    # === 策略 B: Otsu 自动二值化 ===
    # 解决光照不均和灰度值不明确的问题
    try:
        _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = decode(Image.fromarray(binary))
        if res: return True, res[0].data.decode("utf-8")
    except: pass
    
    # === 策略 C: 锐化 (Sharpening) ===
    # 解决边缘模糊问题
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(img_np, -1, kernel)
        res = decode(Image.fromarray(sharp))
        if res: return True, res[0].data.decode("utf-8")
    except: pass

    # === 策略 D: 形态学开运算 (Opening) ===
    # 核心修复手段：断开细小的像素粘连 (GAN 常见的伪影)
    try:
        # 使用 2x2 的微小核，避免破坏有用信息
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
        res = decode(Image.fromarray(opening))
        if res: return True, res[0].data.decode("utf-8")
    except: pass
    
    # === 策略 E: 形态学闭运算 (Closing) ===
    # 核心修复手段：填补黑色块内的细小空洞
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closing = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
        res = decode(Image.fromarray(closing))
        if res: return True, res[0].data.decode("utf-8")
    except: pass

    return False, None

def evaluate():
    print(f"🚀 Loading model from {CONFIG['model_path']} ...")
    device = torch.device(CONFIG['device'])
    
    # 加载模型
    model = QRSuperResolutionNet(num_blocks=CONFIG['num_blocks']).to(device)
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
        
    model.eval()

    # 加载数据集
    if not os.path.exists(CONFIG['val_lr_dir']):
        print(f"❌ Error: Dataset path not found: {CONFIG['val_lr_dir']}")
        return

    # preload=False 节省内存，augment=False 保证结果稳定
    val_dataset = QRSRDataset(CONFIG['val_lr_dir'], CONFIG['val_hr_dir'], augment=False, preload=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 统计数据
    stats = {
        "valid_hr": 0,         # 有效样本总数
        "lr_ok": 0,            # LR 原图能扫出的
        "sr_std_ok": 0,        # SR 标准推理能扫出的
        "sr_tta_ok": 0,        # TTA 救回来的
        "final_ok": 0,         # 最终通过数
        "recovered": 0,        # 真正被系统救回的
        "broken": 0            # 被搞坏的
    }

    print("🕵️ Starting Ultimate Evaluation (Scan+Morphology+TTA)...")
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, unit="batch"):
            lr_imgs = lr_imgs.to(device)
            batch_size = lr_imgs.size(0)

            # 1. 模型推理 (标准方向)
            sr_imgs = model(lr_imgs)

            for i in range(batch_size):
                # --- A. 验证 HR (基准) ---
                hr_pil = Image.fromarray((hr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
                # 对 HR 也使用强力扫描，确保基准公正
                hr_ok, hr_txt = try_scan_robust(hr_pil)
                
                if not hr_ok: 
                    continue # 连原图都扫不出，跳过
                
                stats["valid_hr"] += 1
                
                # --- B. 验证 LR (低清) ---
                lr_pil = Image.fromarray((lr_imgs[i].squeeze().cpu().numpy() * 255).astype(np.uint8), mode='L')
                lr_ok, lr_txt = try_scan_robust(lr_pil)
                
                if lr_ok and lr_txt == hr_txt:
                    stats["lr_ok"] += 1
                    is_lr_pass = True
                else:
                    is_lr_pass = False

                # --- C. 验证 SR (标准推理) ---
                sr_img_np = (sr_imgs[i].squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                sr_pil = Image.fromarray(sr_img_np, mode='L')
                sr_ok, sr_txt = try_scan_robust(sr_pil)

                is_final_pass = False 

                if sr_ok and sr_txt == hr_txt:
                    stats["sr_std_ok"] += 1
                    is_final_pass = True
                else:
                    # --- D. 激活 TTA (旋转救援) ---
                    lr_tensor = lr_imgs[i].unsqueeze(0)
                    
                    for k in [1, 2, 3]: # 旋转 90, 180, 270
                        lr_rot = torch.rot90(lr_tensor, k, [2, 3])
                        sr_rot = model(lr_rot)
                        sr_rot_np = (sr_rot.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        
                        # 直接扫旋转后的图，不需要转回来（转回来会有插值损失）
                        sr_rot_pil = Image.fromarray(sr_rot_np, mode='L')
                        tta_ok, tta_txt = try_scan_robust(sr_rot_pil)
                        
                        if tta_ok and tta_txt == hr_txt:
                            stats["sr_tta_ok"] += 1
                            is_final_pass = True
                            break # 救回一个就行
                
                # --- E. 统计 Recovered / Broken ---
                if is_final_pass:
                    stats["final_ok"] += 1
                
                # 统计逻辑
                if not is_lr_pass and is_final_pass:
                    stats["recovered"] += 1
                
                if is_lr_pass and not is_final_pass:
                    stats["broken"] += 1

    # === 生成最终报表 ===
    N = stats['valid_hr']
    if N == 0:
        print("❌ No valid HR images found.")
        return

    lr_rate = (stats['lr_ok'] / N) * 100
    sr_std_rate = (stats['sr_std_ok'] / N) * 100
    final_rate = (stats['final_ok'] / N) * 100
    
    print("\n" + "="*50)
    print("📊 ULTIMATE EVALUATION REPORT")
    print("="*50)
    print(f"Total Valid HR Images:           {N}")
    print("-" * 50)
    print(f"1️⃣  Baseline (Low Res):         {lr_rate:.2f}%  ({stats['lr_ok']}/{N})")
    print(f"2️⃣  GAN Model (Standard):       {sr_std_rate:.2f}%  ({stats['sr_std_ok']}/{N})")
    print(f"3️⃣  + TTA Rotation Boost:       +{stats['sr_tta_ok']} images rescued")
    print("-" * 50)
    print(f"🏆 Final Recognition Rate:      {final_rate:.2f}%  ({stats['final_ok']}/{N})")
    print("="*50)
    print("❤️  Impact Analysis:")
    print(f"🚑 Recovered (Fail -> Pass):    {stats['recovered']} images")
    print(f"💔 Broken (Pass -> Fail):       {stats['broken']} images")
    print("="*50)

if __name__ == "__main__":
    evaluate()