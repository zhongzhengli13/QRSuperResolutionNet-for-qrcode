import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
from pyzbar.pyzbar import decode

# 导入你的模型
from model import QRSuperResolutionNet

def test_single(image_path, model_path, output_path="result.png", device_name="cuda"):
    # 1. 设备配置
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型
    # 注意：必须与训练时的配置一致，我们之前改成了 num_blocks=16
    model = QRSuperResolutionNet(num_blocks=16).to(device)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容只保存了 state_dict 或保存了完整 checkpoint 的情况
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()
    print("✅ Model loaded successfully.")

    # 3. 图像预处理
    if not os.path.exists(image_path):
        print(f"❌ Error: Input image not found at {image_path}")
        return

    # 读取为灰度图
    img_pil = Image.open(image_path).convert("L")
    
    # 转换为 Tensor
    transform = transforms.Compose([
        transforms.ToTensor() # 归一化到 [0, 1]
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 4. 推理 (Inference)
    with torch.no_grad():
        sr_tensor = model(img_tensor)

    # 5. 后处理结果
    # 从 Tensor 转回 Numpy
    sr_img = sr_tensor.squeeze().cpu().numpy()
    sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)

    # 6. 生成对比图
    # 读取原始图片用于 OpenCV 处理
    lr_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 将 LR 图片放大到 SR 的尺寸以便拼接
    # 使用 INTER_NEAREST (最近邻) 保持马赛克感，这样对比更强烈
    h, w = sr_img.shape
    lr_resized = cv2.resize(lr_cv, (w, h), interpolation=cv2.INTER_NEAREST)

    # 左右拼接 (左：原图放大，右：模型修复)
    combined = np.hstack((lr_resized, sr_img))

    # 7. 尝试识别二维码内容
    # 尝试识别原图
    decoded_lr = decode(Image.fromarray(lr_resized))
    lr_txt = decoded_lr[0].data.decode("utf-8") if decoded_lr else "Unreadable"

    # 尝试识别修复图
    decoded_sr = decode(Image.fromarray(sr_img))
    sr_txt = decoded_sr[0].data.decode("utf-8") if decoded_sr else "Unreadable"
    
    # 在图片上写字显示识别状态
    # 为了写字，转为 BGR 彩色空间
    combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # 绘制分割线
    cv2.line(combined_color, (w, 0), (w, h), (0, 0, 255), 2)
    
    print(f"\n🔍 Scan Results:")
    print(f"Original: {lr_txt}")
    print(f"Restored: {sr_txt}")

    if sr_txt != "Unreadable":
        print("🎉 Success! The restored QR code is readable.")
    else:
        print("⚠️ Warning: The restored QR code is still unreadable.")

    # 8. 保存
    cv2.imwrite(output_path, combined_color)
    print(f"✅ Result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QR Code Super Resolution on a single image")
    parser.add_argument("--img", type=str, required=True, help="Path to the low-resolution input image")
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/mine-qr-code/mine_model_v2/checkpoints_gan/gan_generator_epoch70.pth", help="Path to the .pth model file")
    parser.add_argument("--out", type=str, default="result.png", help="Path to save the result image")
    
    args = parser.parse_args()
    
    test_single(args.img, args.model, args.out)