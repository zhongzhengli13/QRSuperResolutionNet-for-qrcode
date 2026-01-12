# # # # import os
# # # # import cv2
# # # # import numpy as np
# # # # import random
# # # # import shutil
# # # # from glob import glob
# # # # from tqdm import tqdm
# # # # from sklearn.model_selection import train_test_split

# # # # # ================= 配置区域 =================
# # # # # 原始数据集路径 (你下载的 Kaggle 数据解压后的文件夹路径)
# # # # SOURCE_DIR = "/root/autodl-tmp/mine-qr-code/qr_dataset_kaggle"  

# # # # # 输出数据集路径
# # # # OUTPUT_DIR = "./dataset"

# # # # # 目标尺寸
# # # # HR_SIZE = (256, 256)
# # # # LR_SIZE = (64, 64)

# # # # # 随机种子，保证复现性
# # # # SEED = 42
# # # # random.seed(SEED)
# # # # np.random.seed(SEED)
# # # # # ===========================================

# # # # def add_gaussian_blur(img):
# # # #     """随机添加高斯模糊"""
# # # #     # 随机选择核大小 (3, 5, 7)
# # # #     ksize = random.choice([3, 5])
# # # #     # 随机 sigma
# # # #     sigma = random.uniform(0.5, 1.5)
# # # #     return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# # # # def add_gaussian_noise(img):
# # # #     """添加高斯噪声"""
# # # #     row, col = img.shape
# # # #     mean = 0
# # # #     # 随机噪声强度
# # # #     var = random.uniform(10, 50) 
# # # #     sigma = var ** 0.5
# # # #     gauss = np.random.normal(mean, sigma, (row, col))
# # # #     gauss = gauss.reshape(row, col)
# # # #     noisy = img + gauss
# # # #     return np.clip(noisy, 0, 255).astype(np.uint8)

# # # # def add_jpeg_compression(img):
# # # #     """模拟 JPEG 压缩伪影"""
# # # #     # 随机质量因子，越低伪影越重 (30-90)
# # # #     quality = random.randint(30, 85)
# # # #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
# # # #     _, encimg = cv2.imencode('.jpg', img, encode_param)
# # # #     decimg = cv2.imdecode(encimg, 0) # 0 for grayscale
# # # #     return decimg

# # # # def degrade_image(hr_img):
# # # #     """
# # # #     将高清图退化为低清图
# # # #     流程: 下采样 -> (随机模糊) -> (随机噪声) -> (随机JPEG压缩)
# # # #     """
# # # #     # 1. 下采样 (Downsampling) 到 LR 尺寸
# # # #     # 使用 INTER_CUBIC 模拟较好的缩放，或者 INTER_LINEAR
# # # #     lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=cv2.INTER_CUBIC)

# # # #     # 2. 随机混合退化效果
    
# # # #     # 50% 概率添加模糊 (模拟对焦不准)
# # # #     if random.random() < 0.5:
# # # #         lr_img = add_gaussian_blur(lr_img)

# # # #     # 50% 概率添加噪声 (模拟低光照噪点)
# # # #     if random.random() < 0.5:
# # # #         lr_img = add_gaussian_noise(lr_img)
        
# # # #     # 70% 概率添加 JPEG 压缩 (模拟图片传输压缩)
# # # #     if random.random() < 0.7:
# # # #         lr_img = add_jpeg_compression(lr_img)

# # # #     return lr_img

# # # # def process_files(file_list, subset_name):
# # # #     """
# # # #     处理文件列表并保存到对应目录
# # # #     subset_name: 'train' 或 'val'
# # # #     """
# # # #     save_hr_dir = os.path.join(OUTPUT_DIR, subset_name, "hr")
# # # #     save_lr_dir = os.path.join(OUTPUT_DIR, subset_name, "lr")
    
# # # #     os.makedirs(save_hr_dir, exist_ok=True)
# # # #     os.makedirs(save_lr_dir, exist_ok=True)
    
# # # #     print(f"🔄 Processing {subset_name} set ({len(file_list)} images)...")
    
# # # #     for idx, file_path in enumerate(tqdm(file_list)):
# # # #         # 读取图片（灰度模式）
# # # #         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
# # # #         if img is None:
# # # #             continue
            
# # # #         # 1. 确保 HR 尺寸统一 (如果原图不是 256x256，强制调整)
# # # #         if img.shape != HR_SIZE:
# # # #             img_hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
# # # #         else:
# # # #             img_hr = img
            
# # # #         # 2. 生成 LR 图片
# # # #         img_lr = degrade_image(img_hr)
        
# # # #         # 3. 保存
# # # #         # 统一重命名，方便管理，例如: 00001.png
# # # #         filename = f"{idx:05d}.png"
        
# # # #         # 你的 Dataset 代码里可能需要 lr_ 前缀来匹配，这里我们直接存为同名文件
# # # #         # 或者按照你之前的习惯: lr_xxxxx.png 和 hr_xxxxx.png
# # # #         # 这里我改为同名文件，dataset 读取时更简单，或者你可以保留前缀
        
# # # #         cv2.imwrite(os.path.join(save_hr_dir, f"hr_{filename}"), img_hr)
# # # #         cv2.imwrite(os.path.join(save_lr_dir, f"lr_{filename}"), img_lr)

# # # # def main():
# # # #     # 1. 获取所有图片路径
# # # #     # 假设 Kaggle 数据集里全是 .png 或 .jpg
# # # #     all_files = glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True) + \
# # # #                 glob(os.path.join(SOURCE_DIR, "**", "*.jpg"), recursive=True)
    
# # # #     if len(all_files) == 0:
# # # #         print(f"❌ Error: No images found in {SOURCE_DIR}")
# # # #         return

# # # #     print(f"📦 Found {len(all_files)} images total.")

# # # #     # 2. 划分训练集和测试集 (90% 训练, 10% 验证)
# # # #     train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

# # # #     # 3. 执行处理
# # # #     if os.path.exists(OUTPUT_DIR):
# # # #         print(f"⚠️ Warning: Output directory '{OUTPUT_DIR}' already exists. Merging/Overwriting...")
    
# # # #     process_files(train_files, "train")
# # # #     process_files(val_files, "val")
    
# # # #     print("\n✅ Dataset preparation complete!")
# # # #     print(f"Train HR: {os.path.join(OUTPUT_DIR, 'train/hr')}")
# # # #     print(f"Train LR: {os.path.join(OUTPUT_DIR, 'train/lr')}")

# # # # if __name__ == "__main__":
# # # #     main()

# # # import os
# # # import cv2
# # # import numpy as np
# # # import random
# # # import shutil
# # # from glob import glob
# # # from tqdm import tqdm
# # # from sklearn.model_selection import train_test_split

# # # # ================= 配置区域 =================
# # # SOURCE_DIR = "/root/autodl-tmp/mine-qr-code/qr_dataset_kaggle"
# # # OUTPUT_DIR = "./dataset_1"

# # # HR_SIZE = (256, 256)
# # # LR_SIZE = (64, 64)

# # # SEED = 42
# # # random.seed(SEED)
# # # np.random.seed(SEED)
# # # # ===========================================

# # # def add_gaussian_blur(img):
# # #     """随机高斯模糊 (模拟失焦)"""
# # #     ksize = random.choice([3, 5, 7])
# # #     sigma = random.uniform(0.5, 2.0)
# # #     return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# # # def add_motion_blur(img):
# # #     """随机运动模糊 (模拟手抖)"""
# # #     size = random.randint(3, 7)
# # #     kernel = np.zeros((size, size))
# # #     # 随机选择水平或垂直方向
# # #     if random.random() < 0.5:
# # #         kernel[int((size-1)/2), :] = np.ones(size)
# # #     else:
# # #         kernel[:, int((size-1)/2)] = np.ones(size)
# # #     kernel = kernel / size
# # #     return cv2.filter2D(img, -1, kernel)

# # # def add_noise(img):
# # #     """混合噪声: 高斯噪声 + 椒盐噪声 (模拟低光噪点和坏点)"""
# # #     # 1. 高斯噪声
# # #     if random.random() < 0.7:
# # #         row, col = img.shape
# # #         mean = 0
# # #         var = random.uniform(10, 80) # 加大噪声方差
# # #         sigma = var ** 0.5
# # #         gauss = np.random.normal(mean, sigma, (row, col))
# # #         img = img.astype(np.float32) + gauss
# # #         img = np.clip(img, 0, 255)

# # #     # 2. 椒盐噪声 (模拟斑点)
# # #     if random.random() < 0.5:
# # #         prob = random.uniform(0.01, 0.05)
# # #         thres = 1 - prob
# # #         output = np.zeros(img.shape, np.uint8)
# # #         # 生成随机矩阵
# # #         rdn = np.random.random(img.shape)
# # #         # 黑点 (Pepper)
# # #         img[rdn < prob] = 0
# # #         # 白点 (Salt)
# # #         img[rdn > thres] = 255
        
# # #     return img.astype(np.uint8)

# # # def add_fading_and_contrast(img):
# # #     """模拟褪色、低对比度、泛白"""
# # #     img = img.astype(np.float32)
    
# # #     # 随机对比度 (0.5 ~ 0.9 表示降低对比度)
# # #     alpha = random.uniform(0.4, 0.9) 
# # #     # 随机亮度 (0 ~ 60 表示泛白/过曝)
# # #     beta = random.uniform(10, 80)
    
# # #     img = img * alpha + beta
# # #     return np.clip(img, 0, 255).astype(np.uint8)

# # # def add_shadow(img):
# # #     """模拟光照不均 (阴影)"""
# # #     h, w = img.shape
# # #     # 创建一个渐变蒙版
# # #     top_x = random.randint(0, w)
# # #     bot_x = random.randint(0, w)
    
# # #     # 生成一个从 0.4 到 1.0 的线性渐变
# # #     mask = np.indices((h, w))[0]
# # #     mask = (mask - mask.min()) / (mask.max() - mask.min()) # 0 to 1
    
# # #     direction = random.choice([0, 1, 2, 3]) # 上下左右四个方向渐变
# # #     if direction == 0: shadow_mask = mask
# # #     elif direction == 1: shadow_mask = 1 - mask
# # #     elif direction == 2: shadow_mask = mask.T
# # #     else: shadow_mask = 1 - mask.T
    
# # #     # 调整阴影深浅
# # #     shadow_intensity = random.uniform(0.3, 0.8) # 越小越黑
# # #     shadow_mask = shadow_mask * (1 - shadow_intensity) + shadow_intensity
    
# # #     img = img.astype(np.float32) * shadow_mask
# # #     return np.clip(img, 0, 255).astype(np.uint8)

# # # def add_occlusion(img):
# # #     """模拟遮挡/污渍/缺失"""
# # #     h, w = img.shape
# # #     # 随机生成 1~5 个遮挡块
# # #     num_spots = random.randint(1, 5)
    
# # #     img_out = img.copy()
# # #     for _ in range(num_spots):
# # #         # 随机大小
# # #         size = random.randint(5, 15)
# # #         # 随机位置
# # #         x = random.randint(0, w - size)
# # #         y = random.randint(0, h - size)
        
# # #         # 随机颜色 (黑污渍 或者 白破损)
# # #         color = random.choice([0, 255, 128]) 
        
# # #         # 随机形状：矩形或圆形
# # #         if random.random() < 0.5:
# # #             cv2.rectangle(img_out, (x, y), (x+size, y+size), color, -1)
# # #         else:
# # #             cv2.circle(img_out, (x+size//2, y+size//2), size//2, color, -1)
            
# # #     return img_out

# # # def add_jpeg_compression(img):
# # #     """JPEG 压缩"""
# # #     quality = random.randint(20, 80) # 进一步降低质量下限
# # #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
# # #     _, encimg = cv2.imencode('.jpg', img, encode_param)
# # #     decimg = cv2.imdecode(encimg, 0)
# # #     return decimg

# # # def degrade_image(hr_img):
# # #     """
# # #     复杂的退化管线
# # #     注意：HR 图像已经是 256x256，我们先将其缩小到 LR_SIZE (64x64)，
# # #     然后在 LR 尺寸上施加破坏。
# # #     """
# # #     # 1. 下采样
# # #     # 随机使用不同的插值方式，防止模型过拟合某一种下采样
# # #     interp = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA])
# # #     lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=interp)

# # #     # === 物理与环境退化 (按概率叠加) ===

# # #     # 2. 褪色/低对比度 (60% 概率)
# # #     if random.random() < 0.6:
# # #         lr_img = add_fading_and_contrast(lr_img)
        
# # #     # 3. 光照不均/阴影 (50% 概率)
# # #     if random.random() < 0.5:
# # #         lr_img = add_shadow(lr_img)

# # #     # 4. 遮挡/污渍 (40% 概率) - 这个很难恢复，所以概率不要太高
# # #     if random.random() < 0.4:
# # #         lr_img = add_occlusion(lr_img)

# # #     # === 相机与传感器退化 ===

# # #     # 5. 模糊 (70% 概率 - 高斯或运动模糊)
# # #     if random.random() < 0.7:
# # #         if random.random() < 0.7:
# # #             lr_img = add_gaussian_blur(lr_img)
# # #         else:
# # #             lr_img = add_motion_blur(lr_img)

# # #     # 6. 噪声 (80% 概率 - 几乎所有低清图都有噪点)
# # #     if random.random() < 0.8:
# # #         lr_img = add_noise(lr_img)
        
# # #     # 7. JPEG 压缩 (80% 概率)
# # #     if random.random() < 0.8:
# # #         lr_img = add_jpeg_compression(lr_img)

# # #     return lr_img

# # # def process_files(file_list, subset_name):
# # #     save_hr_dir = os.path.join(OUTPUT_DIR, subset_name, "hr")
# # #     save_lr_dir = os.path.join(OUTPUT_DIR, subset_name, "lr")
    
# # #     os.makedirs(save_hr_dir, exist_ok=True)
# # #     os.makedirs(save_lr_dir, exist_ok=True)
    
# # #     print(f"🔄 Processing {subset_name} set ({len(file_list)} images)...")
    
# # #     for idx, file_path in enumerate(tqdm(file_list)):
# # #         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# # #         if img is None: continue
            
# # #         if img.shape != HR_SIZE:
# # #             img_hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
# # #         else:
# # #             img_hr = img
            
# # #         img_lr = degrade_image(img_hr)
        
# # #         filename = f"{idx:05d}.png"
# # #         cv2.imwrite(os.path.join(save_hr_dir, f"hr_{filename}"), img_hr)
# # #         cv2.imwrite(os.path.join(save_lr_dir, f"lr_{filename}"), img_lr)

# # # def main():
# # #     all_files = glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True) + \
# # #                 glob(os.path.join(SOURCE_DIR, "**", "*.jpg"), recursive=True)
    
# # #     if len(all_files) == 0:
# # #         print(f"❌ Error: No images found in {SOURCE_DIR}")
# # #         return

# # #     print(f"📦 Found {len(all_files)} images total.")
# # #     train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

# # #     if os.path.exists(OUTPUT_DIR):
# # #         print(f"⚠️ Warning: Output directory '{OUTPUT_DIR}' already exists. Merging/Overwriting...")
    
# # #     process_files(train_files, "train")
# # #     process_files(val_files, "val")
    
# # #     print("\n✅ Dataset preparation complete!")

# # # if __name__ == "__main__":
# # #     main()

# # import os
# # import cv2
# # import numpy as np
# # import random
# # import shutil
# # from glob import glob
# # from tqdm import tqdm
# # from sklearn.model_selection import train_test_split

# # # ================= 配置区域 =================
# # SOURCE_DIR = "/root/autodl-tmp/mine-qr-code/qr_dataset_kaggle"
# # OUTPUT_DIR = "./dataset_2" # 建议换个文件夹名，避免混淆

# # HR_SIZE = (256, 256)
# # LR_SIZE = (64, 64)

# # SEED = 42
# # random.seed(SEED)
# # np.random.seed(SEED)
# # # ===========================================

# # def add_gaussian_blur(img):
# #     """随机高斯模糊 (模拟失焦) - 降低了 Sigma 上限"""
# #     ksize = random.choice([3, 5])
# #     # sigma 太大会把二维码糊成一团灰，根本无法识别
# #     sigma = random.uniform(0.5, 1.2) 
# #     return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# # def add_motion_blur(img):
# #     """随机运动模糊"""
# #     size = random.randint(2, 5) # 减小核大小，避免过度拖影
# #     kernel = np.zeros((size, size))
# #     if random.random() < 0.5:
# #         kernel[int((size-1)/2), :] = np.ones(size)
# #     else:
# #         kernel[:, int((size-1)/2)] = np.ones(size)
# #     kernel = kernel / size
# #     return cv2.filter2D(img, -1, kernel)

# # def add_noise(img):
# #     """高斯噪声 (模拟低光环境)"""
# #     if random.random() < 0.7:
# #         row, col = img.shape
# #         mean = 0
# #         var = random.uniform(5, 30) # 降低噪声方差，太大的噪声会破坏二值化特征
# #         sigma = var ** 0.5
# #         gauss = np.random.normal(mean, sigma, (row, col))
# #         img = img.astype(np.float32) + gauss
# #         img = np.clip(img, 0, 255)
# #     return img.astype(np.uint8)

# # def add_dirt_and_scratches(img):
# #     """
# #     替代原来的 add_occlusion
# #     模拟：灰尘、细微划痕、墨点
# #     """
# #     h, w = img.shape
# #     img_out = img.copy()
    
# #     # 1. 模拟灰尘/斑点 (Dot)
# #     # 数量多，但尺寸极小 (1-2像素)
# #     num_dust = random.randint(5, 20)
# #     for _ in range(num_dust):
# #         x = random.randint(0, w-1)
# #         y = random.randint(0, h-1)
# #         # 黑色或深灰色污渍
# #         color = random.randint(0, 100)
# #         # 只修改 1个像素 或 2x2 小块
# #         if random.random() < 0.8:
# #             img_out[y, x] = color
# #         elif x < w-1 and y < h-1:
# #             img_out[y:y+2, x:x+2] = color

# #     # 2. 模拟划痕 (Scratch)
# #     # 随机画细线
# #     if random.random() < 0.4:
# #         num_scratch = random.randint(1, 3)
# #         for _ in range(num_scratch):
# #             x1, y1 = random.randint(0, w), random.randint(0, h)
# #             x2, y2 = random.randint(0, w), random.randint(0, h)
# #             # 颜色随机 (可能是白色划痕或黑色笔迹)
# #             color = random.choice([0, 255])
# #             # 线宽必须是 1
# #             cv2.line(img_out, (x1, y1), (x2, y2), color, 1)

# #     return img_out

# # def add_glare(img):
# #     """模拟局部反光 (亮斑)"""
# #     h, w = img.shape
# #     # 创建一个高斯光斑
# #     x = random.randint(0, w)
# #     y = random.randint(0, h)
    
# #     # 制作光斑蒙版
# #     y_grid, x_grid = np.ogrid[:h, :w]
# #     # 光斑大小
# #     sigma = random.randint(5, 15) 
# #     mask = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
    
# #     # 混合
# #     intensity = random.uniform(0.3, 0.7) # 光斑亮度强度
# #     img = img.astype(np.float32)
# #     # 原图变亮
# #     img = img + (255 * mask * intensity)
# #     return np.clip(img, 0, 255).astype(np.uint8)

# # def add_fading(img):
# #     """整体褪色/对比度降低"""
# #     img = img.astype(np.float32)
# #     alpha = random.uniform(0.6, 0.95) # 不要降得太低
# #     beta = random.uniform(0, 30)
# #     img = img * alpha + beta
# #     return np.clip(img, 0, 255).astype(np.uint8)

# # def degrade_image(hr_img):
# #     """
# #     退化流程：
# #     HR (256) -> Resize (64) -> 物理瑕疵 -> 光学瑕疵 -> 压缩
# #     """
# #     # 1. 下采样
# #     interp = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_AREA])
# #     lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=interp)

# #     # === 物理瑕疵 (灰尘、划痕) ===
# #     # 概率较高，因为现实中镜头或屏幕总是不干净的
# #     if random.random() < 0.6:
# #         lr_img = add_dirt_and_scratches(lr_img)

# #     # === 环境光照 (反光、褪色) ===
# #     if random.random() < 0.4:
# #         lr_img = add_glare(lr_img) # 局部反光
    
# #     if random.random() < 0.4:
# #         lr_img = add_fading(lr_img) # 整体泛白

# #     # === 模糊与噪声 ===
# #     # 模糊不宜过重，否则二维码模块会粘连
# #     if random.random() < 0.6:
# #         if random.random() < 0.6:
# #             lr_img = add_gaussian_blur(lr_img)
# #         else:
# #             lr_img = add_motion_blur(lr_img)

# #     if random.random() < 0.7:
# #         lr_img = add_noise(lr_img)
        
# #     # === JPEG 压缩 ===
# #     if random.random() < 0.8:
# #         quality = random.randint(30, 90)
# #         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
# #         _, encimg = cv2.imencode('.jpg', lr_img, encode_param)
# #         lr_img = cv2.imdecode(encimg, 0)

# #     return lr_img

# # def process_files(file_list, subset_name):
# #     save_hr_dir = os.path.join(OUTPUT_DIR, subset_name, "hr")
# #     save_lr_dir = os.path.join(OUTPUT_DIR, subset_name, "lr")
    
# #     os.makedirs(save_hr_dir, exist_ok=True)
# #     os.makedirs(save_lr_dir, exist_ok=True)
    
# #     print(f"🔄 Processing {subset_name} set ({len(file_list)} images)...")
    
# #     for idx, file_path in enumerate(tqdm(file_list)):
# #         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# #         if img is None: continue
            
# #         if img.shape != HR_SIZE:
# #             img_hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
# #         else:
# #             img_hr = img
            
# #         img_lr = degrade_image(img_hr)
        
# #         filename = f"{idx:05d}.png"
# #         cv2.imwrite(os.path.join(save_hr_dir, f"hr_{filename}"), img_hr)
# #         cv2.imwrite(os.path.join(save_lr_dir, f"lr_{filename}"), img_lr)

# # def main():
# #     all_files = glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True) + \
# #                 glob(os.path.join(SOURCE_DIR, "**", "*.jpg"), recursive=True)
    
# #     if len(all_files) == 0:
# #         print(f"❌ Error: No images found in {SOURCE_DIR}")
# #         return

# #     print(f"📦 Found {len(all_files)} images total.")
# #     train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

# #     if os.path.exists(OUTPUT_DIR):
# #         print(f"⚠️ Warning: Output directory '{OUTPUT_DIR}' already exists. Merging/Overwriting...")
    
# #     process_files(train_files, "train")
# #     process_files(val_files, "val")
    
# #     print("\n✅ Dataset preparation complete!")

# # if __name__ == "__main__":
# #     main()

# import os
# import cv2
# import numpy as np
# import random
# import shutil
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split

# # ================= 配置区域 =================
# SOURCE_DIR = "/root/autodl-tmp/mine-qr-code/qr_dataset_kaggle"
# OUTPUT_DIR = "./dataset_3"  # 改名，防止覆盖

# HR_SIZE = (256, 256)
# LR_SIZE = (64, 64)

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# # ===========================================

# def add_heavy_gaussian_blur(img):
#     """
#     加强版高斯模糊
#     针对 64x64 图片，使用 5x5 或 7x7 的核已经非常模糊了
#     """
#     ksize = random.choice([5, 7]) 
#     # Sigma 提高到 1.5 ~ 3.5 (之前是 1.2)
#     # 这会让图像看起来非常“肉”，边缘完全化开
#     sigma = random.uniform(1.5, 3.5) 
#     return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# def add_heavy_motion_blur(img):
#     """加强版运动模糊"""
#     # 增加拖影长度到 5~10 像素 (对于 64宽度的图，10像素拖影是致命的)
#     size = random.randint(3,7) 
#     kernel = np.zeros((size, size))
#     if random.random() < 0.5:
#         kernel[int((size-1)/2), :] = np.ones(size)
#     else:
#         kernel[:, int((size-1)/2)] = np.ones(size)
#     kernel = kernel / size
#     return cv2.filter2D(img, -1, kernel)

# def add_low_res_blur(img):
#     """
#     [新增] 极低分辨率模糊
#     模拟：摄像头像素极低，或者数码变焦放大的效果
#     流程：64x64 -> 24x24 -> 64x64 (强制拉伸)
#     """
#     h, w = img.shape
#     # 随机缩放到极小尺寸
#     temp_size = random.randint(20, 32) 
    
#     # 缩小
#     small = cv2.resize(img, (temp_size, temp_size), interpolation=cv2.INTER_AREA)
#     # 放大回 LR 尺寸 (使用线性或立方插值，产生平滑的模糊感)
#     blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
#     return blurred

# def add_noise(img):
#     """高斯噪声"""
#     if random.random() < 0.6: # 提高概率
#         row, col = img.shape
#         mean = 0
#         var = random.uniform(10, 50) 
#         sigma = var ** 0.5
#         gauss = np.random.normal(mean, sigma, (row, col))
#         img = img.astype(np.float32) + gauss
#         img = np.clip(img, 0, 255)
#     return img.astype(np.uint8)

# def add_dirt_and_scratches(img):
#     """微小瑕疵 (灰尘/划痕)"""
#     h, w = img.shape
#     img_out = img.copy()
    
#     # 灰尘
#     num_dust = random.randint(5, 20)
#     for _ in range(num_dust):
#         x, y = random.randint(0, w-1), random.randint(0, h-1)
#         color = random.randint(0, 150)
#         img_out[y, x] = color

#     # 划痕
#     if random.random() < 0.5:
#         num_scratch = random.randint(1, 4)
#         for _ in range(num_scratch):
#             x1, y1 = random.randint(0, w), random.randint(0, h)
#             x2, y2 = random.randint(0, w), random.randint(0, h)
#             color = random.choice([0, 255])
#             cv2.line(img_out, (x1, y1), (x2, y2), color, 1)

#     return img_out

# def add_glare(img):
#     """反光"""
#     h, w = img.shape
#     x, y = random.randint(0, w), random.randint(0, h)
#     y_grid, x_grid = np.ogrid[:h, :w]
#     sigma = random.randint(8, 20) 
#     mask = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
#     intensity = random.uniform(0.2, 0.6) 
#     img = img.astype(np.float32) + (255 * mask * intensity)
#     return np.clip(img, 0, 255).astype(np.uint8)

# def add_fading(img):
#     """褪色"""
#     img = img.astype(np.float32)
#     alpha = random.uniform(0.5, 0.9) # 对比度降低
#     beta = random.uniform(10, 50)    # 亮度提高(泛白)
#     img = img * alpha + beta
#     return np.clip(img, 0, 255).astype(np.uint8)

# def degrade_image(hr_img):
#     """
#     究极模糊版退化管线
#     """
#     # 1. 基础下采样
#     lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=cv2.INTER_AREA)

#     # === 核心：三重模糊轰炸 ===
    
#     # 策略 A: 极低分辨率模糊 (模拟马赛克感) - 40% 概率独立发生
#     if random.random() < 0.4:
#         lr_img = add_low_res_blur(lr_img)
        
#     # 策略 B: 强力高斯/运动模糊 - 80% 概率发生 (大概率叠加)
#     if random.random() < 0.8:
#         if random.random() < 0.6:
#             lr_img = add_heavy_gaussian_blur(lr_img)
#         else:
#             lr_img = add_heavy_motion_blur(lr_img)

#     # === 物理与环境瑕疵 ===
#     if random.random() < 0.6:
#         lr_img = add_dirt_and_scratches(lr_img)

#     if random.random() < 0.5:
#         if random.random() < 0.5:
#             lr_img = add_glare(lr_img)
#         else:
#             lr_img = add_fading(lr_img)

#     # === 噪声与压缩 ===
#     # 模糊的图片通常伴随着噪声
#     lr_img = add_noise(lr_img)
        
#     if random.random() < 0.8:
#         quality = random.randint(25, 80)
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#         _, encimg = cv2.imencode('.jpg', lr_img, encode_param)
#         lr_img = cv2.imdecode(encimg, 0)

#     return lr_img

# def process_files(file_list, subset_name):
#     save_hr_dir = os.path.join(OUTPUT_DIR, subset_name, "hr")
#     save_lr_dir = os.path.join(OUTPUT_DIR, subset_name, "lr")
#     os.makedirs(save_hr_dir, exist_ok=True)
#     os.makedirs(save_lr_dir, exist_ok=True)
    
#     print(f"🔄 Processing {subset_name} set ({len(file_list)} images)...")
    
#     for idx, file_path in enumerate(tqdm(file_list)):
#         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         if img is None: continue
            
#         if img.shape != HR_SIZE:
#             img_hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
#         else:
#             img_hr = img
            
#         img_lr = degrade_image(img_hr)
        
#         filename = f"{idx:05d}.png"
#         cv2.imwrite(os.path.join(save_hr_dir, f"hr_{filename}"), img_hr)
#         cv2.imwrite(os.path.join(save_lr_dir, f"lr_{filename}"), img_lr)

# def main():
#     all_files = glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True) + \
#                 glob(os.path.join(SOURCE_DIR, "**", "*.jpg"), recursive=True)
    
#     if len(all_files) == 0:
#         print(f"❌ Error: No images found in {SOURCE_DIR}")
#         return

#     print(f"📦 Found {len(all_files)} images total.")
#     train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

#     if os.path.exists(OUTPUT_DIR):
#         print(f"⚠️ Warning: Output directory '{OUTPUT_DIR}' already exists. Merging/Overwriting...")
    
#     process_files(train_files, "train")
#     process_files(val_files, "val")
    
#     print("\n✅ Dataset preparation complete!")

# if __name__ == "__main__":
#     main()
import os
import cv2
import numpy as np
import random
import shutil
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
SOURCE_DIR = "/root/autodl-tmp/mine-qr-code/qr_dataset_kaggle"
OUTPUT_DIR = "./dataset_4"  # 建议用新目录

HR_SIZE = (256, 256)
LR_SIZE = (64, 64)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# ===========================================

def add_realistic_gaussian_blur(img):
    """
    真实感高斯模糊 (模拟对焦不准)
    Sigma 控制在 0.8 ~ 1.6 之间。
    在 64x64 的图上，Sigma=1.5 已经是很明显的失焦了，
    但不会让二维码变成一团灰雾。
    """
    ksize = random.choice([3, 5]) 
    sigma = random.uniform(0.8, 1.6) 
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def add_realistic_motion_blur(img):
    """
    真实感运动模糊 (模拟手微抖)
    拖影长度控制在 3~6 像素。
    """
    size = random.randint(3, 6) 
    kernel = np.zeros((size, size))
    # 随机角度 (不再局限于水平垂直，模拟真实手抖)
    # 这里简化处理，依然使用 xy 方向，但混合使用
    if random.random() < 0.5:
        kernel[int((size-1)/2), :] = np.ones(size) # 水平
    else:
        kernel[:, int((size-1)/2)] = np.ones(size) # 垂直
    
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

def add_iso_noise(img):
    """
    ISO 噪点 (模拟暗光拍摄)
    这种噪点是细碎的，不会破坏大结构
    """
    if random.random() < 0.7:
        row, col = img.shape
        # 适中的噪点强度
        var = random.uniform(5, 25) 
        sigma = var ** 0.5
        gauss = np.random.normal(0, sigma, (row, col))
        img = img.astype(np.float32) + gauss
        img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def add_dirt_specks(img):
    """
    微小污渍 (模拟镜头灰尘或打印纸瑕疵)
    只生成 1px 的点，不生成大块遮挡
    """
    h, w = img.shape
    img_out = img.copy()
    
    # 随机撒点盐/胡椒
    num_specks = random.randint(5, 20)
    for _ in range(num_specks):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        # 随机灰度
        color = random.randint(50, 200)
        img_out[y, x] = color
    return img_out

def add_lighting_effects(img):
    """
    光照综合处理：反光 + 阴影 + 对比度不足
    这是现实中最难处理的部分
    """
    img = img.astype(np.float32)
    h, w = img.shape

    # 1. 随机反光 (Glare) - 模拟闪光灯或顶灯
    if random.random() < 0.3:
        x, y = random.randint(0, w), random.randint(0, h)
        y_grid, x_grid = np.ogrid[:h, :w]
        sigma = random.randint(10, 25) # 光斑稍大且柔和
        mask = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        intensity = random.uniform(0.2, 0.5) 
        img = img + (255 * mask * intensity)

    # 2. 随机阴影 (Shadow)
    if random.random() < 0.3:
        # 简单的线性渐变阴影
        direction = random.choice([0, 1]) # 0=水平渐变, 1=垂直渐变
        factor = np.linspace(0.5, 1.0, w if direction==0 else h)
        if direction == 0:
            mask = np.tile(factor, (h, 1))
        else:
            mask = np.tile(factor[:, np.newaxis], (1, w))
        
        # 随机翻转阴影方向
        if random.random() < 0.5: mask = np.flip(mask)
        img = img * mask

    # 3. 整体对比度降低 (Fading) - 模拟褪色
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 0.95)
        beta = random.uniform(5, 30)
        img = img * alpha + beta

    return np.clip(img, 0, 255).astype(np.uint8)

def degrade_image(hr_img):
    """
    V4 真实感退化管线
    """
    # 1. 基础下采样 (使用最常见的双线性或立方插值)
    # INTER_AREA 在缩小时效果最好，但也最清晰，这里偶尔混入 LINEAR 模拟差算法
    interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
    lr_img = cv2.resize(hr_img, LR_SIZE, interpolation=interp)

    # === 步骤 1: 物理层 (灰尘) ===
    # 只有 40% 的图有灰尘，大部分是干净的
    if random.random() < 0.4:
        lr_img = add_dirt_specks(lr_img)

    # === 步骤 2: 光学层 (模糊) ===
    # 这是必选项，因为我们在做 SR，输入必然是不清晰的
    # 但我们混合使用 高斯(失焦) 和 运动(手抖)
    blur_type = random.random()
    if blur_type < 0.6:
        # 60% 概率是失焦 (最常见)
        lr_img = add_realistic_gaussian_blur(lr_img)
    elif blur_type < 0.9:
        # 30% 概率是手抖
        lr_img = add_realistic_motion_blur(lr_img)
    else:
        # 10% 概率是两者叠加 (极难样本)
        lr_img = add_realistic_motion_blur(add_realistic_gaussian_blur(lr_img))

    # === 步骤 3: 环境层 (光照) ===
    # 50% 的图光照有问题
    if random.random() < 0.5:
        lr_img = add_lighting_effects(lr_img)

    # === 步骤 4: 传感器层 (噪点 & 压缩) ===
    # 几乎所有低清图都有噪点
    lr_img = add_iso_noise(lr_img)
    
    # JPEG 压缩，质量适中
    if random.random() < 0.8:
        quality = random.randint(40, 90) # 最低 40，保证不做成马赛克块
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', lr_img, encode_param)
        lr_img = cv2.imdecode(encimg, 0)

    return lr_img

def process_files(file_list, subset_name):
    save_hr_dir = os.path.join(OUTPUT_DIR, subset_name, "hr")
    save_lr_dir = os.path.join(OUTPUT_DIR, subset_name, "lr")
    os.makedirs(save_hr_dir, exist_ok=True)
    os.makedirs(save_lr_dir, exist_ok=True)
    
    print(f"🔄 Processing {subset_name} set ({len(file_list)} images)...")
    
    for idx, file_path in enumerate(tqdm(file_list)):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        if img.shape != HR_SIZE:
            img_hr = cv2.resize(img, HR_SIZE, interpolation=cv2.INTER_AREA)
        else:
            img_hr = img
            
        img_lr = degrade_image(img_hr)
        
        filename = f"{idx:05d}.png"
        cv2.imwrite(os.path.join(save_hr_dir, f"hr_{filename}"), img_hr)
        cv2.imwrite(os.path.join(save_lr_dir, f"lr_{filename}"), img_lr)

def main():
    all_files = glob(os.path.join(SOURCE_DIR, "**", "*.png"), recursive=True) + \
                glob(os.path.join(SOURCE_DIR, "**", "*.jpg"), recursive=True)
    
    if len(all_files) == 0:
        print(f"❌ Error: No images found in {SOURCE_DIR}")
        return

    print(f"📦 Found {len(all_files)} images total.")
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=SEED)

    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ Warning: Output directory '{OUTPUT_DIR}' already exists. Merging/Overwriting...")
    
    process_files(train_files, "train")
    process_files(val_files, "val")
    
    print("\n✅ Dataset preparation complete!")

if __name__ == "__main__":
    main()