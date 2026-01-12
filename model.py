# @Author : LiZhongzheng
# 开发时间  ：2025-12-27 (Optimized)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.se = SEBlock(channels) # 将 SE 集成到 Block 内部

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * self.se(x5) # 加入 SE 注意力

class RRDB(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out

class QRSuperResolutionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_blocks=16): # 增加默认 block 数到 16
        super().__init__()
        
        # 浅层特征提取
        self.entry = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        # 深层特征提取 (RRDB 主体)
        # 复杂的图片需要更深的网络，建议 num_blocks >= 16
        self.body = nn.Sequential(*[RRDB(base_channels) for _ in range(num_blocks)])
        
        # 也是在 body 后接一个卷积，平滑特征
        self.body_conv = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

        # 上采样部分 (4x)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 输出层
        self.exit = nn.Sequential(
            nn.Conv2d(base_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        # 主分支
        feat_entry = self.entry(x)
        feat_body = self.body(feat_entry)
        feat_body = self.body_conv(feat_body)
        feat = feat_entry + feat_body # 全局残差连接，非常重要！
        
        out = self.upsample(feat)
        out = self.exit(out)
        
        # 直接输出，让 Loss 去处理范围约束，或者在这里使用 Tanh/Sigmoid
        # 对于二维码，可以不强制 clamp，交给 loss 去逼近 0 和 1
        return out 

if __name__ == "__main__":
    model = QRSuperResolutionNet()
    # 打印参数量，看看模型够不够大
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    dummy = torch.randn(1, 1, 64, 64)
    print(model(dummy).shape)