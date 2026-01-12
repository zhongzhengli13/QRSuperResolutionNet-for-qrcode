import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        
        # 128 -> 64 -> 32 -> 16 -> 8
        for i, out_filters in enumerate([base_channels, base_channels*2, base_channels*4, base_channels*8]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i==0)))
            in_filters = out_filters

        self.features = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1) # 输出一个分数：真(>0) 或 假(<0)
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)