import torch
import torch.nn as nn

class Enhancer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Enhancer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 最后一层不用 Sigmoid，因为我们要输出残差（可以为负数）
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        
        # 将原始输入加上残差，并截断到 [0, 1] 保证是合法图像
        out = torch.clamp(x + residual, min=0.0, max=1.0)
        return out