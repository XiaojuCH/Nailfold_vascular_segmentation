import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    轻量级/等容量版 U-Net 基线 (最高 512 通道)
    严格对齐 U-Net++ 和 TransUNet 的参数量级别，作为公平竞技的守门员。
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        # 下采样路径 (Encoder)
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        
        # 512 通道瓶颈层 (Bottleneck)，不再往下走了！
        self.bottleneck = DoubleConv(256, 512)
        
        # 上采样路径 (Decoder)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # 拼接后通道数：256 (来自up3) + 256 (来自down3的跳跃连接) = 512
        self.conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 拼接后通道数：128 + 128 = 256
        self.conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 拼接后通道数：64 + 64 = 128
        self.conv1 = DoubleConv(128, 64)
        
        # 输出层映射回 n_classes
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        c2 = self.down2(F.max_pool2d(c1, 2))
        c3 = self.down3(F.max_pool2d(c2, 2))
        
        # Bottleneck
        mid = self.bottleneck(F.max_pool2d(c3, 2))
        
        # Decoder
        x = self.conv3(torch.cat([self.up3(mid), c3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), c2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), c1], dim=1))
        
        return self.out(x)