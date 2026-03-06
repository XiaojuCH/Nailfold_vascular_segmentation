import torch
import torch.nn as nn

class Enhancer(nn.Module):
    """
    轻量级图像增强网络 (Student) - 带残差连接
    """
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
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        # 残差连接：输入加上变化量，并截断到合法图像范围
        out = torch.clamp(x + residual, min=0.0, max=1.0)
        return out

class JointModel(nn.Module):
    """
    端到端联合训练框架：Enhancer + Segmentor
    """
    def __init__(self, enhancer, segmentor):
        super(JointModel, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor

    def forward(self, x):
        # 1. 原始图像通过增强器，生成高对比度特征图
        enhanced_img = self.enhancer(x)
        
        # 2. 增强后的图输入给下游分割器
        seg_out = self.segmentor(enhanced_img)
        
        # 训练时我们需要 enhanced_img 来和 Teacher 算 Loss，推理时其实只用 seg_out
        return seg_out, enhanced_img