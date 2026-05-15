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


class JointModel_V2(nn.Module):
    """
    升级版联合框架：融入绿通道先验的空间注意力机制 (Spatial Attention Gate)
    """
    def __init__(self, enhancer, segmentor):
        super(JointModel_V2, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor

    def forward(self, x):
        # 1. 物理先验增强分支 (提取类似绿通道的血管高亮特征)
        enhanced_feat = self.enhancer(x)
        
        # 2. 生成空间注意力掩膜 (Spatial Attention Mask)
        # 因为 enhanced_feat 是 3 通道，我们按通道求平均变成单通道的概率图，
        # 然后经过 sigmoid 确保值域在 (0, 1) 之间。
        # 越接近 1 代表越可能是血管，越接近 0 代表是背景皮肤。
        attention_mask = torch.sigmoid(enhanced_feat.mean(dim=1, keepdim=True))
        
        # 3. 物理先验融合 (残差注意力叠加)
        # X_fused = X + X * Mask
        # 这样既过滤了背景噪声（X * Mask），又保留了原图的基础信息结构（+ X）
        x_fused = x + x * attention_mask
        
        # 4. 送入下游的 TransUNet 进行特征提取和分割
        seg_out = self.segmentor(x_fused)
        
        # 返回 seg_out 用于计算 Dice Loss
        # 返回 enhanced_feat 用于计算 MSE 和 Gradient Loss (向绿通道蒸馏)
        # 额外返回 attention_mask 方便以后我们做可视化证明
        return seg_out, enhanced_feat, attention_mask