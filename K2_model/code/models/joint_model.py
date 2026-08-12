import torch
import torch.nn as nn

class JointModel(nn.Module):
    """
    端到端联合训练模型：Enhancer + Segmentor
    """
    def __init__(self, enhancer, segmentor):
        super(JointModel, self).__init__()
        self.enhancer = enhancer
        self.segmentor = segmentor

    def forward(self, x):
        # 1. 原始图像进入 Enhancer，输出增强图像
        enhanced_img = self.enhancer(x)
        
        # 2. 增强后的图像输入到下游分割网络
        # 注意：这里 enhanced_img 会同时产生针对 segmentor 的分割梯度，
        # 以及针对 Teacher 的蒸馏梯度 (MSE + Sobel)。
        seg_out = self.segmentor(enhanced_img)
        
        # 训练时我们需要 enhanced_img 来算蒸馏 Loss，推理时其实只用 seg_out
        return seg_out, enhanced_img