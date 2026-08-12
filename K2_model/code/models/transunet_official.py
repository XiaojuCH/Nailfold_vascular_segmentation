"""
官方 TransUNet 适配器
将官方实现适配到你的项目中
"""
import sys
import os
import numpy as np

# 添加 TransUNet 到路径（append 而非 insert(0)：避免遮蔽本项目的 utils 包，
# 否则 `from utils.metrics import ...` 会命中 TransUNet 官方 utils.py 并触发 medpy 依赖）
transunet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'TransUNet')
sys.path.append(transunet_path)

import torch
import torch.nn as nn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

class TransUNetOfficial(nn.Module):
    """
    官方 TransUNet 的封装类
    """
    def __init__(self, n_channels=3, n_classes=1, img_size=256, vit_name='R50-ViT-B_16',
                 vit_patches_size=16, pretrained_path=None):
        super().__init__()

        # 获取配置
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = n_classes
        config_vit.n_skip = 3

        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        # 创建模型
        self.model = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
        self.decoder_out_channels = config_vit["decoder_channels"][-1]

        # 加载预训练权重
        if pretrained_path is not None and os.path.exists(pretrained_path):
            self.model.load_from(weights=np.load(pretrained_path))
            print(f"[*] 已加载预训练权重: {pretrained_path}")
        else:
            print("[*] 未加载预训练权重，从头训练")

    def forward(self, x, return_decoder_features=False, return_decoder_output=False):
        return self.model(
            x,
            return_decoder_features=return_decoder_features,
            return_decoder_output=return_decoder_output,
        )

    def segment_from_decoder_output(self, decoder_output):
        return self.model.segmentation_head(decoder_output)
