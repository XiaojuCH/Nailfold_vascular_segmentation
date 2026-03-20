"""
测试官方 TransUNet 是否能正常加载
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试导入
try:
    from models.transunet_official import TransUNetOfficial
    print("[OK] 成功导入 TransUNetOfficial")

    # 测试创建模型（不加载预训练）
    print("\n测试1: 创建模型（无预训练）...")
    model = TransUNetOfficial(
        n_channels=3,
        n_classes=1,
        img_size=256,
        pretrained_path=None
    )
    print("[OK] 模型创建成功")

    # 测试前向传播
    print("\n测试2: 前向传播...")
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"[OK] 输入: {x.shape}, 输出: {out.shape}")

    # 测试加载预训练权重
    print("\n测试3: 加载预训练权重...")
    model_pretrained = TransUNetOfficial(
        n_channels=3,
        n_classes=1,
        img_size=256,
        pretrained_path='model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    )
    print("[OK] 预训练权重加载成功")

    print("\n[SUCCESS] 所有测试通过！可以开始训练了！")

except Exception as e:
    print(f"[ERROR] 错误: {e}")
    import traceback
    traceback.print_exc()
