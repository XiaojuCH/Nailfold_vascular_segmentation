"""
诊断脚本：检查 Enhancer 是否在学习
"""
import torch
from models.joint_framework import Enhancer, JointModel
from models.transunet_official import TransUNetOfficial

# 加载训练好的模型
model_path = "results/experiments/ours_transunet/best_model.pth"
enhancer = Enhancer(in_channels=3, out_channels=3)
segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=256, pretrained_path=None)
model = JointModel(enhancer, segmentor)

model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

# 测试：输入一张图，看 Enhancer 是否改变了图像
test_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    enhanced = model.enhancer(test_input)

diff = (enhanced - test_input).abs().mean().item()
print(f"Enhancer 平均改变量: {diff:.6f}")
print(f"输入范围: [{test_input.min():.3f}, {test_input.max():.3f}]")
print(f"输出范围: [{enhanced.min():.3f}, {enhanced.max():.3f}]")

if diff < 0.01:
    print("\n⚠️  警告: Enhancer 几乎没有改变图像！")
    print("可能原因:")
    print("1. 蒸馏权重太小")
    print("2. Enhancer 学习率太低")
    print("3. 官方 TransUNet 太强，Enhancer 无法提供帮助")
else:
    print(f"\n✓ Enhancer 正在工作 (改变量: {diff:.6f})")
