"""
可视化对比：Baseline vs Ours 的中段特征图
验证绿通道蒸馏是否让模型学到更好的血管特征

用法:
  python visualize_features.py \
    --img dataset_all_filtered/test/images/ANFC_000091.png \
    --baseline_weight results/experiments/all_filtered/baseline/best_model.pth \
    --ours_weight results/experiments/all_filtered/ours/best_model.pth \
    --out_dir results/feature_vis
"""
import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.transunet_official import TransUNetOfficial
from models.joint_framework import Enhancer, JointModel


def hook_fn(store, name):
    def fn(_, __, output):
        store[name] = output.detach().cpu()
    return fn


def improved_activation(feat_tensor, crop_edge=10):
    """改进版：去除边缘伪影干扰，并使用 Max 代替 Mean"""
    # 1. 使用 max 而不是 mean，能更好地捕捉对血管响应最强烈的通道
    act = feat_tensor.max(0)[0].numpy() # PyTorch 的 max 返回 (values, indices)
    act = np.maximum(act, 0)
    
    # 2. 核心修复：计算 min 和 max 时，忽略边缘的一圈像素 (比如 10 个像素)
    center_act = act[crop_edge:-crop_edge, crop_edge:-crop_edge]
    valid_min = center_act.min()
    valid_max = center_act.max()
    
    # 3. 归一化并截断异常的边缘高值
    act = (act - valid_min) / (valid_max - valid_min + 1e-8)
    act = np.clip(act, 0, 1) # 把大于 1 的边缘伪影强行切平
    
    heatmap = cv2.applyColorMap(np.uint8(255 * act), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)



def load_baseline(weight_path, device):
    model = TransUNetOfficial(n_channels=3, n_classes=1, img_size=256).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_ours(weight_path, device):
    segmentor = TransUNetOfficial(n_channels=3, n_classes=1, img_size=256)
    model = JointModel(Enhancer(), segmentor).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model


def extract_features(model, img_tensor, hook_target):
    store = {}
    handle = hook_target.register_forward_hook(hook_fn(store, "feat"))
    with torch.no_grad():
        model(img_tensor)
    handle.remove()
    return store["feat"][0]  # [C, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--baseline_weight", required=True)
    parser.add_argument("--ours_weight", required=True)
    parser.add_argument("--out_dir", default="results/feature_vis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取图片
    img_bgr = cv2.imread(args.img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 读取对应mask（同名文件，在masks目录下）
    mask_path = args.img.replace("/images/", "/masks/")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 加载模型
    baseline = load_baseline(args.baseline_weight, device)
    ours = load_ours(args.ours_weight, device)

    # 提取中段特征（ResNet body.block1 输出）
    block1_baseline = baseline.model.transformer.embeddings.hybrid_model.body.block1
    block1_ours = ours.segmentor.model.transformer.embeddings.hybrid_model.body.block1

    feat_base = extract_features(baseline, img_tensor, block1_baseline)
    feat_ours = extract_features(ours, img_tensor, block1_ours)

    # 获取 Enhancer 增强后的图像
    with torch.no_grad():
        enhanced = ours.enhancer(img_tensor)[0].cpu().permute(1, 2, 0).numpy()
    enhanced = np.clip(enhanced, 0, 1)

    # 绘图：原图 | mask | 增强图 | baseline特征 | ours特征
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(mask, cmap="gray") if mask is not None else axes[1].text(0.5, 0.5, "No Mask", ha="center")
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(enhanced)
    axes[2].set_title("Enhancer Output (Ours)")
    axes[3].imshow(improved_activation(feat_base))
    axes[3].set_title("Baseline - Block1 Feature")
    axes[4].imshow(improved_activation(feat_ours))
    axes[4].set_title("Ours - Block1 Feature")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "feature_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")

    # 额外：对比绿通道 vs 增强图
    green_ch = img_rgb[:, :, 1]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    axes2[0].imshow(img_rgb)
    axes2[0].set_title("Original")
    axes2[1].imshow(green_ch, cmap="gray")
    axes2[1].set_title("Green Channel")
    axes2[2].imshow(enhanced)
    axes2[2].set_title("Enhancer Output")
    for ax in axes2:
        ax.axis("off")
    plt.tight_layout()
    out_path2 = os.path.join(args.out_dir, "enhancement_comparison.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path2}")


if __name__ == "__main__":
    main()
