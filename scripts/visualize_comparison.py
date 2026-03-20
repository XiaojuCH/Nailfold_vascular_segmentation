import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入数据集
from datasets.dataset_vessel import VesselDataset

# 导入所有基线模型和我们的创新框架
from models.unet_baseline import UNet
from models.unet_plus_plus import UNetPlusPlus
from models.transunet import TransUNet
from models.joint_framework import Enhancer, JointModel

def overlay_mask(image, mask, color=(1, 0, 0), alpha=0.5):
    """
    将二值 Mask 以半透明颜色叠加到 RGB 原图上
    :param image: 原图, shape (H, W, 3), 范围 [0, 1]
    :param mask: 二值掩码, shape (H, W), 范围 {0, 1}
    :param color: 叠加颜色, RGB 格式, 例如红色为 (1, 0, 0), 绿色为 (0, 1, 0)
    :param alpha: 透明度 (0~1)
    :return: 叠加后的 RGB 图像
    """
    overlay = image.copy()
    for c in range(3):
        # np.where: 如果 mask 为 1，则混合颜色；否则保持原图颜色
        overlay[:, :, c] = np.where(
            mask > 0, 
            image[:, :, c] * (1 - alpha) + color[c] * alpha, 
            image[:, :, c]
        )
    # 限制范围在 [0, 1] 防止 matplotlib 报错
    return np.clip(overlay, 0, 1)

def main():
    # ================= 1. 路径与配置设置 =================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "./dataset_raw_split"
    
    # 权重路径配置 (确保这四个文件都真实存在)
    CKPT_UNET = "./experiments/baselines/unet/best_model.pth"
    CKPT_UNET_PP = "./experiments/baselines/unet++/best_model.pth"
    CKPT_TRANSUNET = "./experiments/baselines/transunet/best_model.pth"
    CKPT_OURS = "./experiments/ours_transunet/best_model.pth"
    
    # 保存叠加对比图的新文件夹
    SAVE_DIR = "./vis_results_overlay"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ================= 2. 加载测试集 =================
    test_dataset = VesselDataset(
        image_dir=os.path.join(DATA_DIR, "test/images"),
        mask_dir=os.path.join(DATA_DIR, "test/masks"),
        teacher_dir=None, 
        img_size=256, 
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ================= 3. 实例化并加载 4 个模型 =================
    print("[*] 正在加载所有模型的权重...")
    
    def load_weights(model, ckpt_path, name):
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            model.eval()
            print(f" [+] 成功加载: {name}")
        else:
            print(f" [!] 警告: 找不到权重文件 -> {ckpt_path}")
        return model

    model_unet = load_weights(UNet(n_channels=3, n_classes=1).to(DEVICE), CKPT_UNET, "U-Net")
    model_unet_pp = load_weights(UNetPlusPlus(n_channels=3, n_classes=1).to(DEVICE), CKPT_UNET_PP, "U-Net++")
    model_transunet = load_weights(TransUNet(n_channels=3, n_classes=1, img_size=256).to(DEVICE), CKPT_TRANSUNET, "TransUNet")
    
    enhancer = Enhancer(in_channels=3, out_channels=3)
    segmentor = TransUNet(n_channels=3, n_classes=1, img_size=256)
    model_ours = load_weights(JointModel(enhancer, segmentor).to(DEVICE), CKPT_OURS, "OURS")

    # ================= 4. 开始推理与绘图 =================
    print(f"[*] 开始生成 SCI 叠加对比大图，将保存至 {SAVE_DIR} ...")
    num_to_plot = 50  # 选前50张画图

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=num_to_plot)):
            if i >= num_to_plot: break
                
            image = batch["image"].to(DEVICE)
            mask_gt = batch["mask"].to(DEVICE)

            # --- 4 模型分别推理 ---
            pred_unet = torch.sigmoid(model_unet(image)) > 0.5
            pred_unet_pp = torch.sigmoid(model_unet_pp(image)) > 0.5
            pred_transunet = torch.sigmoid(model_transunet(image)) > 0.5
            out_ours, _ = model_ours(image)
            pred_ours = torch.sigmoid(out_ours) > 0.5

            # --- 数据提取与转换 (转到 CPU numpy, 并确保尺寸正确) ---
            # image 原本是 [1, 3, 256, 256]，转为 [256, 256, 3] 以供 matplotlib 画图
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            
            # Mask 转为二维数组 [256, 256]
            gt_np = mask_gt[0, 0].cpu().numpy() > 0.5
            unet_np = pred_unet[0, 0].cpu().numpy()
            unet_pp_np = pred_unet_pp[0, 0].cpu().numpy()
            transunet_np = pred_transunet[0, 0].cpu().numpy()
            ours_np = pred_ours[0, 0].cpu().numpy()

            # --- 生成色彩叠加图 ---
            # 金标准用绿色 (0, 1, 0)
            img_gt = overlay_mask(img_np, gt_np, color=(0, 1, 0), alpha=0.4)
            # 预测结果统一用红色 (1, 0, 0)
            img_unet = overlay_mask(img_np, unet_np, color=(1, 0, 0), alpha=0.4)
            img_unet_pp = overlay_mask(img_np, unet_pp_np, color=(1, 0, 0), alpha=0.4)
            img_transunet = overlay_mask(img_np, transunet_np, color=(1, 0, 0), alpha=0.4)
            img_ours = overlay_mask(img_np, ours_np, color=(1, 0, 0), alpha=0.4)

            # --- 绘图排版 (1行6列大图) ---
            fig, axes = plt.subplots(1, 6, figsize=(24, 4)) 
            
            titles = ["Original", "Ground Truth (Green)", "U-Net", "U-Net++", "TransUNet", "OURS (Proposed)"]
            images_to_show = [img_np, img_gt, img_unet, img_unet_pp, img_transunet, img_ours]
            
            for ax, img, title in zip(axes, images_to_show, titles):
                ax.imshow(img)
                if title == "OURS (Proposed)":
                    ax.set_title(title, fontsize=16, fontweight='bold', color="red")
                else:
                    ax.set_title(title, fontsize=16)
                ax.axis("off") # 关闭坐标轴
            
            plt.subplots_adjust(wspace=0.05, hspace=0)
            
            save_path = os.path.join(SAVE_DIR, f"overlay_compare_{i+1:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()

    print(f"[*] 🎉 4模型色彩叠加图绘制完成！请前往 {SAVE_DIR} 文件夹查看！")

if __name__ == "__main__":
    main()