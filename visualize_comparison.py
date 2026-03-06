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

def main():
    # ================= 1. 路径与配置设置 =================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "./dataset_raw_split"
    
    # 权重路径配置 (确保这四个文件都真实存在！)
    CKPT_UNET = "./experiments/baselines/unet/best_model.pth"
    CKPT_UNET_PP = "./experiments/baselines/unet++/best_model.pth"
    CKPT_TRANSUNET = "./experiments/baselines/transunet/best_model.pth"
    CKPT_OURS = "./experiments/ours_transunet/best_model.pth"
    
    SAVE_DIR = "./vis_results_sci_all_models"
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

    # 1. U-Net
    model_unet = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model_unet = load_weights(model_unet, CKPT_UNET, "U-Net")

    # 2. U-Net++
    model_unet_pp = UNetPlusPlus(n_channels=3, n_classes=1).to(DEVICE)
    model_unet_pp = load_weights(model_unet_pp, CKPT_UNET_PP, "U-Net++")

    # 3. TransUNet
    model_transunet = TransUNet(n_channels=3, n_classes=1, img_size=256).to(DEVICE)
    model_transunet = load_weights(model_transunet, CKPT_TRANSUNET, "TransUNet")

    # 4. OURS (Enhancer + TransUNet)
    enhancer = Enhancer(in_channels=3, out_channels=3)
    segmentor = TransUNet(n_channels=3, n_classes=1, img_size=256)
    model_ours = JointModel(enhancer, segmentor).to(DEVICE)
    model_ours = load_weights(model_ours, CKPT_OURS, "OURS (Joint Distillation)")

    # ================= 4. 开始推理与绘图 =================
    print(f"[*] 开始生成 SCI 4模型对比大图，将保存至 {SAVE_DIR} ...")
    num_to_plot = 20  # 选前20张画图

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=num_to_plot)):
            if i >= num_to_plot: break
                
            image = batch["image"].to(DEVICE)
            mask_gt = batch["mask"].to(DEVICE)

            # --- 4 模型分别推理 ---
            pred_unet = torch.sigmoid(model_unet(image)) > 0.5
            pred_unet_pp = torch.sigmoid(model_unet_pp(image)) > 0.5
            pred_transunet = torch.sigmoid(model_transunet(image)) > 0.5
            
            # OURS 推理 (需解包)
            out_ours, _ = model_ours(image)
            pred_ours = torch.sigmoid(out_ours) > 0.5

            # --- 数据提取与转换 ---
            img_show = image[0].cpu().numpy().transpose(1, 2, 0)
            gt_show = mask_gt[0, 0].cpu().numpy()
            
            show_unet = pred_unet[0, 0].cpu().numpy()
            show_unet_pp = pred_unet_pp[0, 0].cpu().numpy()
            show_transunet = pred_transunet[0, 0].cpu().numpy()
            show_ours = pred_ours[0, 0].cpu().numpy()

            # --- 绘图排版 (1行6列大图) ---
            # figsize设得宽一点，保证论文里每张图的正方形比例
            fig, axes = plt.subplots(1, 6, figsize=(24, 4)) 
            
            # 统一配置参数
            titles = ["Original", "Ground Truth", "U-Net", "U-Net++", "TransUNet", "OURS (Proposed)"]
            images_to_show = [img_show, gt_show, show_unet, show_unet_pp, show_transunet, show_ours]
            
            for ax, img, title in zip(axes, images_to_show, titles):
                if title == "Original":
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap="gray")
                
                # 突出显示我们自己的模型名称
                if title == "OURS (Proposed)":
                    ax.set_title(title, fontsize=16, fontweight='bold', color="red")
                else:
                    ax.set_title(title, fontsize=16)
                    
                ax.axis("off") # 关闭坐标轴，让画面更干净
            
            # 缩减子图之间的空白间距
            plt.subplots_adjust(wspace=0.05, hspace=0)
            
            # 保存高质量图片
            save_path = os.path.join(SAVE_DIR, f"compare_4models_{i+1:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()

    print(f"[*] 🎉 4模型全景图绘制完成！请前往 {SAVE_DIR} 文件夹查看！")

if __name__ == "__main__":
    main()