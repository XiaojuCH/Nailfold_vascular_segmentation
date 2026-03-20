import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_vessel import VesselDataset
from models.joint_framework import Enhancer, JointModel
from models.transunet import TransUNet

def get_activation_heatmap(activation_tensor, original_img):
    """将高维特征图转换为热力图并叠加到原图上"""
    # 取通道平均值作为激活强度
    heatmap = torch.mean(activation_tensor, dim=1).squeeze(0).cpu().numpy()
    # 归一化到 0-255
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = np.uint8(255 * heatmap)
    
    # Resize 回原图尺寸并应用伪彩色
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加到原图
    original_img_bgr = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap_color * 0.4 + original_img_bgr * 0.6
    return cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "./dataset_raw_split"
    OURS_CKPT = "./experiments/ours_transunet/best_model.pth"
    SAVE_DIR = "./vis_results_mechanism"
    os.makedirs(SAVE_DIR, exist_ok=True)

    test_dataset = VesselDataset(
        image_dir=os.path.join(DATA_DIR, "test/images"),
        mask_dir=os.path.join(DATA_DIR, "test/masks"),
        teacher_dir=os.path.join(DATA_DIR, "test/teacher_priors"),
        img_size=256, augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 实例化并加载模型
    enhancer = Enhancer(in_channels=3, out_channels=3)
    segmentor = TransUNet(n_channels=3, n_classes=1, img_size=256)
    model = JointModel(enhancer, segmentor).to(DEVICE)
    model.load_state_dict(torch.load(OURS_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    # 注册 Hook 提取特征
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # 挂载在 TransUNet 的底层特征 (例如 enc4 之后)
    model.segmentor.enc4.register_forward_hook(get_activation('bottleneck'))

    print(f"[*] 开始生成机制可视化大图...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=10)): # 取前10张
            if i >= 10:     # <--- 加上这一行：一旦到了第 10 张
                break       # <--- 加上这一行：立刻强行跳出循环
            image = batch["image"].to(DEVICE)
            teacher = batch["teacher"].to(DEVICE) if "teacher" in batch else image

            out_ours, enhanced_img = model(image)
            
            # 数据转换
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            teacher_np = teacher[0].cpu().numpy().transpose(1, 2, 0)
            enh_np = enhanced_img[0].cpu().numpy().transpose(1, 2, 0)
            
            # 生成热力图
            heatmap_vis = get_activation_heatmap(activation['bottleneck'], img_np)

            # 绘图排版 (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("1. Original Input", fontsize=14)
            axes[0, 0].axis("off")
            
            axes[0, 1].imshow(heatmap_vis)
            axes[0, 1].set_title("2. Network Attention Heatmap", fontsize=14)
            axes[0, 1].axis("off")

            axes[1, 0].imshow(teacher_np)
            axes[1, 0].set_title("3. Physical Prior (Teacher)", fontsize=14)
            axes[1, 0].axis("off")
            
            axes[1, 1].imshow(enh_np)
            axes[1, 1].set_title("4. Learned Enhancer (Student)", fontsize=14)
            axes[1, 1].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"mechanism_sample_{i+1:03d}.png"), dpi=300)
            plt.close()

if __name__ == "__main__":
    main()