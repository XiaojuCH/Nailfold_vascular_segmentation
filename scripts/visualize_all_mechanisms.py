import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_vessel import VesselDataset
from models.unet_baseline import UNet
from models.unet_plus_plus import UNetPlusPlus
from models.transunet import TransUNet
from models.joint_framework import Enhancer, JointModel

def get_activation_heatmap(activation_tensor, original_img):
    """将高维特征图转换为热力图并叠加到原图上"""
    if activation_tensor is None:
        return original_img # 如果没抓到特征，就返回原图防报错
        
    heatmap = torch.mean(activation_tensor, dim=1).squeeze(0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = np.uint8(255 * heatmap)
    
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original_img_bgr = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap_color * 0.4 + original_img_bgr * 0.6
    return cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "./dataset_raw_split"
    
    # 所有模型的权重路径
    CKPT_UNET = "./experiments/baselines/unet/best_model.pth"
    CKPT_UNET_PP = "./experiments/baselines/unet++/best_model.pth"
    CKPT_TRANSUNET = "./experiments/baselines/transunet/best_model.pth"
    CKPT_OURS = "./experiments/ours_transunet/best_model.pth"
    
    SAVE_DIR = "./vis_results_all_mechanisms"
    os.makedirs(SAVE_DIR, exist_ok=True)

    test_dataset = VesselDataset(
        image_dir=os.path.join(DATA_DIR, "test/images"),
        mask_dir=os.path.join(DATA_DIR, "test/masks"),
        teacher_dir=os.path.join(DATA_DIR, "test/teacher_priors"), # 正确指向测试集 Teacher
        img_size=256, augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("[*] 正在加载所有模型的权重并挂载 Hook...")
    
    # 实例化模型
    model_unet = UNet(n_channels=3, n_classes=1).to(DEVICE)
    model_unet_pp = UNetPlusPlus(n_channels=3, n_classes=1).to(DEVICE)
    model_transunet = TransUNet(n_channels=3, n_classes=1, img_size=256).to(DEVICE)
    model_ours = JointModel(Enhancer(3, 3), TransUNet(n_channels=3, n_classes=1, img_size=256)).to(DEVICE)

    # 加载权重
    model_unet.load_state_dict(torch.load(CKPT_UNET, map_location=DEVICE, weights_only=True))
    model_unet_pp.load_state_dict(torch.load(CKPT_UNET_PP, map_location=DEVICE, weights_only=True))
    model_transunet.load_state_dict(torch.load(CKPT_TRANSUNET, map_location=DEVICE, weights_only=True))
    model_ours.load_state_dict(torch.load(CKPT_OURS, map_location=DEVICE, weights_only=True))

    for m in [model_unet, model_unet_pp, model_transunet, model_ours]:
        m.eval()

    # ================= 注册 Hook 提取各模型的深层特征 =================
    activations = {'unet': None, 'unet_pp': None, 'transunet': None, 'ours': None}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # 挂载 U-Net 瓶颈层
    model_unet.bottleneck.register_forward_hook(get_activation('unet'))
    
    # 挂载 TransUNet 瓶颈层前 (enc4)
    model_transunet.enc4.register_forward_hook(get_activation('transunet'))
    model_ours.segmentor.enc4.register_forward_hook(get_activation('ours'))
    
    # 挂载 U-Net++ (由于不确定你具体代码的变量名，尝试挂载倒数第二个模块，或直接略过报错)
    try:
        # 尝试寻找包含通道最多的深层卷积
        last_conv = [m for m in model_unet_pp.modules() if isinstance(m, torch.nn.Conv2d)][-2]
        last_conv.register_forward_hook(get_activation('unet_pp'))
    except Exception as e:
        print(f"[!] U-Net++ 挂载 Hook 失败，热力图将用原图代替。错误: {e}")

    # ================= 开始推理与绘图 =================
    print(f"[*] 开始生成全景机制大图...")
    num_to_plot = 15 # 防止跑太多
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, total=num_to_plot)):
            if i >= num_to_plot: break
                
            image = batch["image"].to(DEVICE)
            mask_gt = batch["mask"].to(DEVICE)
            teacher = batch["teacher"].to(DEVICE) if "teacher" in batch else image

            # 推理激活 Hook
            _ = model_unet(image)
            _ = model_unet_pp(image)
            _ = model_transunet(image)
            _, enhanced_img = model_ours(image)
            
            # 数据转换 (NumPy)
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            gt_np = mask_gt[0, 0].cpu().numpy()
            teacher_np = teacher[0].cpu().numpy().transpose(1, 2, 0)
            enh_np = enhanced_img[0].cpu().numpy().transpose(1, 2, 0)
            
            # 生成热力图
            hm_unet = get_activation_heatmap(activations['unet'], img_np)
            hm_unet_pp = get_activation_heatmap(activations['unet_pp'], img_np)
            hm_transunet = get_activation_heatmap(activations['transunet'], img_np)
            hm_ours = get_activation_heatmap(activations['ours'], img_np)

            # --- 绘图排版 (2行4列) ---
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # 第一行：参考组
            axes[0, 0].imshow(img_np); axes[0, 0].set_title("1. Original", fontsize=14)
            axes[0, 1].imshow(gt_np, cmap='gray'); axes[0, 1].set_title("2. Ground Truth", fontsize=14)
            axes[0, 2].imshow(teacher_np); axes[0, 2].set_title("3. Teacher (Prior)", fontsize=14)
            axes[0, 3].imshow(enh_np); axes[0, 3].set_title("4. Enhancer (Student)", fontsize=14)
            
            # 第二行：热力图对比
            axes[1, 0].imshow(hm_unet); axes[1, 0].set_title("5. U-Net Attention", fontsize=14)
            axes[1, 1].imshow(hm_unet_pp); axes[1, 1].set_title("6. U-Net++ Attention", fontsize=14)
            axes[1, 2].imshow(hm_transunet); axes[1, 2].set_title("7. TransUNet Attention", fontsize=14)
            axes[1, 3].imshow(hm_ours); axes[1, 3].set_title("8. OURS Attention", fontsize=14, color='red', fontweight='bold')
            
            for ax in axes.flatten():
                ax.axis("off")
            
            plt.subplots_adjust(wspace=0.05, hspace=0.1)
            plt.savefig(os.path.join(SAVE_DIR, f"all_mechanisms_{i+1:03d}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"[*] 🎉 完美！请前往 {SAVE_DIR} 挑选放入 PPT 的极致对比图！")

if __name__ == "__main__":
    main()