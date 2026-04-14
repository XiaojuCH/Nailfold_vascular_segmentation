import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# 路径
mask_dir = Path(r"C:\Workfolder\NailFold\nailData\orgin_data\masks")
save_dir = Path(r"C:\Workfolder\NailFold\nailData\orgin_data\样本分析")
save_dir.mkdir(exist_ok=True)

ratios = []
all_areas = []
acc = np.zeros((256, 256), dtype=np.float32)
img_count = 0

# 统计
for p in mask_dir.glob("*.png"):
    mask = cv2.imread(str(p), 0)
    mask_bin = (mask > 0).astype(np.uint8)

    # 1. 前景像素比例
    fg = np.count_nonzero(mask_bin)
    ratios.append(fg / mask_bin.size)

    # 2. 空间分布
    acc += mask_bin
    img_count += 1

    # 3. 连通区域面积
    num, labels = cv2.connectedComponents(mask_bin)
    for i in range(1, num):
        area = np.sum(labels == i)
        all_areas.append(area)

ratios = np.array(ratios)

# 计算均值和标准差
mu, sigma = ratios.mean(), ratios.std()

# 绘制直方图 + 正态拟合曲线
plt.figure(figsize=(6,4))
count, bins, ignored = plt.hist(ratios, bins=30, density=True, alpha=0.6, color='g')
x = np.linspace(ratios.min(), ratios.max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2)
plt.xlabel("Foreground pixel ratio")
plt.ylabel("Density")
plt.title(f"Foreground ratio distribution with normal fit\nμ={mu:.4f}, σ={sigma:.4f}")
plt.tight_layout()
plt.savefig(save_dir / "foreground_ratio_normal_fit.png", dpi=300)
plt.show()

# 正态性检验
stat, p = shapiro(ratios)
with open(save_dir / "statistics.txt", "a", encoding="utf-8") as f:
    f.write("\nNormality test (Shapiro-Wilk)\n")
    f.write(f"Statistic: {stat:.6f}, p-value: {p:.6f}\n")
    if p > 0.05:
        f.write("Sample can be considered approximately normal.\n")
    else:
        f.write("Sample is not normally distributed.\n")

# 前景比例分布
plt.figure()
plt.hist(ratios, bins=30)
plt.xlabel("Foreground pixel ratio")
plt.ylabel("Image count")
plt.title("Foreground ratio distribution")
plt.tight_layout()
plt.savefig(save_dir / "foreground_ratio_hist.png", dpi=300)
plt.show()

# 保存数值结果
with open(save_dir / "statistics.txt", "w", encoding="utf-8") as f:
    f.write(f"Foreground ratio statistics\n")
    f.write(f"Mean: {ratios.mean():.6f}\n")
    f.write(f"Min : {ratios.min():.6f}\n")
    f.write(f"Max : {ratios.max():.6f}\n")

print(f"前景比例: mean={ratios.mean():.4f}, "
      f"min={ratios.min():.4f}, max={ratios.max():.4f}")

# 血管空间概率图
prob_map = acc / img_count
plt.figure()
plt.imshow(prob_map, cmap="hot")
plt.colorbar(label="Vessel probability")
plt.title("Spatial distribution of vessels")
plt.tight_layout()
plt.savefig(save_dir / "spatial_distribution.png", dpi=300)
plt.show()

# 连通区域面积分布
plt.figure()
plt.hist(all_areas, bins=50, log=True)
plt.xlabel("Connected component area (pixels)")
plt.ylabel("Count (log)")
plt.title("Vessel area distribution")
plt.tight_layout()
plt.savefig(save_dir / "vessel_area_distribution.png", dpi=300)
plt.show()

print(f"统计结果已保存到: {save_dir.resolve()}")
