# Nailfold Capillary Vessel Segmentation

甲襞毛细血管分割项目，核心创新为**物理先验知识蒸馏 + 边缘感知梯度损失**，基于 TransUNet 骨干网络。

---

## 创新点

### 1. 物理先验知识蒸馏 (Physics-Prior Knowledge Distillation)

**核心思想**：用绿色通道 + CLAHE 增强生成"教师先验图"，训练一个轻量级 Enhancer 网络去模仿教师的增强效果，再将增强后的图像送入分割器。

**关键文件**：
- `models/joint_framework.py` — `Enhancer`（学生网络）+ `JointModel`（端到端框架）
- `scripts/generate_teacher.py` — 生成教师先验图（绿色通道 + CLAHE）
- `losses/joint_loss.py` — `JointDistillationLoss`，包含 MSE 蒸馏项

**原理**：甲襞血管在绿色通道对比度最高，CLAHE 进一步增强局部对比度。Enhancer 通过 MSE Loss 向教师先验对齐，使分割器获得更清晰的输入。

### 2. 边缘感知梯度损失 (Edge-Aware Gradient Loss)

**核心思想**：在增强图像和教师先验图的**绿色通道**上计算 Sobel 梯度，约束 Enhancer 保留血管边缘细节。

**关键文件**：
- `losses/joint_loss.py` — `GradientLoss`（Sobel 算子梯度匹配）

**原理**：MSE Loss 只约束像素值，梯度 Loss 额外约束边缘结构，防止 Enhancer 过度平滑血管边界。

### 联合损失函数

```
L_total = L_seg(BCE + Dice) + λ_mse * L_mse + λ_grad * L_grad
```

默认参数：`λ_mse=10.0`，`λ_grad=30.0`

---

## 项目结构

```
├── train_unified.py          # 主训练脚本（baseline / ours 模式）
├── train_baselines.py        # 对比实验脚本（UNet / UNet++）
├── models/
│   ├── joint_framework.py    # Enhancer + JointModel
│   ├── transunet_official.py # 官方 TransUNet 实现
│   ├── unet_baseline.py      # UNet 基线
│   └── unet_plus_plus.py     # UNet++ 基线
├── losses/
│   └── joint_loss.py         # GradientLoss + JointDistillationLoss
├── datasets/
│   └── dataset_vessel.py     # 数据加载（支持 teacher_priors）
├── utils/
│   ├── metrics.py            # Dice / IoU / HD95 等评估指标
│   └── model_stats.py        # 参数量统计
├── scripts/
│   ├── generate_teacher.py   # 生成教师先验图
│   ├── split_all_dataset.py  # 按患者ID分割数据集
│   └── visualize_*.py        # 可视化脚本
└── dataset_tools/
    └── 连通域分析.py           # 连通域筛选脏数据
```

---

## 数据集

| 选项 | 路径 | 说明 |
|------|------|------|
| `anfc256` | `dataset_anfc256_split/` | 纯 ANFC 数据集（68 患者） |
| `all` | `dataset_all_split/` | ANFC + JiaBi 混合（119 患者） |
| `all_filtered` | `dataset_all_filtered/` | `all` 经连通域筛选后（剔除块状斑块） |

数据集按**患者级别**分割（train 81 / val 19 / test 19 患者），避免数据泄露。

---

## 训练命令

### 对比基线

```bash
# UNet
D:/anaconda3/envs/pytorch/python.exe train_baselines.py --model unet --dataset all_filtered --epochs 50 --batch_size 4

# UNet++
python train_baselines.py --model unet++ --dataset all_filtered --epochs 50 --batch_size 4
```

### TransUNet Baseline

```bash
python train_unified.py --mode baseline --dataset all_filtered --epochs 50 --batch_size 4
```

### Ours（联合蒸馏）

```bash
python train_unified.py --mode ours --dataset all_filtered --lambda_mse 10.0 --lambda_grad 30.0 --epochs 50 --batch_size 4
```

### 使用筛选后数据集

```bash
D:/anaconda3/envs/pytorch/python.exe train_unified.py --mode ours --dataset all_filtered --lambda_mse 10.0 --lambda_grad 30.0 --epochs 50 --batch_size 4
```

### 生成教师先验（训练 ours 前需要）

```bash
python scripts/generate_teacher.py \
  --input_dir ./dataset_anfc256_split/train/images \
  --output_dir ./dataset_anfc256_split/train/teacher_priors
```

---

## 评估指标

Dice、IoU、HD95、Sensitivity、Specificity、Accuracy、Precision（见 `utils/metrics.py`）

---

## 依赖

```bash
pip install torch torchvision opencv-python scikit-image tqdm matplotlib
```

TransUNet 官方实现位于 `third_party/TransUNet/`，需按其 README 配置预训练权重。


D:\anaconda3\envs\pytorch\python.exe  visualize_features.py --img dataset_all_filtered/test/images/ANFC_000497.png --baseline_weight results/experiments/all_filtered/baseline/best_model.pth --ours_weight results\experiments\all_filtered\ours_green_only\best_model.pth --out_dir results/feature_vis