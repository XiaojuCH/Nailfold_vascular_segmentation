# 官方 DeepLabV3+（MMSegmentation）审计与复现计划

## 结论先行

师妹提供的 `TYT_Code/mmsegmentation-main` 是 OpenMMLab MMSegmentation 1.2.2 的源码，MobileNetV2 + `DepthwiseSeparableASPPHead` 确实是官方 DeepLabV3+ 实现；因此值得作为强 baseline 复现。

但她提供的 `my_deeplabv3plus_config.py` / `evaluate_deeplabv3plus_to_csv.py` 不能原样用于本项目，更不能直接把其 HD95 15.x 与我们现有数值并排比较。必须先对齐数据、标签与指标。

## 已发现的关键差异

| 项目 | 师妹代码现状 | 对当前项目的影响 | 处理 |
|---|---|---|---|
| 框架 | MMSegmentation 1.2.2，官方 DeepLabV3+ | 架构来源可靠 | 保留官方实现 |
| 配置路径 | `_base_`、`data_root` 都是 `C:\Users\33101\...` 绝对路径 | 当前机器无法直接运行 | 新建可移植配置 |
| 数据划分 | 她的路径为 `UNet_test/train_data`，文件列表未提供 | 不确定是否与我们的 patient split 相同 | 固定 `dataset_all_filtered/train/val/test` |
| 标签编码 | 当前 mask 有 0--255 灰度边缘；MMSeg 二分类必须是 0/1 | 原样训练会将大量灰度值视作非法类别，255 还会与 ignore index 冲突 | 按现有规则 `mask > 127` 生成非破坏性 0/1 视图 |
| 训练测试集 | 原配置只定义 train/val；`test_dataloader` 未显式定义 | 没有标准 test loop | 补全固定 test dataloader |
| checkpoint | 以 MMSeg `mDice` 选最佳 | MMSeg mDice 是类别层面的集合指标，不等于本项目的逐图前景 Dice | 保留作 val 选权重，再用统一逐图指标评估 |
| 自定义评估 | 引用未提供的 `metrics.py`，且 HD95 实现未知 | HD95 15.x 无法复核 | 使用本项目 surface HD95、clDice、Boundary F1 |
| 测试代码 | CPU 手工推理、手工归一化，mask 以 `>0` 二值化 | 评估和本项目 `>127` 口径不同 | 新增统一 MMSeg 评估脚本 |

## 可复现实验

已新增：

- `dataset_tools/prepare_mmseg_binary_dataset.py`：构建 `dataset_all_filtered_mmseg`；图像使用硬链接，不改原数据；mask 严格转为 0/1。
- `configs/deeplabv3plus_mobilenetv2_official_all_filtered.py`：固定当前 split、256、seed42、从头训练、MobileNetV2 + ASPP、CE + Dice、10k iterations。
- `evaluate_mmseg_deeplabv3plus.py`：将官方 checkpoint 用现有 Dice/IoU/Recall/Precision/Specificity/Accuracy/surface-HD95/clDice/Boundary F1 统一评估。
- `scripts/run_official_deeplabv3plus_20260730.ps1`：一键准备、训练与评估。

后续若决定正式复现，先建立隔离环境：

```powershell
.\scripts\setup_official_mmseg_env.ps1
```

再执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_official_deeplabv3plus_20260730.ps1 -Stage smoke
```

确认 smoke 后正式运行：

```powershell
.\scripts\run_official_deeplabv3plus_20260730.ps1 -Stage train
```

## 比较边界

本轮目标是回答：在同一 patient-level split、同一二值标签、同一最终指标下，官方 DeepLabV3+ 是否确实明显优于现有 TransUNet/green prior。

它不是对师妹“15.x HD95”的直接复刻，因为尚未知她使用的：原始数据版本、患者划分、mask 阈值、HD95 函数、是否做后处理与最优阈值选择。若官方复现结果仍显著更好，再逐项回查她的增强、迭代数和预处理；若不能显著更好，首先应怀疑评估口径/数据划分差异，而不是简单认定我们的模型更差。

## 已完成复现（2026-07-31）

已完成 scratch、seed 42 的 10000 iteration 官方训练。最佳 checkpoint 为 `results/official_deeplabv3plus_20260730/work_dirs/seed42_scratch_10k/best_mDice_iter_10000.pth`，由 val `mDice=87.79` 选出；随后使用本项目统一指标在 436 张 development-test 评估：

| Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7382 | 0.5956 | 0.7196 | 0.7809 | 23.34 | 0.8221 | 0.6214 |

相对 scratch TransUNet，Dice `-0.0140`、Recall `-0.0641`，但 Precision `+0.0381`、HD95 `-0.77`。官方 DeepLabV3+ 在当前统一口径下更保守，未超过 TransUNet/green prior；师妹的 HD95 15.x 仍需以其 checkpoint、split、标签规则和原始 HD95 实现进一步复核。
