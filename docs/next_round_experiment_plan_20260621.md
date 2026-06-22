# 下一轮实验计划：绿通道先验基础上的结构/边界损失探索

更新日期：2026-06-21

## 1. 当前出发点

当前最稳主线不是旧的 `green+CLAHE + MSE + Gradient`，而是：

```text
TransUNet + JointModel v1 + basic Enhancer
teacher_mode = green_only
loss_weighting = fixed
lambda_mse = 10
lambda_grad = 0
seg_loss = BCE + Dice
```

统一复评中，它相对重训 TransUNet baseline 的提升为：

| 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TransUNet baseline | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 |
| Ours green only MSE10 Grad0 | 0.7571 | 0.6193 | 0.7960 | 0.7399 | 23.27 | 0.8451 | 0.6477 |
| Delta | +0.0050 | +0.0053 | +0.0123 | -0.0030 | -0.84 | +0.0048 | +0.0073 |

结论：绿通道先验确实有效，但提升偏小。下一轮不再优先堆 enhancer 结构，而是先围绕“小血管、细边界、类别不平衡、拓扑连通性”做低成本损失函数探索。

## 2. 为什么先跑 loss sweep

甲襞毛细血管分割的困难点和视网膜血管/管状结构类似：目标细、前景占比低、边界模糊、断裂会影响结构分析。最近几年可借鉴方向里，最低成本且最贴合当前结果的是这些：

| 方向 | 对应实验 | 主要想改善 | 风险 |
|---|---|---|---|
| Focal Tversky | `focal_tversky` | 小目标和漏检，提升 Recall/Dice | 可能误检增加、Precision 降低 |
| Unified Focal | `unified_focal` | 类别不平衡下更稳的 CE/Dice 组合 | 超参不一定适合甲襞数据 |
| soft-clDice | `bce_dice_cldice` | 管状结构连通性，提升 clDice | 可能牺牲局部边界或 Dice |
| Boundary Dice | `bce_dice_boundary` | 轮廓和边界贴合，提升 Boundary F1/HD95 | 对标注噪声敏感 |
| clDice + Boundary | `bce_dice_cldice_boundary` | 同时约束中心线和边界 | 多损失叠加可能不稳定 |

这些实验的优点是：不改 backbone，不改变 green-prior 主线，只改变 segmentation loss。即使提升不大，也能形成很清楚的消融闭环：

```text
green physical prior consistency + structure-aware segmentation objective
```

## 3. 已实现的代码改动

### `losses/joint_loss.py`

新增/整理了以下损失：

- `BCEDiceLoss`
- `FocalTverskyLoss`
- `UnifiedFocalLoss`
- `SoftClDiceLoss`
- `BoundaryDiceLoss`
- `CompositeSegmentationLoss`
- `build_segmentation_loss(...)`

`JointDistillationLoss` 现在支持：

```text
--seg_loss bce_dice / focal_tversky / unified_focal / bce_dice_cldice / bce_dice_boundary / bce_dice_cldice_boundary
--cldice_weight
--boundary_weight
--focal_alpha
--focal_beta
--focal_gamma
```

默认仍是 `bce_dice`，所以旧训练逻辑不受影响。

### `train_unified.py`

新增：

```text
--seed
--seg_loss
--cldice_weight
--boundary_weight
--focal_alpha
--focal_beta
--focal_gamma
```

baseline 和 ours 都可以切换 segmentation loss。后续多 seed 稳定性也可以直接用 `--seed 42/43/44` 跑。

### `evaluate_all.py`

统一评估表现在会记录 loss 元信息：

```text
seg_loss, cldice_weight, boundary_weight, focal_alpha, focal_beta, focal_gamma
```

指标计算逻辑不变，只是 aggregate CSV/XLSX 能追溯实验配置。

### `scripts/run_seg_loss_sweep_20260621.ps1`

新增串联脚本，按顺序训练并统一评估 5 个实验。所有实验固定：

```text
mode = ours
dataset = all_filtered
teacher_mode = green_only
joint_model = v1
enhancer = basic
loss_weighting = fixed
lambda_mse = 10
lambda_grad = 0
threshold = 0.5
seed = 42
```

输出位置：

```text
results/seg_loss_sweep_20260621/run_summary.csv
results/seg_loss_sweep_20260621/metrics_summary.csv
results/seg_loss_sweep_20260621/logs
results/unified_eval_seg_loss_20260621
```

## 4. 直接运行命令

在项目根目录运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_seg_loss_sweep_20260621.ps1
```

如果想先用 1 epoch 快速检查脚本链路，不想等完整训练：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_seg_loss_sweep_20260621.ps1 -Epochs 1 -Patience 1
```

正式结果还是要用默认 `50 epoch / patience 20`。


## 4.1 先跑已有最佳权重的 val 阈值选择

导师 agent 建议把阈值选择列为 `P0`，因为当前 Ours 相比 TransUNet 的 Recall 更高、Precision 略低，固定 `threshold=0.5` 可能不是最优工作点。这个实验不重训，只做评估：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_threshold_selection_current_best_20260621.ps1
```

默认会对两个已有权重执行：

| 模型 | 权重 |
|---|---|
| TransUNet baseline | `results/experiments/all_filtered/baseline_retrain_20260619/0619_0232/best_model.pth` |
| Ours green MSE10 Grad0 | `results/experiments/all_filtered/ours_green_only_mse_only_20260620/0620_0616/best_model.pth` |

流程：

1. 在 `dataset_all_filtered/val` 上扫 `threshold=0.30:0.70:0.02`。
2. 默认按 `Dice` 选择最佳阈值。
3. 固定该阈值，在 `dataset_all_filtered/test` 上评估一次。
4. 输出到 `results/threshold_selection_20260621`。

如果想按结构指标选择阈值，也可以跑：

```powershell
.\scripts\run_threshold_selection_current_best_20260621.ps1 -SelectionMetric structure_combo
```

`structure_combo = Dice + 0.5 * clDice + 0.5 * BoundaryF1`。论文正式表格建议优先报告 Dice-selected threshold；structure_combo 可以作为补充分析，避免审稿人觉得我们过度挑指标。
## 5. 跑完后优先看什么

跑完后先打开：

```text
results/seg_loss_sweep_20260621/metrics_summary.csv
```

判断标准：

| 情况 | 下一步判断 |
|---|---|
| Dice >= 0.762 且 clDice/Boundary F1 不下降 | 可以把该 loss 升为新主线，立刻做 3-seed 稳定性 |
| Dice 只到 0.758-0.760，但 Boundary F1/clDice 明显更好 | 可作为结构指标分支，论文主张从 Dice 转向结构收益 |
| Dice/clDice/Boundary F1 都没有超过 MSE10 Grad0 | 保留为 negative ablation，说明简单结构损失不足，需要转向 backbone/结构改进 |
| Recall 大幅升、Precision 大幅降 | 小血管更敏感但误检增加，不宜作为主线 |
| HD95 变好但 Dice/Boundary F1 变差 | 只作为补充现象，不作为主方法 |

当前要打败的内部基准：

```text
Ours green only MSE10 Grad0:
Dice 0.7571, IoU 0.6193, clDice 0.8451, BoundaryF1 0.6477, HD95 23.27
```

强 baseline：

```text
TransUNet baseline:
Dice 0.7522, IoU 0.6140, clDice 0.8403, BoundaryF1 0.6405, HD95 24.11
```

## 6. 如果 loss sweep 仍然提升很小

下一步建议按成本从低到高做：

1. `P0` 多 seed 稳定性：TransUNet 和最佳 Ours 各跑 seed 42/43/44。
2. `P0` paired bootstrap + Wilcoxon：给 Dice、IoU、clDice、Boundary F1 补统计显著性。
3. `P1` val threshold selection：在 val 上统一选阈值，再一次性 test，不能用 test 调阈值。
4. `P1` 强 baseline：至少补一个 Attention U-Net / DeepLabV3+ / Swin-UNet 或 MedNeXt 类 baseline。
5. `P2` 结构方向：尝试 boundary/skeleton auxiliary head，而不是只在 loss 上加项。
6. `P2` backbone 方向：尝试 MedNeXt、U-Mamba/VM-UNet 作为新 backbone 或对照。

## 7. 文献依据和可借鉴点

- clDice: https://arxiv.org/abs/2003.07311  
  面向血管、神经等管状结构，强调中心线和拓扑连通性，适合解释 clDice/soft-clDice 实验。
- Boundary loss: https://arxiv.org/abs/1812.07032  
  针对高度类别不平衡的医学分割，用边界/轮廓视角补充区域 Dice/CE。
- Unified Focal loss: https://arxiv.org/abs/2102.04525  
  统一 Dice 和 CE 系列 loss，处理类别不平衡；论文里包含 DRIVE 视网膜血管实验。
- TransUNet: https://arxiv.org/abs/2102.04306  
  当前强 baseline 的文献基础。
- ANFC nailfold dataset/pipeline: https://arxiv.org/abs/2312.05930  
  甲襞毛细血管自动分析任务背景。
- Metrics Reloaded: https://arxiv.org/abs/2206.01653  
  支持我们使用问题导向指标，而不是只报 Accuracy/Dice。
- Retinal vessel evaluation inconsistency: https://arxiv.org/abs/2111.03853  
  支持统一评估口径、统一 threshold、统一 test split 的必要性。
- U-Mamba: https://arxiv.org/abs/2401.04722  
  中高成本 backbone 方向，适合以后作为新 baseline 或替换 TransUNet。
- VM-UNet: https://arxiv.org/abs/2402.02491  
  Mamba 医学分割 U-Net 方向，适合作为“近期模型对照”。
- MedNeXt: https://arxiv.org/abs/2303.09975  
  数据稀缺医学场景下的大核 ConvNeXt 风格分割网络，可作为比 TransUNet 更近的强 baseline。

## 8. 组会汇报时可以这么讲

当前不应该说“我们已经大幅领先”，而应该说：

```text
我们已经证明 green-channel prior 不是简单预处理，它作为一致性监督能在强 TransUNet 上带来稳定的小幅结构收益。由于 Dice 提升仍偏小，下一步将围绕血管任务最相关的结构连通性、边界和类别不平衡进行低成本 loss sweep；若找到更优组合，再做多 seed 和统计检验。如果 loss sweep 无法带来明显提升，则转向强 baseline 和结构辅助头/backbone 改进。
```