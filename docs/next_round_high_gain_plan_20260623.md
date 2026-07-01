# 2026-06-23 下一轮冲击更大提升的实验路线

## 当前判断

当前最强 Dice 主线仍是：

```text
TransUNet + JointModel v1 + basic Enhancer + green_only teacher
BCE-Dice + MSE consistency(lambda_mse=10) + Grad0
```

统一 test Dice 约为 `0.7571`，相对正式 TransUNet baseline `0.7522` 只提升约 `+0.005`。这个幅度可以说明 green prior 有效，但如果目标是 SCI 2-3 区，仅靠这个提升偏弱。

最近一轮 `clDice + Boundary` 的意义不是提高 Dice，而是把结构指标推高：

| 模型 | Dice | Precision | clDice | Boundary F1 |
|---|---:|---:|---:|---:|
| Ours MSE10 Grad0 | 0.7571 | 0.7399 | 0.8451 | 0.6477 |
| Ours clDice+Boundary | 0.7567 | 0.7421 | 0.8533 | 0.6519 |

因此现在要找 1-2 个点级别的提升，优先级应从“小 loss tweak”转向：

1. 强初始化 / 预训练。
2. 更强 backbone 或 encoder-decoder baseline。
3. 结构化 teacher 与边界辅助分支。
4. 最后才是多 seed、统计显著性、可视化，作为论文证据链。

## 文献启发方向

### 1. 强 baseline / 强 backbone

近期医学分割里，TransUNet 已经不是唯一强 baseline。审稿人很可能期待看到更现代的 encoder/backbone 对照：

- MedNeXt：大核 ConvNeXt 风格医学分割网络，强调可扩展卷积 backbone。
- U-Mamba / VM-UNet：Mamba 类长程依赖建模，2024 后医学分割常见新 baseline。
- ImageNet encoder 的 DeepLabV3+ / FPN / Unet++：虽然不是最新概念，但作为强工程 baseline 很有价值。

我们本地环境已安装 `segmentation_models_pytorch` 和 `timm`，所以可以低成本先跑 SMP 强 baseline 探针。

目的不是立刻替代 Ours，而是回答：

```text
当前 0.75 左右是不是 TransUNet 没吃满？
如果强 encoder baseline 能到 0.77，说明我们要换 backbone/融合 green prior。
如果强 encoder baseline 也上不去，说明数据/标注/任务上限更强，论文应强调结构指标和统计证据。
```

### 2. 预训练 TransUNet

项目里已有官方 ImageNet21k 预训练：

```text
model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

这是 P0，因为它会改变全部对照口径。如果 pretrained TransUNet 自身大涨，旧的从头训练 baseline 不能作为最终主表。如果 Ours 在 pretrained baseline 上仍然稳定提升，论文说服力会强很多。

### 3. 结构化 teacher + cbDice

当前 green_only teacher 有效，但原始 green prior 可能过于弱。下一步尝试：

- `green_blackhat`：形态学 black-hat 提取暗细管结构。
- `green_clahe_blackhat`：先增强局部对比，再提暗管结构。
- `green_frangi`：血管滤波，但目前风险是噪声敏感或输出过弱。

对应损失用：

```text
BCE-Dice + cbDice + Boundary
```

如果 blackhat teacher 比 green_only 好，创新点会明显升级：从“绿通道先验”变成“绿通道结构先验”。

### 4. 边界辅助 / refinement head

纯 boundary loss 已经证明提升有限，但显式边界辅助头可能更有方法贡献：

```text
enhanced image + coarse mask -> boundary logits -> residual refine segmentation logits
```

如果 Dice 不涨但 Boundary F1 / HD95 明显改善，可以作为结构质量分支；如果 Dice 也涨，才适合升为主方法。

## 已新增/可运行脚本

### A. 预训练 P0

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_pretrained_p0_20260622.ps1
```

输出：

```text
results/pretrained_p0_20260622/metrics_summary.csv
results/unified_eval_pretrained_p0_20260622
```

重点看：

| 判断 | 下一步 |
|---|---|
| Ours pretrained 比 TransUNet pretrained 高 >= +0.005 | 立刻补 seed 43/44 |
| Ours pretrained Dice >= 0.762 | 有机会形成更强主线 |
| TransUNet pretrained 自身明显超过 Ours | 说明旧 baseline 被低估，要重排论文表 |

### B. SMP 强 baseline 探针

默认不下载 ImageNet 权重，先跑 scratch 版本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_smp_strong_baselines_20260623.ps1
```

如果你的环境能联网/已有缓存，可以跑 ImageNet encoder：

```powershell
.\scripts\run_smp_strong_baselines_20260623.ps1 -EncoderWeights imagenet
```

默认跑：

| 模型 | encoder | 目的 |
|---|---|---|
| DeepLabV3+ | ResNet34 | 强语义分割 baseline |
| FPN | ResNet34 | 多尺度特征融合 baseline |
| Unet++ | EfficientNet-B3 | 较强 encoder-decoder baseline |

输出：

```text
results/smp_strong_baselines_20260623/metrics_summary.csv
results/experiments/all_filtered/smp_*/.../aggregate_results.csv
```

判断：

| 结果 | 意义 |
|---|---|
| SMP 任一模型 Dice >= 0.77 | 当前 TransUNet backbone/训练不是最优，应考虑把 green prior 迁移到该 backbone |
| SMP 约 0.75-0.76 | 说明单纯换常规 backbone 不够，继续 P1/P2 |
| SMP 低于 TransUNet | 论文可继续以 TransUNet 为主 baseline，但仍需补预训练和多 seed |

### C. 结构 teacher + cbDice

普通版本：

```powershell
.\scripts\run_structure_teacher_cbdice_20260622.ps1
```

预训练版本：

```powershell
.\scripts\run_structure_teacher_cbdice_20260622.ps1 -UsePretrained
```

优先看：

1. `green_blackhat + cbDiceBoundary`
2. `green_clahe_blackhat + cbDiceBoundary`
3. `green_frangi + cbDiceBoundary`

### D. Boundary refine

普通版本：

```powershell
.\scripts\run_boundary_refine_20260622.ps1
```

预训练版本：

```powershell
.\scripts\run_boundary_refine_20260622.ps1 -UsePretrained
```

## 推荐运行顺序

### 最稳顺序

```text
1. run_pretrained_p0_20260622.ps1
2. run_smp_strong_baselines_20260623.ps1
3. 根据 1/2 的结果决定是否跑 structure teacher 或 boundary refine
```

如果晚上想多跑一点：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_pretrained_p0_20260622.ps1
.\scripts\run_smp_strong_baselines_20260623.ps1
```

如果预训练 P0 结果不错，再跑：

```powershell
.\scripts\run_structure_teacher_cbdice_20260622.ps1 -UsePretrained
```

## 论文策略

如果最终能做到：

```text
TransUNet/pretrained baseline Dice ~= 0.752-0.760
Ours final Dice >= baseline + 0.015 到 0.020
并且 clDice / Boundary F1 同时显著提升
```

那就可以比较自信地按 SCI 2-3 区目标组织论文。

如果 Dice 仍然只有 +0.005 左右，则论文主张要换成：

```text
green-channel physical prior improves structure-preserving nailfold capillary segmentation,
with statistically significant topology/boundary gains rather than large Dice dominance.
```

这种路线仍可写，但需要更强证据：多 seed、paired bootstrap、Wilcoxon、可视化成功/失败案例、复杂度和强 baseline 对照。

### E. SMP green prior 迁移实验

这一步用于回答一个关键问题：

```text
如果更强 encoder-decoder baseline 比 TransUNet 更适合这个任务，green prior 是否还能继续带来增益？
```

已新增训练入口：

```text
train_smp_joint.py
```

它保持当前 Ours 的核心设计：

```text
Enhancer + green teacher consistency + segmentation loss
```

但把 segmentor 从 TransUNet 换成 `segmentation_models_pytorch` 的强 encoder-decoder，例如 DeepLabV3+、FPN、Unet++。这比单纯跑 SMP baseline 更进一步：如果 SMP baseline 已经很强，而 SMP joint 还能提升，说明 green prior 不是依附于 TransUNet 的小技巧，而是可迁移的物理/结构先验。

默认运行：

```powershell
.\scripts\run_smp_green_prior_20260623.ps1
```

如果要顺便跑结构分支：

```powershell
.\scripts\run_smp_green_prior_20260623.ps1 -IncludeStructure
```

如果本地已有 ImageNet encoder 缓存，或环境能下载权重：

```powershell
.\scripts\run_smp_green_prior_20260623.ps1 -EncoderWeights imagenet
```

默认包含：

| 实验 | encoder | prior/loss | 目的 |
|---|---|---|---|
| SMP Joint DeepLabV3+ | ResNet34 | green_only + MSE10 Grad0 | 看强语义分割头是否吃到 green prior |
| SMP Joint FPN | ResNet34 | green_only + MSE10 Grad0 | 看多尺度融合 backbone 是否更适合细管结构 |
| SMP Joint Unet++ | EfficientNet-B3 | green_only + MSE10 Grad0 | 看强 encoder-decoder 是否抬高上限 |
| 可选结构分支 | ResNet34 / EfficientNet-B3 | clDice + Boundary | 看结构指标能否继续放大 |

判断逻辑：

| 结果 | 下一步 |
|---|---|
| SMP baseline 强，SMP joint 继续提升 | 把 Ours 主干迁移到该 SMP backbone，作为新主线 |
| SMP baseline 强，但 SMP joint 不提升 | green prior 可能只适合 TransUNet 或当前蒸馏方式，需改融合方式 |
| SMP baseline 不强，SMP joint 也不强 | 回到 TransUNet 预训练 + 结构 teacher 路线 |
| SMP joint Dice 达到 `0.765+` | 进入多 seed 稳定性验证 |
| SMP joint Dice 达到 `0.77+` | 有机会作为 SCI 2-3 区主表核心结果 |

注意：`results/experiments/all_filtered/_smoke_smp_joint_fpn_resnet18_green_20260623` 只是 1 epoch 链路测试，不能作为正式实验结果。
