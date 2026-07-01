# 甲襞毛细血管分割阶段汇报

更新日期：2026-06-26  
数据集：`dataset_all_filtered`  
测试图数：436  
统一评估设置：`img_size=256`，`threshold=0.5`，统一指标实现：Dice、IoU、Recall/Sensitivity、Precision、Specificity、Accuracy、HD95、clDice、Boundary F1。

## 1. 汇报主结论

本阶段主要完成了三件事：

1. 统一复评了已有模型和旧权重，修正了原始 xlsx/log/脚本之间不一致的问题。
2. 系统验证了 green channel prior、direct input、loss 消融、预训练、结构 teacher、SMP 强 baseline 和 SMP green-prior 迁移。
3. 方向从“堆叠复杂模块追求 Dice 提升”调整为“green-channel physical prior 对管状结构和边界保真的稳定改善”。

当前最好 Dice 结果是：

```text
TransUNet official pretrained + JointModel v1 + basic Enhancer
teacher_mode = green_only
loss = BCE-Dice + MSE consistency
lambda_mse = 10
lambda_grad = 0
```

测试 Dice 为 `0.7583`。

但需要诚实说明：相对旧 TransUNet baseline `0.7522`，提升为 `+0.0061`；相对预训练 TransUNet `0.7567`，只提升 `+0.0017`。目前还没有达到预期的 `+0.02` Dice 目标。因此后续如果目标是 SCI 2-3 区，不能只打“大幅性能提升”牌，而应强调：

```text
green-channel physical prior improves structure- and boundary-preserving
nailfold capillary segmentation.
```

## 2. 阶段性工作进展

### 2.1 统一评估体系

已经新增/规范了统一评估入口 `evaluate_all.py`，保证所有模型使用：

- 同一测试集：`dataset_all_filtered/test`
- 同一输入尺寸：`256 x 256`
- 同一阈值：`threshold=0.5`
- 同一指标实现：Dice、IoU、Recall、Precision、Specificity、Accuracy、HD95、clDice(专门评估拓扑连通完整性、CVPR)、Boundary F1(分割边界评估指标、CCF A)
- 输出 aggregate CSV/XLSX 和 per-image CSV

统一评估后发现旧 `Ours green+CLAHE` 权重异常：统一复评 Dice 只有 `0.5295`，与旧日志不一致，因此已经从正式结果中剔除。

### 2.2 主线方向调整

早期主线为：

```text
green_only teacher + MSE consistency + Gradient consistency
lambda_mse = 10
lambda_grad = 30
```

后续消融发现 gradient consistency 不稳定，常见现象是 Recall 上升但 Precision、Boundary F1 或 Dice 下降。因此当前主线调整为：

```text
green_only teacher + MSE consistency only
lambda_mse = 10
lambda_grad = 0
```

### 2.3 最近新增的高提升方向探索

为了尝试获得更大提升，后续又跑了：

- TransUNet ImageNet21k 预训练 P0
- 结构 teacher：green blackhat / green CLAHE blackhat / green Frangi
- cbDice + Boundary 结构损失
- SMP 强 baseline：DeepLabV3+、FPN、Unet++
- SMP + green prior 迁移
- SMP ImageNet encoder 版本

这些实验整体说明：预训练有用，但 green prior 的 Dice 增益仍小；SMP 系列没有超过 TransUNet；结构 teacher 没有带来 Dice 大幅提升。

## 3. 当前关键指标总表

### 3.1 主要 baseline 与当前最优模型

| 类别 | 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | UNet | 0.7374 | 0.5962 | 0.7189 | 0.7851 | 24.58 | 0.8231 | 0.6351 | 传统 CNN baseline |
| Baseline | UNet++ | 0.7484 | 0.6082 | 0.7291 | 0.7897 | 23.49 | 0.8391 | 0.6456 | Precision 较高 |
| Baseline | TransUNet old | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 | 旧正式 baseline |
| Baseline | TransUNet pretrained | 0.7567 | 0.6188 | 0.8016 | 0.7331 | 22.69 | 0.8489 | 0.6336 | 预训练提升明显 |
| Ours | Ours green MSE old | 0.7571 | 0.6193 | 0.7960 | 0.7399 | 23.27 | 0.8451 | 0.6477 | 旧 best Ours |
| Ours | Ours green MSE pretrained | **0.7583** | **0.6208** | 0.8031 | 0.7347 | **22.05** | 0.8465 | 0.6414 | 当前 Dice 最好 |
| Ours | Ours clDice+Boundary old | 0.7567 | 0.6183 | 0.7928 | 0.7421 | 23.55 | **0.8533** | **0.6519** | 当前结构指标最好 |

核心解释：

- 如果只看 Dice，当前最好是 `Ours green MSE pretrained`，Dice `0.7583`。
- 如果看结构质量，`Ours clDice+Boundary old` 的 clDice 和 Boundary F1 最强。
- 预训练 TransUNet 自身已达到 `0.7567`，说明旧 baseline 被低估；Ours 相对预训练 baseline 的 Dice 优势较小。

## 4. Direct Input 对照

Direct input 实验用于验证：green/CLAHE 是否只是简单预处理就能提升。

| 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 相对 TransUNet old |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TransUNet direct green | 0.7479 | 0.6086 | 0.8132 | 0.7128 | 25.13 | 0.8332 | 0.6247 | -0.0042 |
| TransUNet direct CLAHE | 0.7425 | 0.6019 | 0.8032 | 0.7121 | 26.60 | 0.8293 | 0.6239 | -0.0096 |
| TransUNet direct green+CLAHE | 0.7497 | 0.6102 | 0.7917 | 0.7323 | 24.00 | 0.8362 | 0.6303 | -0.0025 |
| Ours green MSE old | 0.7571 | 0.6193 | 0.7960 | 0.7399 | 23.27 | 0.8451 | 0.6477 | +0.0050 |

结论：

```text
直接把 green/CLAHE 图喂给网络并不能解释 Ours 的提升。
green prior 更适合作为 teacher consistency，而不是简单替换输入。
```

这个对照是论文里比较重要的消融证据。

## 5. Loss 与一致性约束消融

### 5.1 MSE / Gradient consistency

| 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MSE10 Grad0 | **0.7571** | **0.6193** | 0.7960 | 0.7399 | 23.27 | 0.8451 | **0.6477** | 当前最稳 |
| MSE5 Grad0 | 0.7571 | 0.6186 | 0.7973 | 0.7391 | 23.03 | 0.8443 | 0.6361 | 与 MSE10 接近 |
| MSE20 Grad0 | 0.7560 | 0.6180 | 0.7932 | 0.7400 | 23.41 | 0.8419 | 0.6435 | MSE 过大无明显收益 |
| MSE10 Grad20 | 0.7557 | 0.6171 | 0.8169 | 0.7205 | 23.79 | 0.8413 | 0.6327 | Recall 高但误检增多 |
| MSE10 Grad40 | 0.7566 | 0.6189 | 0.8010 | 0.7351 | **22.63** | 0.8425 | 0.6366 | HD95 较好，但综合不如 MSE-only |
| MSE10 Grad30 retrain | 0.7546 | 0.6160 | 0.8044 | 0.7284 | 22.77 | 0.8409 | 0.6335 | 原主线重训后不稳定 |

结论：

```text
MSE consistency 是当前主要有效项。
Gradient consistency 可能改善 HD95，但整体不稳定，不适合作为当前主线核心。
```

### 5.2 分割损失 sweep

| 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| focal_tversky | 0.7457 | 0.6056 | 0.8558 | 0.6785 | 26.26 | 0.8382 | 0.6007 | 淘汰，误检明显 |
| unified_focal | 0.7184 | 0.5744 | 0.8448 | 0.6489 | 31.12 | 0.7990 | 0.5638 | 明显失败 |
| bce_dice_cldice | 0.7507 | 0.6113 | 0.8208 | 0.7088 | 23.95 | 0.8504 | 0.6256 | clDice 提升但 Precision/Boundary 差 |
| bce_dice_boundary | 0.7546 | 0.6155 | 0.7914 | 0.7378 | 23.72 | 0.8422 | 0.6429 | 可作补充 |
| bce_dice_cldice_boundary | 0.7567 | 0.6183 | 0.7928 | 0.7421 | 23.55 | **0.8533** | **0.6519** | 结构指标最好 |

结论：

```text
Focal/Tversky 系列在当前任务上不适合，主要问题是 Recall 偏高、Precision 和边界变差。
clDice+Boundary 不提升 Dice 冠军，但明显增强结构指标，适合放入论文结构质量分析。
```

## 6. 预训练 P0 实验

| 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TransUNet old | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 | 旧 baseline |
| TransUNet pretrained | 0.7567 | 0.6188 | 0.8016 | 0.7331 | 22.69 | 0.8489 | 0.6336 | 预训练使 baseline 变强 |
| Ours green MSE pretrained | **0.7583** | **0.6208** | 0.8031 | 0.7347 | **22.05** | 0.8465 | 0.6414 | 当前 Dice 最好 |
| Ours clDice+Boundary pretrained | 0.7517 | 0.6137 | 0.8329 | 0.7031 | 22.96 | 0.8516 | 0.6230 | Recall 高但 Precision/Boundary 差 |

配对统计：

| 对比 | Dice 平均差值 | Dice Wilcoxon p | clDice 平均差值 | Boundary F1 平均差值 | 解读 |
|---|---:|---:|---:|---:|---|
| Ours green MSE pretrained - TransUNet pretrained | +0.0017 | 0.170 | -0.0024 | +0.0078 | Dice 不显著，Boundary F1 显著正向 |
| Ours green MSE pretrained - Ours green MSE old | +0.0012 | 0.655 | +0.0014 | -0.0063 | 预训练略涨 Dice/HD95，但 Boundary F1 下降 |

结论：

```text
预训练是必要对照，但没有带来 Ours 相对 baseline 的大幅增益。
预训练后 Ours 的优势主要体现在 Dice/HD95 略好和 Boundary F1 对预训练 baseline 有改善。
```

## 7. 结构 teacher + cbDice / Boundary

| 模型 | Teacher | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| green_blackhat + cbDiceBoundary pretrained | green_blackhat | 0.7559 | 0.6175 | 0.8016 | 0.7332 | 22.61 | 0.8512 | 0.6414 | 结构指标尚可，但 Dice 不如 green MSE |
| green_clahe_blackhat + cbDiceBoundary pretrained | green_clahe_blackhat | 0.7538 | 0.6160 | 0.8038 | 0.7294 | 23.63 | 0.8497 | 0.6410 | 未带来增益 |
| green_frangi + cbDiceBoundary pretrained | green_frangi | 0.7516 | 0.6126 | 0.7933 | 0.7342 | 23.74 | 0.8471 | 0.6308 | 不建议继续 |

结论：

```text
blackhat/frangi 结构 teacher 没有提升 Dice 上限。
这说明当前 teacher 设计可能过强或过窄，会牺牲部分区域分割性能。
```

后续不建议继续在 blackhat/frangi teacher 上投入大量训练。

## 8. SMP 强 baseline 与 SMP green-prior 迁移

### 8.1 SMP baseline

| 模型 | Encoder | 权重 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLabV3+ | ResNet34 | scratch | 0.7438 | 0.6033 | 0.7744 | 0.7369 | 24.47 | 0.8304 | 0.6243 |
| FPN | ResNet34 | scratch | 0.7448 | 0.6050 | 0.7781 | 0.7338 | 23.78 | 0.8320 | 0.6249 |
| Unet++ | EfficientNet-B3 | scratch | **0.7514** | **0.6130** | 0.7830 | 0.7419 | 24.48 | 0.8392 | 0.6391 |
| DeepLabV3+ | ResNet34 | ImageNet | 0.7456 | 0.6054 | 0.7868 | 0.7277 | 23.58 | 0.8341 | 0.6247 |
| FPN | ResNet34 | ImageNet | 0.7446 | 0.6045 | 0.7911 | 0.7243 | 25.69 | 0.8301 | 0.6131 |
| Unet++ | EfficientNet-B3 | ImageNet | 0.7505 | 0.6116 | 0.7894 | 0.7326 | 23.34 | 0.8395 | 0.6315 |

结论：

```text
SMP 系列没有超过 TransUNet。
当前数据上，TransUNet 仍然是更合适的主 baseline。
```

### 8.2 SMP + green prior

| 模型 | Encoder | 权重 | Loss | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMP Joint DeepLabV3+ | ResNet34 | scratch | BCE-Dice | 0.7456 | 0.6061 | 0.7487 | 0.7653 | 23.95 | 0.8316 | 0.6340 |
| SMP Joint FPN | ResNet34 | scratch | BCE-Dice | 0.7491 | 0.6098 | 0.7839 | 0.7360 | 22.97 | 0.8369 | 0.6297 |
| SMP Joint Unet++ | EfficientNet-B3 | scratch | BCE-Dice | 0.7480 | 0.6093 | 0.7883 | 0.7320 | 24.95 | 0.8369 | 0.6323 |
| SMP Joint FPN | ResNet34 | scratch | clDice+Boundary | 0.7444 | 0.6043 | 0.8118 | 0.7051 | 24.26 | 0.8379 | 0.6186 |
| SMP Joint Unet++ | EfficientNet-B3 | scratch | clDice+Boundary | 0.7502 | 0.6114 | 0.7922 | 0.7319 | 23.97 | 0.8458 | 0.6389 |
| SMP Joint DeepLabV3+ | ResNet34 | ImageNet | BCE-Dice | 0.7463 | 0.6066 | 0.7863 | 0.7298 | 25.14 | 0.8352 | 0.6226 |
| SMP Joint FPN | ResNet34 | ImageNet | BCE-Dice | 0.7490 | 0.6094 | 0.7898 | 0.7315 | 24.02 | 0.8382 | 0.6282 |
| SMP Joint Unet++ | EfficientNet-B3 | ImageNet | BCE-Dice | 0.7494 | 0.6109 | 0.7817 | 0.7392 | 22.90 | 0.8395 | 0.6349 |

结论：

```text
green prior 迁移到 SMP 后没有带来明显提升。
SMP 路线不建议作为当前论文主线。
可以作为强 baseline 对照保留，说明我们没有只和弱模型比较。
```

## 9. 当前整体判断

### 9.1 已经验证有效的点

- Green prior 比 direct input 更合理。
- MSE consistency 是当前最稳定的 green prior 使用方式。
- 预训练能提高 TransUNet 和 Ours 的绝对表现。
- clDice+Boundary 能显著改善结构/边界指标。
- 统一评估后，指标口径和结果记录更可靠。

### 9.2 被削弱或淘汰的点

- `green+CLAHE` 旧权重异常，不能用于正式表格。
- Gradient consistency 不稳定，不适合作为当前核心贡献。
- Focal/Tversky 类损失导致误检偏多，不建议主推。
- blackhat/frangi teacher 未提升 Dice。
- SMP baseline 和 SMP joint 没有超过 TransUNet，不建议继续作为主要优化路线。

### 9.3 方向调整

原来的叙事倾向：

```text
提出一个 edge-aware green-prior distillation framework，大幅提升分割性能。
```

现在更稳妥的叙事：

```text
提出 green-channel physical prior guided consistency learning，
在甲襞毛细血管这种低对比、细管状结构任务中，
带来小幅 Dice/IoU 提升，并显著改善拓扑和边界相关指标。
```

## 10. SCI 2-3 区风险评估

当前结果有论文雏形，但还不够硬。

### 有利证据

- 有完整统一复评。
- 有 direct input 对照证明不是简单预处理。
- 有 loss 消融说明 MSE consistency 是主要贡献。
- 有预训练 baseline 和 SMP 强 baseline。
- 有 clDice、Boundary F1、HD95 等结构相关指标。
- 结构指标有显著性证据，尤其旧 clDice+Boundary 分支。

### 主要风险

- Dice 最高只到 `0.7583`。
- 相对旧 TransUNet 只提升 `+0.0061`，相对预训练 TransUNet 只提升 `+0.0017`。
- 还没有达到预期 `+0.02` Dice。
- 单 seed 结果仍可能被审稿人质疑。
- 没有外部验证或 cross-validation。
- 如果论文主张写成“大幅提升”，风险较高。

### 当前建议

如果短期要准备论文或开题汇报，应避免说“显著提高 Dice”。更建议说：

```text
我们发现 green-channel physical prior 对甲襞毛细血管分割有稳定正向信号，
但其主要价值不是大幅提高 Dice，而是改善细小管状结构的边界和拓扑保真。
```

## 11. 今晚组会建议怎么讲

建议汇报顺序：

1. **研究问题**：甲襞毛细血管是细小、低对比、管状结构，普通 RGB 分割容易漏检或边界断裂。
2. **初始假设**：green channel 能增强血管可见性，但简单 direct input 可能不足。
3. **统一复评**：统一数据集、阈值和指标后，确认旧结果中存在权重异常，剔除异常结果。
4. **主实验结果**：Ours green MSE old Dice `0.7571`，Ours green MSE pretrained Dice `0.7583`。
5. **关键消融**：direct input 不如 Ours，MSE consistency 有效，gradient consistency 不稳定。
6. **高提升尝试**：预训练、结构 teacher、SMP baseline、SMP joint 都已跑；没有获得 +2 点 Dice。
7. **方向调整**：从“追求 Dice 大幅提升”转为“物理先验引导的结构/边界保真分割”。
8. **下一步计划**：多 seed 稳定性、统计检验、可视化案例、可能尝试 4/6 通道 prior input 或 semi-supervised/TTA/ensemble。

可以用这句话收尾：

```text
目前 green prior 的有效性已经验证，但提升幅度不足以单独支撑强性能贡献。
下一步重点不是继续堆小模块，而是补稳定性与统计证据，同时尝试更直接的 prior fusion 或半监督/TTA/ensemble 来冲击更大的 Dice 提升。
```

## 12. 下一步建议

### P0：先做论文证据链

1. 对 `TransUNet pretrained`、`Ours green MSE pretrained`、`Ours clDice+Boundary old` 做 paired bootstrap / Wilcoxon 正式统计表。
2. 从 per-image CSV 里筛选成功/失败案例，做可视化图：
   - 原图
   - GT
   - TransUNet
   - Ours green MSE
   - Ours clDice+Boundary
   - error map / boundary overlay / skeleton overlay
3. 补 3-seed 稳定性，至少 seed 42/43/44。

### P1：如果继续冲 +2 Dice 点

目前已经尝试过的方向基本都没有大提升。下一步更可能有效的是训练范式或输入融合，而不是继续微调 loss：

- `RGB + green prior` 4 通道输入融合
- `RGB + green + CLAHE + blackhat` 6 通道输入融合
- Test-time augmentation
- 简单 ensemble：TransUNet pretrained + Ours green MSE + Ours clDice+Boundary
- 半监督 pseudo-label，如果有未标注甲襞图像
- 按困难样本分层训练或 loss reweighting

### P2：外部验证或替代 split

如果数据允许，建议做：

- repeated split
- 5-fold cross-validation
- 外部数据集测试

这对 SCI 2-3 区比继续调 `0.001` Dice 更重要。

## 13. 可引用的结果文件

主要汇总文件：

- `results/latest_experiment_combined_summary_20260626.csv`
- `results/unified_eval/20260620_004729/aggregate_results.csv`
- `results/unified_eval_next_20260620/20260620_074510/aggregate_results.csv`
- `results/unified_eval_seg_loss_20260621/ours_green_mse10_grad0_cldice_boundary_20260621/20260621_182316/aggregate_results.csv`
- `results/pretrained_p0_20260622/metrics_summary.csv`
- `results/structure_teacher_cbdice_20260622/metrics_summary.csv`
- `results/smp_strong_baselines_20260623/metrics_summary.csv`
- `results/smp_green_prior_20260623/metrics_summary.csv`

当前建议后续正式主表至少保留：

- UNet
- UNet++
- TransUNet old
- TransUNet pretrained
- Direct green / CLAHE / green+CLAHE
- Ours green MSE old
- Ours green MSE pretrained
- Ours clDice+Boundary old
- SMP Unet++ EfficientNet-B3 baseline

