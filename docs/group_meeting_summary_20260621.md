# 甲襞毛细血管分割实验阶段总结与下一步计划

更新日期：2026-06-21  
数据集：`dataset_all_filtered/test`  
测试图数：436  
统一评估设置：`img_size=256`，`threshold=0.5`，同一套 `evaluate_all.py` / `utils/metrics.py` 指标实现。

## 1. 当前一句话结论

目前最适合作为新候选主线的是：

```text
TransUNet + JointModel v1 + basic Enhancer
teacher_mode = green_only
loss_weighting = fixed
lambda_mse = 10
lambda_grad = 0
```

也就是 **green prior 的 MSE consistency 有效，但 gradient consistency 当前不稳定**。

相对重训后的 TransUNet baseline，当前最佳候选 `Ours green only MSE10 Grad0` 的统一复评结果为：

| 模型 | Dice | IoU | Recall | Precision | Specificity | Accuracy | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransUNet baseline | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 0.9670 | 0.9482 | 24.11 | 0.8403 | 0.6405 |
| Ours green only MSE10 Grad0 | 0.7571 | 0.6193 | 0.7960 | 0.7399 | 0.9661 | 0.9485 | 23.27 | 0.8451 | 0.6477 |
| 差值 | +0.0050 | +0.0053 | +0.0123 | -0.0030 | -0.0008 | +0.0003 | -0.84 | +0.0048 | +0.0073 |

结论：指标提升是存在的，但幅度偏小；亮点主要是 Dice/IoU/clDice/Boundary F1 同时小幅提升，并且 direct input 对照证明不是简单预处理导致。

## 2. 可用结果总表

以下表格剔除了已确认异常的旧 `Ours green+CLAHE` 权重。旧 `Ours green+CLAHE` 统一复评 Dice 只有 0.5295，与旧日志不一致，不进入正式主表。

| 类别 | 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | UNet | 0.7374 | 0.5962 | 0.7189 | 0.7851 | 24.58 | 0.8231 | 0.6351 | 传统 CNN baseline |
| Baseline | UNet++ | 0.7484 | 0.6082 | 0.7291 | 0.7897 | 23.49 | 0.8391 | 0.6456 | Precision/Boundary 较好 |
| Baseline | TransUNet retrain | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 | 正式 transformer baseline |
| Direct input | TransUNet direct green | 0.7479 | 0.6086 | 0.8132 | 0.7128 | 25.13 | 0.8332 | 0.6247 | 直接喂 green，不如 Ours |
| Direct input | TransUNet direct CLAHE | 0.7425 | 0.6019 | 0.8032 | 0.7121 | 26.60 | 0.8293 | 0.6239 | 直接喂 CLAHE，不如 Ours |
| Direct input | TransUNet direct green+CLAHE | 0.7497 | 0.6102 | 0.7917 | 0.7323 | 24.00 | 0.8362 | 0.6303 | direct input 中最好，但仍低于 baseline/Ours |
| Ours old | Ours CLAHE only | 0.7552 | 0.6165 | 0.8063 | 0.7297 | 24.03 | 0.8427 | 0.6354 | teacher prior 是 CLAHE only |
| Ours old | Ours green only, MSE10 Grad30 | 0.7567 | 0.6184 | 0.8156 | 0.7230 | 23.26 | 0.8440 | 0.6348 | 原主线，Recall 高但 Precision/Boundary 弱 |
| Ours old | Ours green only multiscale | 0.7569 | 0.6192 | 0.7742 | 0.7589 | 23.28 | 0.8459 | 0.6489 | 数值不错，但后续 multiscale+MSE 未复现增益 |
| Ours old | Ours green only gated | 0.7544 | 0.6160 | 0.8166 | 0.7209 | 24.52 | 0.8418 | 0.6348 | 不建议继续优先投入 |
| Ours old | Ours inverse attention | 0.7531 | 0.6150 | 0.8116 | 0.7207 | 24.63 | 0.8390 | 0.6301 | 不建议继续优先投入 |
| Ours old | Ours learnable loss | 0.7552 | 0.6166 | 0.8053 | 0.7283 | 23.67 | 0.8414 | 0.6337 | 未明显优于固定权重 |
| Loss ablation | Ours MSE10 Grad0 | **0.7571** | **0.6193** | 0.7960 | 0.7399 | 23.27 | 0.8451 | **0.6477** | 当前最佳候选主线 |
| Loss ablation | Ours MSE5 Grad0 | 0.7571 | 0.6186 | 0.7973 | 0.7391 | 23.03 | 0.8443 | 0.6361 | 与 MSE10 接近 |
| Loss ablation | Ours MSE20 Grad0 | 0.7560 | 0.6180 | 0.7932 | 0.7400 | 23.41 | 0.8419 | 0.6435 | Boundary F1 尚可，但 Dice 低一点 |
| Loss ablation | Ours MSE10 Grad20 | 0.7557 | 0.6171 | 0.8169 | 0.7205 | 23.79 | 0.8413 | 0.6327 | gradient 后 Recall 高但 Precision/Boundary 降 |
| Loss ablation | Ours MSE10 Grad40 | 0.7566 | 0.6189 | 0.8010 | 0.7351 | **22.63** | 0.8425 | 0.6366 | HD95 最好，但整体不如 MSE-only |
| Loss ablation | Ours MSE10 Grad30 retrain | 0.7546 | 0.6160 | 0.8044 | 0.7284 | 22.77 | 0.8409 | 0.6335 | 原主线重训后不稳定 |
| Structure | Multiscale + MSE only | 0.7552 | 0.6168 | 0.7955 | 0.7362 | 24.69 | 0.8440 | 0.6397 | 没有叠加收益 |

## 3. Direct input 对照说明什么

Direct input 的目的，是验证“我们的提升是否只是因为把 green/CLAHE 预处理图直接喂给 TransUNet”。

结果：

| 对照 | Dice | 相对 TransUNet | 相对 Ours MSE10 Grad0 | 判断 |
|---|---:|---:|---:|---|
| Direct green | 0.7479 | -0.0042 | -0.0092 | 不足以解释 Ours 提升 |
| Direct CLAHE | 0.7425 | -0.0096 | -0.0146 | 明显较差 |
| Direct green+CLAHE | 0.7497 | -0.0025 | -0.0075 | direct input 中最好，但仍不如 Ours |

结论：**直接预处理输入不是主要原因**。这支持我们的创新叙事：green prior 更适合作为联合增强/一致性约束，而不是简单替换原始输入。

## 4. Loss 消融说明什么

原主线是 `lambda_mse=10, lambda_grad=30`，但后续实验显示：

| 实验 | Dice | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| MSE10 Grad0 | 0.7571 | 0.7960 | 0.7399 | 23.27 | 0.8451 | 0.6477 | 当前最好，Precision/Boundary 更平衡 |
| MSE5 Grad0 | 0.7571 | 0.7973 | 0.7391 | 23.03 | 0.8443 | 0.6361 | 与 MSE10 接近 |
| MSE20 Grad0 | 0.7560 | 0.7932 | 0.7400 | 23.41 | 0.8419 | 0.6435 | MSE 过大没有继续提升 |
| MSE10 Grad20 | 0.7557 | 0.8169 | 0.7205 | 23.79 | 0.8413 | 0.6327 | Recall 高但误检增加 |
| MSE10 Grad40 | 0.7566 | 0.8010 | 0.7351 | 22.63 | 0.8425 | 0.6366 | HD95 最好，但 Dice/clDice/Boundary 不最优 |
| MSE10 Grad30 retrain | 0.7546 | 0.8044 | 0.7284 | 22.77 | 0.8409 | 0.6335 | 原主线不够稳定 |

结论：**gradient consistency 目前不是稳定贡献项**。它可能改善 HD95，但会拉低 Precision、Boundary F1 或整体 Dice。当前更稳的主线应改为 **MSE-only distillation**。

## 5. 当前最佳候选的逐图 paired 分析

比较对象：`Ours green only MSE10 Grad0` vs `TransUNet baseline`。  
测试图数：436。  
差值定义：Ours - TransUNet。HD95 越低越好，所以 HD95 为负表示 Ours 更好。

| 指标 | 平均差值 | 近似 95% CI | 中位差值 | 提升图数 | 退化图数 | 解读 |
|---|---:|---:|---:|---:|---:|---|
| Dice | +0.00497 | [0.00110, 0.00884] | +0.00089 | 222 | 214 | 平均提升为正，但逐图胜负接近对半 |
| IoU | +0.00528 | [0.00100, 0.00957] | +0.00130 | 222 | 214 | 与 Dice 一致，小幅正向 |
| Recall | +0.01227 | [0.00690, 0.01764] | +0.00569 | 242 | 194 | 召回提升，但比旧 Grad30 主线温和 |
| Precision | -0.00297 | [-0.00735, 0.00141] | -0.00025 | 216 | 220 | Precision 基本持平，略降 |
| Specificity | -0.00084 | [-0.00177, 0.00010] | -0.00039 | 204 | 231 | 背景抑制基本持平，略降 |
| Accuracy | +0.00028 | [-0.00048, 0.00103] | +0.00005 | 219 | 216 | 基本持平 |
| HD95 | -0.83637 | [-2.35168, 0.67895] | 0.00000 | 207 | 190 | 平均略好，但波动大 |
| clDice | +0.00479 | [-0.00000, 0.00958] | +0.00134 | 228 | 208 | 接近正向边界，结构略有收益 |
| Boundary F1 | +0.00728 | [0.00241, 0.01215] | +0.00386 | 239 | 197 | 边界指标有更明确正向收益 |

这个 paired 分析很重要：虽然均值提升不大，但 Dice/IoU/Boundary F1 的平均差值为正，尤其 Boundary F1 对当前 MSE-only 主线更友好。

## 6. 创新点目前怎么讲

目前最稳的创新故事应从“复杂模块”转向“物理先验一致性”：

1. 甲襞毛细血管图像中，green channel 对血管结构有更强可见性。
2. 直接把 green/CLAHE 图作为输入并不能带来最好结果，说明简单预处理不足。
3. 将 green prior 作为 teacher，引导 enhancer 学习与物理先验一致的增强表示，可以小幅提升 Dice/IoU/clDice/Boundary F1。
4. 梯度一致性并非稳定贡献项，当前实验更支持 MSE consistency，而不是 edge-aware gradient consistency。
5. 方法收益主要是“稳健的小幅结构收益”，不是大幅性能碾压。

建议论文方法名暂时围绕：

```text
Green-prior guided enhancement / Green-channel prior consistency / Physical-prior distillation
```

不建议继续把 “gradient consistency” 放在主创新核心，除非后续实验能显著救回来。

## 7. SCI 2-3 区风险评估

当前结果可以支撑一篇较完整的实验论文雏形，但要冲 SCI 2-3 区仍有风险。

### 有利点

- 有统一评估入口，指标口径已规范。
- 有 direct input 对照，能证明不是简单预处理。
- 有 loss 消融，能解释 MSE consistency 是主要贡献。
- 指标覆盖较完整：Dice、IoU、Recall、Precision、Specificity、Accuracy、HD95、clDice、Boundary F1。
- 任务是甲襞毛细血管分割，场景有医学应用价值。

### 风险点

- 相对强 baseline 的 Dice 提升约 +0.005，幅度偏小。
- 逐图胜负接近对半，不是压倒性优势。
- 当前还缺多 seed / 重复训练稳定性证明。
- 还缺外部数据集或跨数据集验证。
- 方法结构本身不算特别复杂，如果只靠小幅指标提升，创新强度可能不够。
- 旧权重异常说明实验记录需要进一步规范，否则审稿时风险较高。

### 判断

目前更像是：

```text
可以支撑“方法有效 + 有一定创新”的论文雏形；
但若目标是 SCI 2-3 区，需要继续补强稳定性、统计分析和可视化证据。
```

如果后续能做到以下几点，投稿把握会明显变好：

1. 最佳主线多次复训稳定优于 TransUNet。
2. paired/bootstrapping 统计显示 Dice、IoU、Boundary F1 的提升可靠。
3. 可视化案例清楚展示细血管/边界改善。
4. 加入至少一个外部或替代 split 验证。
5. 文献叙事聚焦 green-channel physical prior，而不是泛泛地说加模块。

## 8. 接下来建议跑的实验

### P0：复训最佳候选，确认稳定性

当前最关键是证明 `MSE10 Grad0` 不是偶然。建议再跑 2 次不同 exp_name，但代码目前固定 seed=42，如果不改代码，多次复训随机性有限。更规范的做法是给 `train_unified.py` 加 `--seed` 参数，再跑 seed 43/44。

如果暂时不改代码，可以先跑一版复训：

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 0.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --exp_name all_filtered/ours_green_only_mse10_grad0_retrain_20260621
```

复评：

```powershell
$w=(Get-ChildItem results\experiments\all_filtered\ours_green_only_mse10_grad0_retrain_20260621 | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName + "\best_model.pth"
D:\anaconda3\envs\pytorch\python.exe evaluate_all.py --name Ours_green_only_mse10_grad0_retrain --model_type ours --weight $w --dataset all_filtered --split test --threshold 0.5 --batch_size 4 --teacher_mode green_only --enhancer basic --joint_model v1 --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 0.0
```

### P1：加 seed 参数后做 3-seed 稳定性

建议我下一步帮你改 `train_unified.py`，增加：

```text
--seed 42/43/44
```

然后跑：

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 0.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --seed 43 --exp_name all_filtered/ours_green_only_mse10_grad0_seed43
```

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 0.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --seed 44 --exp_name all_filtered/ours_green_only_mse10_grad0_seed44
```

同时最好给 TransUNet baseline 也补 seed 43/44，否则 reviewer 可能质疑只给 Ours 多 seed。

### P1：做 val threshold selection

当前统一阈值是 0.5。可以在 val 上为每个模型统一选阈值，再固定到 test。这个可能会改善 Dice/Precision/Recall 平衡，但必须注意：不能在 test 上调阈值。

建议后续新增脚本：

```text
select_threshold_on_val.py
```

流程：val 上扫 threshold 0.3-0.7，选 Dice 最优阈值；test 上只用这个阈值评估一次。

### P2：外部验证或替代 split

如果时间允许，建议跑一个替代 split 或外部子集验证。即使提升仍小，只要趋势一致，SCI 说服力会比单 split 强很多。

## 9. 组会汇报建议结构

1. 先说明统一复评已完成，旧权重异常已剔除。
2. 展示主表：TransUNet 0.7522，当前最佳 Ours 0.7571。
3. 展示 direct input 对照：直接 green/CLAHE 输入不如 Ours，证明不是简单预处理。
4. 展示 loss 消融：MSE-only 最稳，gradient loss 不稳定。
5. 展示 paired 分析：Dice/IoU/Boundary F1 平均差值为正，但提升幅度小。
6. 诚实说明 SCI 风险：提升小，需要多 seed、统计和可视化补强。
7. 下一步计划：复训最佳主线，做 seed 稳定性，做 val threshold selection，筛选成功/失败案例图。

## 10. 当前正式推荐主线

当前推荐写入后续实验记录的主线：

```text
Backbone: TransUNet
Framework: JointModel v1 + basic Enhancer
Teacher prior: green_only
Loss: BCE + Dice segmentation loss + MSE consistency
lambda_mse: 10.0
lambda_grad: 0.0
Evaluation: dataset_all_filtered/test, threshold=0.5
```

暂不推荐作为主线：

- `lambda_grad=30` 原主线：重训后 Dice 降至 0.7546，不稳定。
- multiscale enhancer：旧结果略好，但 multiscale + MSE-only 没有复现增益。
- gated / inverse attention / learnable loss：当前没有稳定收益。
- old Ours green+CLAHE 权重：统一复评异常，不可用于正式表格。
