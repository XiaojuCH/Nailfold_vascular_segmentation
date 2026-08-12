# 2026-07-21 组会准备：从堆叠模块转向紧凑形态先验重构

## 当前结论

项目已经得到两类清楚的证据：

1. green image prior 是真实线索。旧三 seed scratch 实验中，green MSE10 Grad0 相对 TransUNet 的平均 Dice 为 +0.0066，并同步改善 clDice 与 Boundary F1。
2. 复杂方向融合与双教师 KD 还不是可发表的单模型主线。F3 主要带来 HD95/Boundary F1 信号；双教师 ensemble Dice 为 0.7636，而最佳单模型 K2 为 0.7609，未完整保留 ensemble 的边界收益。
3. 本周的固定局部 green contrast 与 boundary/centerline 辅助任务都完成了强增强复核，但没有在 val 上稳定超过同协议 TransUNet。因此 CGMA 结构辅助线在本轮正式停止，不将 development-test 的微小涨点包装为主线结论。

因此，下一步不再搜索 KD 系数、增加 strip branch，或继续微调 CGMA 辅助损失。后续资源优先投向强 baseline（nnU-Net v2/MedNeXt）、患者级外层验证，以及误检来源和标注上限分析；green prior 保留为已验证的候选信息源。

dataset_all_filtered/test 已参与前期研发决策，以下统一称 development-test。7 月 18--20 日只以 val 选模型；development-test 只在一整套预注册候选结束后统一评估。论文最终必须以患者级 outer-CV 或未参与选模的 final holdout 确认。

## 2026-07-10--21 完整实验档案

本节保留 7 月 10 日至 21 日期间所有已完成训练的结果，包括负结果。除另有说明外，development-test 为固定的 436 张图、`img_size=256`、阈值 0.5；表中保留 Dice、HD95、clDice、Boundary F1 以及相对同协议对照的 Dice 变化。完整的 IoU、Recall、Precision、Specificity、Accuracy 和逐图结果均保存在每轮对应的 `metrics_summary.csv` 与 unified-eval 目录中。

重要说明：不同轮次的初始化方式、增强策略、模型实现和 checkpoint 选择协议不同，绝对 Dice 不能跨轮次直接排名；只比较同一表中明确的对照。此前被反复查看的 `test` 均属于 development-test，不作为论文最终独立测试结果。

### A. 旧 green-MSE 主线与 C3 的三 seed 复现

这里的 C3 为旧实现的 `anisotropic + no BN + no intensity augmentation` 组合。后续代码审计确认其 `1x7 -> 7x1` 串联更接近因式分解的大二维卷积，并非真正独立的方向卷积；因此它是重要的历史对照，不能与后来的 F2/F3 真实并行方向分支混称为同一实现。

| 初始化（seed 42/43/44） | 方法 | Dice mean +/- std | 相对 TransUNet Dice | HD95 mean | clDice mean | Boundary F1 mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| scratch | TransUNet | 0.7518 +/- 0.0009 | baseline | 23.64 | 0.8403 | 0.6373 | 对照 |
| scratch | green MSE10 Grad0 | 0.7584 +/- 0.0011 | +0.0066 | 23.42 | 0.8467 | 0.6468 | 本项目最稳定的 green-prior 证据 |
| scratch | C3 old anisotropic/no-BN/no-aug | 0.7576 +/- 0.0025 | +0.0058 | **22.62** | 0.8461 | 0.6453 | HD95 改善，但 Dice 不稳定且未超过 old Ours |
| ImageNet21k pretrained | TransUNet | 0.7569 +/- 0.0012 | baseline | 22.95 | 0.8494 | 0.6369 | 强初始化对照 |
| ImageNet21k pretrained | green MSE10 Grad0 | 0.7578 +/- 0.0027 | +0.0009 | **22.04** | 0.8481 | 0.6370 | 预训练后 Dice 边际收益被压缩 |
| ImageNet21k pretrained | C3 old anisotropic/no-BN/no-aug | **0.7592 +/- 0.0022** | +0.0023 | 22.45 | **0.8499** | **0.6455** | 平均最高，但 seed44 回落，不能作为最终主线 |

| 口径 | seed42 Dice（TransUNet / green MSE / C3） | seed43 | seed44 |
|---|---|---|---|
| scratch | 0.7522 / 0.7571 / 0.7583 | 0.7524 / 0.7588 / 0.7549 | 0.7507 / 0.7593 / 0.7597 |
| pretrained | 0.7567 / 0.7583 / 0.7606 | 0.7583 / 0.7549 / 0.7603 | 0.7559 / 0.7602 / 0.7566 |

来源：`results/scratch_delta_multiseed_20260710/metrics_summary.csv`。该表也是“从头训练下 green prior 的提升程度比预训练下更清楚”的主要证据。

### B. 四个创新点初筛（E1--E4，seed42）

共同设置为旧 green-MSE 主线，E1--E4 分别检验各向异性 enhancer、decoder feature consistency、末端 CNN-Transformer 融合、BN/强度增强的影响。预训练口径的同协议 TransUNet 为 Dice 0.7567；scratch 口径为 Dice 0.7522。

| 口径 | 实验 | Dice | Delta Dice vs TransUNet | HD95 | clDice | Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| pretrained | E1 anisotropic enhancer | 0.7576 | +0.0009 | 23.13 | 0.8476 | **0.6514** | 边界信号，Dice 未超过 old Ours |
| pretrained | E2 decoder feature consistency V1 | 0.7502 | -0.0065 | 25.07 | 0.8399 | 0.6260 | 明确失败 |
| pretrained | E3 final-layer dual fusion | 0.7546 | -0.0021 | 23.38 | 0.8443 | 0.6418 | 暂停；不是完整多尺度 TransFuse |
| pretrained | E4a basic enhancer, no BN | 0.7573 | +0.0006 | 23.14 | 0.8469 | 0.6434 | 弱信号 |
| pretrained | E4b no intensity augmentation | 0.7581 | +0.0014 | 22.81 | 0.8488 | 0.6464 | 支持强度先验消融，未超过 old Ours |
| scratch | E1 anisotropic enhancer | 0.7554 | +0.0032 | 23.31 | 0.8421 | 0.6454 | 不如 old Ours |
| scratch | E2 decoder feature consistency V1 | 0.7549 | +0.0026 | **22.62** | 0.8412 | 0.6361 | Dice/结构不足，暂停 |
| scratch | E3 final-layer dual fusion | 0.7546 | +0.0024 | 22.76 | 0.8428 | **0.6477** | 边界略好，但 Dice 不够 |
| scratch | E4a basic enhancer, no BN | 0.7578 | +0.0055 | 22.91 | 0.8438 | 0.6379 | 单次弱阳性，后续未形成主线 |
| scratch | E4b no intensity augmentation | 0.7568 | +0.0046 | 23.90 | 0.8461 | 0.6377 | 不如 old Ours |

来源：`results/four_innovation_probes_20260710/metrics_summary_complete.csv`。E2 的 V1 为同一网络双视图的裸 decoder MSE，不是独立教师蒸馏；因此负结果不能泛化为“所有 decoder distillation 都无效”。

### C. 形态学/强度先验组合与结构损失（C1--C4，seed42）

| 口径 | 实验 | Dice | Delta Dice vs TransUNet | HD95 | clDice | Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| pretrained | C1 old anisotropic + no intensity aug | 0.7592 | +0.0025 | 22.40 | 0.8496 | **0.6542** | 单次边界指标最佳，后续 C3 多 seed 检验 |
| pretrained | C2 no BN + no intensity aug | 0.7488 | -0.0079 | 24.70 | 0.8398 | 0.6190 | 明确失败 |
| pretrained | C3 old anisotropic + no BN + no intensity aug | **0.7606** | **+0.0039** | 22.31 | 0.8494 | 0.6463 | 单次 Dice 最好，三 seed 后仅保留为候选 |
| pretrained | C4 C1 + soft clDice/Boundary loss | 0.7559 | -0.0008 | 22.60 | **0.8542** | 0.6461 | clDice 上升但 Dice 下降，停止调主损失 |
| scratch | C1 old anisotropic + no intensity aug | 0.7561 | +0.0039 | 24.26 | 0.8435 | 0.6451 | 不如 old Ours |
| scratch | C2 no BN + no intensity aug | 0.7573 | +0.0051 | 23.99 | 0.8449 | **0.6484** | 弱信号 |
| scratch | C3 old anisotropic + no BN + no intensity aug | 0.7583 | **+0.0061** | **22.11** | 0.8466 | 0.6481 | seed42 阳性，但三 seed Dice 不稳定 |
| scratch | C4 structure loss | 未完成 | - | - | - | - | 手动中断，不纳入比较 |

来源：`results/morph_intensity_combo_20260710/metrics_summary.csv`。C3 的三 seed 汇总已在 A 节保存，结论是它改善 HD95，但没有稳定超越 green MSE10 Grad0。

### D. 更规范的 decoder distillation V2（seed42）

V2 使用独立、冻结的 direct-green teacher，仅蒸馏最终 decoder layer，采用 `1x1 projection + cosine_mse`，且蒸馏权重从 V1 的 1.0 降至 0.1。随机前反向、真实 1 epoch、teacher 冻结、特征 shape 和权重 reload 均已通过；这不是路径、shape 或 teacher 文件错位导致的失败。

| 实验 | Dice | Delta Dice vs pretrained TransUNet | HD95 | clDice | Boundary F1 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| DecoderDistillV2 direct-green teacher | 0.7543 | -0.0024 | 23.44 | 0.8438 | 0.6290 | 仍低于 baseline、green MSE 和 C3，停止该线 |

来源：`results/decoder_distill_v2_20260710/metrics_summary.csv`。

### E. 审计后真实并行方向先验融合（F0--F3）

代码审计后，F2/F3 使用独立并行 `3x3/1x7/7x1/1x21/21x1` 分支与方向门控，不再把旧 C3 的串联大核表述为方向卷积。所有模型由 val Dice 选择权重，完成整套候选后才统一评估 development-test。

| 初始化 | 方法 | val Dice | dev-test Dice | Delta Dice vs F0 | HD95 delta vs F0 | clDice delta | Boundary F1 delta | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| scratch | F0 corrected TransUNet | 0.7897 | 0.7535 | baseline | 0.00 | 0.0000 | 0.0000 | 对照 |
| scratch | F1 plain green single | 0.7897 | 0.7463 | -0.0072 | +1.33 | -0.0050 | +0.0028 | 停止 |
| scratch | F2 directional single | 0.7896 | 0.7517 | -0.0018 | +0.05 | -0.0012 | +0.0045 | 停止 |
| scratch | F3 directional multiscale | **0.7938** | 0.7536 | +0.0001 | **-0.47** | +0.0025 | **+0.0089** | 仅结构指标阳性，保留为互补教师 |
| ImageNet21k pretrained | F0 corrected TransUNet | 0.7932 | **0.7571** | baseline | 0.00 | 0.0000 | 0.0000 | 强初始化对照 |
| ImageNet21k pretrained | F3 directional multiscale | 0.7920 | 0.7542 | -0.0029 | +0.60 | -0.0005 | -0.0112 | 预训练下失败，停止 F3 扩展 |

来源：`results/directional_prior_probes_20260715/metrics_summary.csv`、`results/pretrained_f0_f3_pair_20260715/metrics_summary.csv`。F3 scratch 的患者级结构信号存在，但 Dice 95% CI 跨 0；因此它不作为单模型 Ours，而是双专家实验中的形态教师。

### F. 双专家 ensemble 与离线 soft-KD（K0--K4，seed42）

教师为 pretrained RGB F0 与 scratch directional F3。两套模型按顺序推理并保存 float16 `.npy` soft target/disagreement，train/val/test 文件数分别为 1838/449/436，均已核对文件名、shape 与概率范围；student 为单个 RGB TransUNet，推理不保留第二教师。

| 模型 | Dice | 相对 K0 Dice | 相对原始 F0 Dice | HD95 | clDice | Boundary F1 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| F0 pretrained RGB | 0.7571 | +0.0010 | baseline | 21.72 | 0.8468 | 0.6346 | 单模型教师 |
| F3 scratch directional | 0.7536 | -0.0025 | -0.0035 | 23.70 | 0.8429 | 0.6384 | 形态教师 |
| 0.5 probability ensemble | **0.7636** | +0.0074 | **+0.0064** | **20.97** | **0.8542** | **0.6483** | 双模型上限，不能报为单模型 |
| K0 fine-tune control | 0.7561 | baseline | -0.0010 | 21.90 | 0.8490 | 0.6451 | 排除额外训练 |
| K1 uniform KD, lambda=0.3 | 0.7585 | +0.0024 | +0.0014 | 21.92 | 0.8496 | 0.6448 | 弱正向 |
| K2 uniform KD, lambda=1.0 | **0.7609** | **+0.0047** | **+0.0037** | **21.35** | **0.8537** | 0.6451 | 最优单模型 KD，但未达预注册保留阈值 |
| K3 agreement KD, lambda=0.3 | 0.7575 | +0.0014 | +0.0004 | 22.61 | 0.8481 | **0.6465** | HD95 退化，不保留 |
| K4 uniform KD, lambda=0.1 | 0.7558 | -0.0003 | -0.0013 | 22.47 | 0.8498 | 0.6436 | 退化 |

K2 vs K0 的患者级 Dice delta 为 `+0.00265`（95% CI `+0.00035` 到 `+0.00517`），但 K2 的绝对 Dice 0.7609 未达预设的 0.7615 保留线，也未达到 0.7630 强阳性线。因此不补 KD 多 seed，也不继续在当前 development-test 搜索 KD 系数。来源：`results/dual_teacher_kd_20260717/metrics_summary.csv`、`results/dual_teacher_kd_20260717/fallback_metrics_summary.csv`、`results/dual_teacher_kd_20260717/K2_vs_K0_patient_bootstrap.csv`。

## 2026-07-20 首轮 CGMA 结果

M0--M3 已按同一 scratch、seed42、BCE-Dice、intensity augmentation off、val-Dice checkpoint 选择完成。四组训练期间未读取 development-test；development-test 仅在四组都完成后统一评估。

| 模型 | val Dice | 相对 M0 | development-test Dice | 相对 M0 | HD95 delta | clDice delta | Boundary F1 delta | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0: no prior, no auxiliary | 0.7886 | baseline | 0.7452 | baseline | 0.00 | 0.0000 | 0.0000 | 当前 no-intensity control |
| M1: fixed green local contrast | 0.7887 | +0.0001 | 0.7515 | +0.0063 | -0.45 | +0.0038 | +0.0049 | 有单因素 development signal，但 val 未通过预设门槛 |
| M2: boundary + centerline auxiliary | 0.7885 | -0.0002 | 0.7537 | +0.0084 | -0.86 | +0.0068 | +0.0080 | 当前最强单因素 development signal |
| M3: prior + auxiliary | 0.7889 | +0.0003 | 0.7504 | +0.0052 | -0.28 | +0.0046 | +0.0040 | 未超过 M1/M2，不支持两因素叠加 |

自动决策为 stop_cgma_and_review_factors，因为 M1/M2/M3 的 val 结果均未达到预先固定阈值，M3 也没有超过两个单因素。这条结论应被保留，不能因为 development-test 的数字较好而事后改变为“正式成功”。

但 M2 相对 M0 的 development-test 患者级 paired bootstrap 显示，19 位患者中 14 位 Dice 改善：Dice +0.00780 (95% CI +0.00221 to +0.01393)，HD95 -2.24 (-4.61 to -0.37)，clDice +0.00706 (+0.00103 to +0.01358)，Boundary F1 +0.00812 (+0.00099 to +0.01554)。这不是少数病例造成的偶然涨点；其代价也清楚，即 Recall 在所有 19 位患者上升而 Precision 在所有 19 位患者下降。

重要边界：M0 的 development-test Dice 0.7452 明显低于此前 intensity augmentation on 的 corrected scratch TransUNet 0.7535。因此 M2 的 0.7537 主要证明它可补偿关闭 intensity augmentation 带来的损失，尚未超过更强的 intensity-on baseline。不能直接把 M2 +0.0084 解释成相对最佳 scratch baseline 的净提升。

上述分析因此促成了强增强 S0--S4 复核：固定 prior 从已经增强后的输入 RGB 在线计算，令 G' = alpha G + beta 时，未发生 clip 的区域满足 ReLU(blur(G') - G') = alpha ReLU(blur(G) - G)。亮度偏移 beta 被局部差分抵消，正对比缩放只改变幅度，因此它不像旧 image-MSE teacher 那样与 photometric augmentation 天然冲突。S0--S4 的结果列于下一节：理论上的增强相容性并未转化为 val 上的稳定分割增益，故不继续重跑 M3 或扩展 CGMA。

## 2026-07-20--21 强增强复核：S0--S4 已完成

该轮专门验证：M2 的结构收益能否在 `intensity augmentation on` 的较强训练协议下成立，并拆开 centerline 的贡献。所有模型固定 scratch、seed42、BCE-Dice、50 epochs、patience20、batch4、lr 1e-4；仅按 val Dice 选择 checkpoint，五组结束后才统一评估 development-test。

| 编号 | 强度增强 | prior | boundary weight | centerline weight | 要回答的问题 |
|---|---|---|---:|---:|---|
| S0 | on | off | 0.00 | 0.00 | 同协议 TransUNet control |
| S1 | on | fixed local contrast | 0.00 | 0.00 | M1 的 prior 在强协议下是否仍有效 |
| S2 | on | off | 0.10 | 0.10 | 原始 M2 是否在强协议下仍有净收益 |
| S3 | on | off | 0.05 | 0.05 | 降低辅助权重能否减少 Recall 上升和 Precision 下降 |
| S4 | on | off | 0.10 | 0.00 | centerline head 是否真的必要 |

| 实验 | val Dice | 相对 S0 | development-test Dice | 相对 S0 | HD95 delta | clDice delta | Boundary F1 delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S0: intensity-on TransUNet | 0.7904 | baseline | 0.7513 | baseline | 0.00 | 0.0000 | 0.0000 | 同协议对照 |
| S1: fixed local contrast prior | 0.7900 | -0.0004 | 0.7534 | +0.0021 | -0.63 | +0.0018 | +0.0002 | test 有极弱增益，但 val 未复现，不能进入多 seed |
| S2: boundary + centerline, 0.10/0.10 | 0.7893 | -0.0011 | 0.7523 | +0.0010 | -1.13 | +0.0036 | +0.0008 | 结构距离指标有信号，但 Precision -0.0062，倾向过分割 |
| S3: boundary + centerline, 0.05/0.05 | 0.7885 | -0.0019 | 0.7501 | -0.0012 | +0.41 | -0.0022 | -0.0107 | 明确失败 |
| S4: boundary only, 0.10 | 0.7875 | -0.0029 | 0.7517 | +0.0004 | -0.63 | -0.0004 | +0.0012 | Precision +0.0090、Recall -0.0095，只是更保守的取舍 |

预先固定的继续条件为：候选须在 val Dice 相对 S0 至少 +0.001，且 val Boundary F1 不下降、HD95 不变差，才补 S0 和候选的 seed43/44。结果中最佳 val Dice 的 S1 仍为 -0.0004，未满足条件；自动决策为 `stop_structure_auxiliary_line_and_prepare_group_meeting`。

这轮是对 no-intensity CGMA 信号的重要反证：S2 虽仍获得较好的 HD95/clDice，但没有在强协议形成稳定 Dice 增益，并再次暴露 Recall 上升、Precision 下降。故不补多 seed、不继续调权重，也不将 S1/S2 作为论文最终 Ours。

结果归档：`results/structure_aux_intensity_on_20260720/metrics_summary.csv`、`results/structure_aux_intensity_on_20260720/decision.json`、`results/unified_eval_structure_aux_intensity_on_20260720/20260721_073348/aggregate_results.csv`。

## 为什么需要减法式大改

### 现有 F3 的实际诊断

F3 虽有 5 个方向尺度分支和 3 个 decoder 注入点，但训练日志显示：

| 观察 | 最佳 val checkpoint 记录 | 含义 |
|---|---|---|
| 方向 gate 塌缩 | scratch F3: horizontal7=0.778，其余四支合计 0.222 | 五分支没有形成充分的多方向协作 |
| 深层融合没有正贡献 | 1/4、1/2 尺度 alpha 为 -0.026、-0.041；最终尺度为 +0.072 | 当前证据只支持最终 decoder 尺度的 prior 注入 |
| 标注方向高度集中 | train/val skeleton 边中约 67% 为 70--110 度近垂直方向 | 条带卷积的高权重不能直接解释为血管方向建模成功 |

所以，不再把更多卷积核或更多融合尺度当作默认优化方向。

### 绿色先验的训练集审计

固定局部暗管对比图定义为：

~~~
P_green = ReLU(GaussianBlur(green, sigma=9) - green)
~~~

它强调比局部背景更暗的细管区域。sigma 仅由训练集审计选定，再在 val 上确认，未使用 development-test。

| 信号 | train 像素 AUC，固定 200 张样本 | val 像素 AUC | 解释 |
|---|---:|---:|---|
| -green 原始暗度 | 0.6588 | 0.7503 | 原始 green 有用，但受整体亮度变化影响 |
| GaussianBlur sigma9 - green | 0.8295 | 0.8109 | 局部对比更能区分血管和背景 |

此审计不是分割性能，不能写入主结果；它是选择固定 morphology prior 的依据。

## 已检验候选：Compact Green-Contrast Morphology Adapter

### 结构

~~~
RGB ------------------------> TransUNet encoder/decoder --------> segmentation logits
                                      |
green -> fixed local contrast P_green -> tiny adapter -> one final decoder residual gate
                                      |
                                      +-> boundary auxiliary logit, training only
                                      +-> centerline auxiliary logit, training only
~~~

模型只做三件事：

1. RGB 原图直接进入 TransUNet；删除 enhancer、teacher image MSE 和 gradient MSE。
2. 固定 P_green 经一个很小的 1 到 16 channel adapter，只在最终 decoder feature 注入一次；删除五条 strip branch 和三尺度 fusion。
3. 主 segmentation head 仍使用 BCE-Dice。共享 decoder feature 另接 boundary 与 centerline 辅助头；两者只在训练期监督，不在推理期反馈 logits。

最终尺度采用 identity-preserving gate：

~~~
F_out = F_decoder + sigmoid(gamma) * sigmoid(G([F_decoder, P_proj])) * P_proj
~~~

gamma 从较小负值开始，使初始化接近 TransUNet，但 adapter 仍可获得梯度。P_green 不做逐图 min-max normalization，避免重新破坏强度关系。

### 这是减法式候选，而非堆叠

| 删除 | 替换 | 原因 |
|---|---|---|
| enhancer、green teacher image MSE、gradient MSE | 固定 blur-green 对比先验 | 直接使用可审计的局部物理信号 |
| 5 条 strip branch、softmax direction gate | 单一 prior adapter | F3 gate 已塌缩，额外分支没有独立贡献 |
| 三尺度 decoder fusion | 最终尺度一次融合 | F3 的深层注入为负残差 |
| clDice 或 Boundary 直接加到主 logits | 独立 boundary 和 centerline 辅助监督 | 旧结构 loss 存在 Dice 与 Precision 冲突 |
| 双教师 feature KD | 无教师单模型 | K2 只迁移部分 ensemble 优势，继续调 KD 信息增益低 |

## 文献依据

| 文献 | 可借鉴的结论 | 对本项目的约束 |
|---|---|---|
| Ye and Yin, Microvascular Research, 2024，PMID 38484792，DOI 10.1016/j.mvr.2024.104680 | 甲襞分割的核心难点是背景分离、模糊边界和高分辨率结构恢复 | 优先解决细管边界，而非继续换通用模块 |
| Qiu et al., IEEE JBHI, 2024，PMID 39137084，DOI 10.1109/JBHI.2024.3442528 | 结构或骨架知识可进入 TransUNet 表示层 | 小数据集先用轻量 centerline auxiliary task，不直接引入图网络 |
| Huang et al., Computers in Biology and Medicine, 2024，PMID 38461696，DOI 10.1016/j.compbiomed.2024.108255 | prior-supervised edge-aware multi-task 可改善模糊血管边界 | 支持独立 boundary head，而不是硬加 boundary loss |
| Jian and Wu, Medical and Biological Engineering and Computing, 2024，PMID 38898202，DOI 10.1007/s11517-024-03150-8 | 方向、连续性和边界是不同目标 | 将先验和结构监督解耦 |
| Shu et al., Frontiers in Cell and Developmental Biology, 2026，PMID 42293763，DOI 10.3389/fcell.2026.1825518 | 血管 KD 应关注形态重构，而非裸 feature mimic | 支持停止当前 decoder MSE 和 KD 系数搜索 |

## 已完成的首轮实验矩阵

首轮采用 scratch，不是因为预训练无效，而是此前 green prior 对 scratch baseline 的边际收益更清楚。原计划只有 scratch 候选通过门槛后，再成对补 pretrained M0/M3；实际未达到该条件，因此没有启动该分支。

统一设置：dataset_all_filtered 固定 patient split；seed42；BCE-Dice 主损失；intensity augmentation off；50 epochs；patience20；batch4；lr 1e-4；按 val Dice 保存 checkpoint；每个 run 保存 config、val per-image CSV 和 target 版本。

| 编号 | fixed local-contrast prior | boundary 和 centerline auxiliary heads | 目的 |
|---|---|---|---|
| M0 | 否 | 否 | 同协议 TransUNet 对照，排除 no-intensity confound |
| M1 | 是 | 否 | 固定局部 green contrast 是否优于 image MSE |
| M2 | 否 | 是 | 结构辅助任务本身是否有效 |
| M3 | 是 | 是 | 完整模型，检验先验与结构监督是否互补 |

boundary target 由 GT mask 形态梯度生成，centerline target 由二值 skeleton 生成；所有几何增强后再生成 target，避免旋转造成 target 错位。辅助权重首轮固定 boundary=0.10、centerline=0.10，不做网格搜索。

### 继续和停止规则

1. M1：val Dice 相对 M0 至少 +0.002，或 clDice 与 Boundary F1 同时改善且 Dice 不下降。
2. M2：val Boundary F1 或 clDice 至少 +0.005，Dice 不低于 M0 -0.001。
3. M3：同时超过 M0 的 val Dice +0.004，且至少一个结构指标正向，才补 M0/M3 seed43/44，并在 development-test 一次性评估。
4. M3 未超过 max(M1, M2)，或 Precision/HD95 明显退化，就停止 CGMA；不再加入长核、attention 或更多 loss。

这是一个 2 x 2 因子设计，可以直接回答先验、辅助任务及其互补性是否成立。

## 7 月 18--21 日安排

| 时间 | 工作 | 产出 |
|---|---|---|
| 7 月 18 日 | 实现 CGMA、GT boundary/centerline target、配置归档；完成随机 forward/backward 和 1 epoch smoke | 训练不暴露 test 的可复现代码 |
| 7 月 18--19 日晚 | 顺序运行 M0、M1 | 固定 contrast prior 的独立贡献 |
| 7 月 19--20 日晚 | 顺序运行 M2、M3 | 完整 2 x 2 因子矩阵 |
| 7 月 20--21 日 | 顺序运行 S0--S4 强增强复核 | 验证 prior/structure 是否在强协议下稳定有效 |
| 7 月 21 日 | 统一 val/development-test 评估；S0--S4 未达预设门槛，停止结构辅助线 | 不把 development-test 研发数字写成最终泛化结论 |

下一步应准备 nnU-Net v2 的 2D pipeline。它用于强 baseline 和投稿可信度建设，不是本论文创新；应以相同 patient split 或 outer-CV 单独运行。

## 代码改动与结果归档

本周已实现的代码服务于可证伪实验，而非为结果叠加复杂度：

| 文件 | 改动 | 核查状态 |
|---|---|---|
| `models/compact_green_morphology.py` | 新增 CGMA：`ReLU(GaussianBlur(G, sigma=9)-G)` 固定 local-contrast prior，经轻量 adapter 只在最终 decoder feature 进行一次 gated residual 注入；可选 boundary/centerline 训练头 | 随机 forward/backward、真实 1 epoch、权重 reload 通过 |
| `datasets/dataset_vessel.py` | `structure_targets=True` 时，在所有几何增强后生成 inner boundary 与 skeleton target；光度增强仅作用 image | 避免图像与结构 target 错位 |
| `losses/joint_loss.py` | 新增 `StructureAuxiliaryLoss`：主 BCE-Dice 加可选 boundary/centerline BCE-Dice | S0--S4 覆盖 auxiliary on/off |
| `train_unified.py` | 新增 `--mode cgma`、prior/auxiliary 参数；每次训练结束写 `val_per_image.csv`；默认不在逐 epoch 读取 test | checkpoint 仅由 val Dice 决定 |
| `evaluate_all.py` | 支持 `model_type=cgma`，统一输出 Dice、IoU、Recall、Precision、Specificity、Accuracy、HD95、clDice、Boundary F1 | val/dev 五组均成功加载评估 |
| `scripts/run_cgma_2x2_20260719.ps1` 与 `scripts/run_structure_aux_intensity_on_20260720.ps1` | 顺序训练、完成后才统一评估、支持 `-SkipExisting` 与自动决策汇总 | 两轮均完整跑通 |

| 文件/目录 | 内容 |
|---|---|
| results/cgma_2x2_20260719/run_summary.csv | 四组权重、最佳 val epoch 和 val Dice |
| results/cgma_2x2_20260719/metrics_summary.csv | M0 相对增益、val/development-test 全指标 |
| results/cgma_2x2_20260719/decision.json | 是否满足 M3 多 seed 条件 |
| results/cgma_2x2_20260719/logs | 每组训练、val、development-test 与汇总日志 |
| results/unified_eval_cgma_2x2_val_20260719 | val aggregate/per-image CSV |
| results/unified_eval_cgma_2x2_20260719 | development-test aggregate/per-image CSV |
| results/structure_aux_intensity_on_20260720/metrics_summary.csv | S0--S4 强增强复核全指标与相对 S0 delta |
| results/structure_aux_intensity_on_20260720/decision.json | 固定继续规则及停止结论 |
| results/unified_eval_structure_aux_intensity_on_val_20260720 | S0--S4 val aggregate/per-image CSV |
| results/unified_eval_structure_aux_intensity_on_20260720 | S0--S4 development-test aggregate/per-image CSV |

实现已通过随机前反向、真实 1 epoch、best checkpoint reload、val per-image CSV 和统一评估加载 smoke test。M0--M3 使用 intensity augmentation off；S0--S4 用 intensity augmentation on 进行正式反证。两轮均使用 seed42、scratch、BCE-Dice、50 epochs、patience20、batch4、lr 1e-4。

## 组会讲法

~~~
前一阶段确认了 green prior 的稳定价值，也确认了堆叠的边界：
五分支方向融合塌缩为一个短条带响应，深层融合没有正贡献；
双教师蒸馏可以小幅提升，但没有保留 ensemble 的完整边界收益。

因此我们做了减法式验证：用训练集审计过的固定 green 局部对比图替代 enhancer，
只在最终 decoder 注入一次，并将边界和中心线设为独立辅助任务。
2 x 2 首轮有 development-test 信号，但强增强 S0--S4 复核未在 val 上成立；这说明该紧凑结构目前不能作为主线。

所以我们按预设规则停止结构辅助线，转向 nnU-Net/MedNeXt 强 baseline、患者级 outer-CV 和数据标注上限分析，而不是继续叠加模块。
~~~

## SCI 2--3 区目标

现有 green MSE 三 seed、K2、C3 和 ensemble 都是积极线索，但不足以只凭内部 Dice 数字投稿。CGMA 两轮试验的价值在于建立了可解释的反证：局部 green contrast 和结构辅助在当前训练协议下没有形成可复现的单模型净增益。

SCI 2--3 区的后续门槛应改为：在 nnU-Net v2/MedNeXt 等强 baseline 上完成公平比较；确定最终模型后进行患者级 outer-CV 或未触碰 holdout；报告多 seed mean +/- std、患者级 paired CI、失败病例可视化和下游形态量化。只有方法相对强 baseline 同时具有稳定 Dice/结构指标收益，才具备可靠叙事基础；当前不把 S1/S2 的单次 development-test 改善作为投稿证据。
