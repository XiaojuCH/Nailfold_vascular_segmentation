# 2026-07-10 组会汇报草稿：四个创新点探针实验

更新日期：2026-07-09  
当前目标：围绕甲襞毛细血管形态学先验，验证 4 个可能带来更大提升的方向。

## 1. 本周核心问题

前期实验已经说明 green-channel prior 有正向信号，但相对强 baseline 的 Dice 增益仍偏小：

| 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TransUNet pretrained | 0.7567 | 0.6188 | 0.8016 | 0.7331 | 22.69 | 0.8489 | 0.6336 | 当前主要强 baseline |
| Ours green MSE pretrained | 0.7583 | 0.6208 | 0.8031 | 0.7347 | 22.05 | 0.8465 | 0.6414 | 当前 Dice/HD95 最好 |
| Ours clDice+Boundary old | 0.7567 | 0.6183 | 0.7928 | 0.7421 | 23.55 | 0.8533 | 0.6519 | 当前结构指标最好 |
| C3 anisotropic + no BN + no intensity aug pretrained | 0.7606 | 0.6241 | 0.7872 | 0.7544 | 22.31 | 0.8494 | 0.6463 | 单次新 Dice 最好，需多 seed 验证 |

因此本周不再只微调 lambda，而是围绕 4 个更有方法贡献的方向做探针实验：

1. 各向异性卷积核：利用甲襞毛细血管细长、方向性强的形态学特点。
2. Decoder 蒸馏：把蒸馏位置从图像级 prior 对齐推进到解码器特征层。
3. CNN + TransUNet 双路融合：增加 CNN 局部纹理分支，避免同模型双视图特征多样性不足。
4. 强度先验/归一化消融：验证 BN 和亮度对比度增强是否削弱 green-channel 对比度先验。

## 2. 文献依据

| 方向 | 参考文献 | 对我们的启发 |
|---|---|---|
| Decoder 蒸馏 | Morphology-aware distillation for lightweight retinal vessel segmentation across fundus photography and OCT angiography | 普通 KD 容易忽略血管拓扑结构，decoder-oriented morphology information 更适合血管类分割。 |
| 双路融合 | TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation | CNN 保留局部细节，Transformer 建模全局上下文；二者融合适合细小结构分割。 |
| 各向异性卷积 | 视网膜血管/管状结构分割中的 strip/directional convolution 思路 | 细长血管比普通 3x3 卷积更需要方向敏感感受野。 |
| 强度先验保护 | 多模态医学图像中 normalization 可能削弱绝对/相对强度先验的经验 | 当前没有 InstanceNorm，但 Enhancer 的 BN 和亮度 jitter 仍可能影响 green contrast prior。 |

## 3. 已实现的探针实验

所有实验先分成两套口径：

| 口径 | 含义 | 汇报定位 |
|---|---|---|
| `scratch` | 不加载 ImageNet21k 权重，从 0 开始训练 TransUNet | 更符合部分医学图像实验习惯，可能更能体现方法带来的相对增益 |
| `pretrained` | 加载 `R50+ViT-B_16.npz` ImageNet21k 预训练 | 更强 baseline 口径，适合证明方法在强初始化下仍有效 |

注意：两套口径不能混在同一个主表里直接比较。最终论文应选择一个主口径，另一个作为补充分析。

共同设置：

```text
dataset = dataset_all_filtered
seed = 42
epochs = 50
patience = 20
threshold = 0.5
seg_loss = BCE-Dice
teacher_mode = green_only
lambda_mse = 10
lambda_grad = 0
```

| 编号 | 实验名 | 方法变化 | 预期观察 |
|---|---|---|---|
| E1 | anisotropic_enhancer_pretrained | Enhancer 加入 `1x7/7x1` 与 `1x21/21x1` strip convolution | 是否提升细长血管连续性和 clDice |
| E2 | decoder_distill_pretrained | 对 decoder 第 2、3 层做 feature consistency | 是否比图像级 MSE 更能提升 Dice/HD95 |
| E3 | dual_fusion_pretrained | RGB 走浅层 CNN，enhanced image 走 TransUNet，decoder final feature residual fusion | 是否获得局部纹理和全局上下文互补 |
| E4a | no_enhancer_bn_pretrained | 去除 Enhancer 内 BN | 判断 BN 是否削弱强度/对比度先验 |
| E4b | no_intensity_aug_pretrained | 关闭亮度/对比度增强 | 判断 intensity augmentation 是否干扰 green prior |

## 4. 实验结果表

运行脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_four_innovation_probes_20260710.ps1
```

默认只跑 `pretrained` 口径。如果要按导师建议先看从 0 训练：

```powershell
.\scripts\run_four_innovation_probes_20260710.ps1 -PretrainMode scratch
```

如果时间充足，同时跑 scratch 和 pretrained 两套：

```powershell
.\scripts\run_four_innovation_probes_20260710.ps1 -PretrainMode both
```

结果汇总文件：

```text
results/four_innovation_probes_20260710/metrics_summary.csv
results/four_innovation_probes_20260710/metrics_summary_complete.csv
```

当前最终结果（截至 2026-07-07，5 个探针的 `pretrained` 与 `scratch` 口径均已完成）：

| 口径 | 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained | E1 anisotropic enhancer | 0.7576 | 0.6208 | 0.7747 | 0.7601 | 23.13 | 0.8476 | 0.6514 | 结构指标可继续 |
| pretrained | E2 decoder distill | 0.7502 | 0.6119 | 0.8164 | 0.7121 | 25.07 | 0.8399 | 0.6260 | 暂停当前版本 |
| pretrained | E3 dual fusion | 0.7546 | 0.6167 | 0.8069 | 0.7274 | 23.38 | 0.8443 | 0.6418 | 暂停当前版本 |
| pretrained | E4a no enhancer BN | 0.7573 | 0.6204 | 0.7884 | 0.7472 | 23.14 | 0.8469 | 0.6434 | 观察 |
| pretrained | E4b no intensity aug | 0.7581 | 0.6210 | 0.7992 | 0.7384 | 22.81 | 0.8488 | 0.6464 | 结构指标可继续 |
| scratch | E1 anisotropic enhancer | 0.7554 | 0.6179 | 0.7852 | 0.7465 | 23.31 | 0.8421 | 0.6454 | 暂停当前版本 |
| scratch | E2 decoder distill | 0.7549 | 0.6163 | 0.7902 | 0.7424 | 22.62 | 0.8412 | 0.6361 | 暂停当前版本 |
| scratch | E3 dual fusion | 0.7546 | 0.6166 | 0.7630 | 0.7651 | 22.76 | 0.8428 | 0.6477 | 暂停当前版本 |
| scratch | E4a no enhancer BN | 0.7578 | 0.6197 | 0.8049 | 0.7333 | 22.91 | 0.8438 | 0.6379 | 观察 |
| scratch | E4b no intensity aug | 0.7568 | 0.6190 | 0.7991 | 0.7354 | 23.90 | 0.8461 | 0.6377 | 观察 |

判断标准：

| 等级 | 标准 | 下一步 |
|---|---|---|
| 强阳性 | Dice >= 0.7625 且 HD95 <= 22.69 | 进入多 seed，并作为下一轮主线 |
| 可继续 | Dice >= 0.7600，或 clDice/Boundary F1 明显提升且 Dice 不低于 0.7567 | 做消融和可视化 |
| 暂停 | Dice < 0.7567 或 Precision 明显下降 | 暂停该方向 |

### 4.1 四个探针初筛结果分析

本轮 10 个初筛探针没有出现强阳性，即没有达到 `Dice >= 0.7625` 且 `HD95 <= 22.69`。结果更像是揭示了三个方向的取舍：

| 候选 | 对照 | Dice 变化 | 主要收益 | 主要问题 | 判断 |
|---|---|---:|---|---|---|
| E4b no intensity aug pretrained | Ours green MSE pretrained 0.7583 | -0.0003 | IoU/clDice/Boundary F1 更平衡，Boundary F1 +0.0050 | HD95 22.81 差于 Ours 22.05 | 保留为强度先验消融证据 |
| E1 anisotropic pretrained | Ours green MSE pretrained 0.7583 | -0.0007 | Boundary F1 +0.0100，Precision 明显更高 | Recall 明显下降，HD95 变差 | 保留为边界/形态增强候选 |
| E4a no BN scratch | Ours green MSE old 0.7571 | +0.0006 | Recall 和 HD95 略好 | Precision、clDice、Boundary F1 下降 | 弱阳性，不能单独做主线 |

初筛 pretrained 口径下，最好 Dice 仍然是 `Ours green MSE pretrained` 的 `0.7583`。新探针中最接近的是 E4b no intensity aug pretrained，Dice `0.7581`，但没有超过当时最好；它的价值在于支持“亮度/对比度增强可能干扰 green prior”的叙事。E1 anisotropic pretrained 的 Boundary F1 `0.6514` 很接近旧结构指标最好值 `0.6519`，说明各向异性卷积确实对边界或细长结构有帮助，但它牺牲了 Recall，因此不是单独 Dice 突破点。

scratch 口径下，E4a no enhancer BN 的 Dice `0.7578` 是本轮 scratch 最高，也略高于旧 Ours green MSE old `0.7571`。但这不是全面提升：它提高了 Recall，HD95 也略好，但 Precision、clDice 和 Boundary F1 都下降。因此可以作为“BN/归一化可能影响强度先验”的弱证据，不建议直接作为论文主模型。

E2 decoder distill 当前版本整体失败：pretrained Dice `0.7502`，scratch Dice `0.7549`。这说明当前 decoder feature consistency 可能过强，或 teacher-view feature 目标并不合适。后续如果继续做，应改成更低 `lambda_decoder_distill=0.1/0.3`、只蒸馏最后一层，或加入 projection/attention 后再对齐。

E3 dual fusion 当前版本也不理想：pretrained Dice `0.7546`，scratch Dice `0.7546`。说明只在 decoder final feature 做 residual fusion 太晚，CNN 局部特征没有充分参与分割过程。如果继续双路融合，应改成更早层级或多尺度交互，而不是只做末端 residual refinement。

### 4.2 对论文叙事的影响

本轮结果还不足以支撑“显著性能提升”的主叙事。更稳妥的组会说法是：

```text
四个创新点探针没有带来明显 Dice 突破，但给出了两个有价值信号：
第一，各向异性卷积和关闭强度增强都更有利于 Boundary F1，说明形态学/强度先验方向是合理的；
第二，当前 decoder 蒸馏和末端双路融合设计不合适，不能简单把文献模块移植到本任务。
下一轮应该把方向收敛到“形态学增强 + 强度先验保护”的组合实验，而不是继续堆复杂融合模块。
```

建议下一轮优先组合两个有信号的因素：

| 优先级 | 实验 | 目的 |
|---|---|---|
| P0 | `anisotropic + intensity_aug=off` | 检查 E1 的边界收益和 E4b 的强度先验收益能否叠加 |
| P0 | `basic + enhancer_norm=none + intensity_aug=off` | 检查去 BN 与关 intensity aug 是否在 scratch 口径下继续提升 |
| P1 | `anisotropic + enhancer_norm=none + intensity_aug=off` | 激进组合，验证形态学增强和强度先验保护的上限 |
| P1 | E1/E4b 做 val 阈值选择 | E1 Precision 高、Recall 低，可能通过阈值调整找回 Dice |
| P2 | decoder distill lambda 0.1/0.3、final layer only | 只作为后续备选，不作为当前主线 |

### 4.3 周二晚组合实验安排

周二晚不再继续铺新的大模块，而是验证本轮已经出现正信号的两个因素能否叠加：`anisotropic enhancer` 代表形态学增强，`intensity_aug=off` 与 `enhancer_norm=none` 代表强度先验保护。

运行脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_morph_intensity_combo_20260710.ps1 -PretrainMode both
```

默认会跑 3 个组合实验，每个都跑 `pretrained` 和 `scratch` 两套口径：

| 编号 | 实验 | 口径 | 目的 |
|---|---|---|---|
| C1 | `anisotropic + intensity_aug=off` | pretrained/scratch | 检查 E1 的边界收益和 E4b 的强度先验收益能否叠加 |
| C2 | `basic + enhancer_norm=none + intensity_aug=off` | pretrained/scratch | 检查去 BN 与关 intensity aug 是否能稳定保护 green contrast prior |
| C3 | `anisotropic + enhancer_norm=none + intensity_aug=off` | pretrained/scratch | 激进组合，验证形态学增强和强度先验保护的上限 |

如果时间充足，可以额外加入结构损失分支：

```powershell
.\scripts\run_morph_intensity_combo_20260710.ps1 -PretrainMode both -IncludeStructureLoss
```

如果想同时做 val 阈值选择补充分析：

```powershell
.\scripts\run_morph_intensity_combo_20260710.ps1 -PretrainMode both -RunThresholdSelection
```

阈值选择只能作为补充分析，论文主表仍建议使用统一 `threshold=0.5`。成功标准：

| 等级 | 标准 | 解释 |
|---|---|---|
| 强阳性 | Dice >= 0.7625 且 HD95 <= 22.69 | 可以进入多 seed，作为下一轮主线 |
| 主线候选 | Dice > 0.7583，或 Dice 接近 0.7583 且 Boundary F1 >= 0.6510 | 说明组合策略超过或接近当前最好，并保留结构优势 |
| 结构候选 | Boundary F1/clDice 明显提升，但 Dice 不低于 0.7567 | 可以作为结构质量支线，不单独做主表最佳模型 |
| 暂停 | Dice < 0.7567 或 Precision 明显下降 | 不继续扩展 |

### 4.4 周三组合实验结果

周二晚/周三已完成 C1-C3 的 pretrained 与 scratch 口径，以及 C4 的 pretrained 口径。C4 scratch 训练被手动中断，当前不纳入结果比较。

| 口径 | 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained | C1 anisotropic + no intensity aug | 0.7592 | 0.6227 | 0.7740 | 0.7651 | 22.40 | 0.8496 | 0.6542 | 结构与 Dice 均有提升 |
| pretrained | C2 no BN + no intensity aug | 0.7488 | 0.6114 | 0.8302 | 0.7003 | 24.70 | 0.8398 | 0.6190 | 暂停 |
| pretrained | C3 anisotropic + no BN + no intensity aug | 0.7606 | 0.6241 | 0.7872 | 0.7544 | 22.31 | 0.8494 | 0.6463 | 当前主线候选 |
| pretrained | C4 C1 + clDice/Boundary loss | 0.7559 | 0.6190 | 0.7991 | 0.7366 | 22.60 | 0.8542 | 0.6461 | 结构指标高，但 Dice 下降 |
| scratch | C1 anisotropic + no intensity aug | 0.7561 | 0.6179 | 0.8003 | 0.7347 | 24.26 | 0.8435 | 0.6451 | 暂停/观察 |
| scratch | C2 no BN + no intensity aug | 0.7573 | 0.6193 | 0.7720 | 0.7621 | 23.99 | 0.8449 | 0.6484 | 观察 |
| scratch | C3 anisotropic + no BN + no intensity aug | 0.7583 | 0.6204 | 0.7842 | 0.7517 | 22.11 | 0.8466 | 0.6481 | scratch 最好 |

与前期最好模型相比：

| 对比 | Dice 变化 | clDice 变化 | Boundary F1 变化 | 解释 |
|---|---:|---:|---:|---|
| C3 pretrained vs Ours green MSE pretrained | +0.0023 | +0.0029 | +0.0048 | 当前最有希望的主线，Precision 提升明显，Recall 有下降 |
| C1 pretrained vs Ours green MSE pretrained | +0.0009 | +0.0032 | +0.0128 | 边界收益最明显，适合作为形态学增强证据 |
| C3 scratch vs Ours green MSE old | +0.0011 | +0.0015 | +0.0003 | scratch 下也有轻微正向，但没有 pretrained 明显 |
| C4 pretrained vs Ours clDice+Boundary old | -0.0008 | +0.0009 | -0.0058 | clDice 更高，但 Boundary F1 和 Dice 不如旧结构损失分支 |

阶段性结论：

```text
组合实验第一次把 Dice 推到 0.7606，说明“各向异性形态学增强 + 强度先验保护”比单独模块更有效。
其中 C3 pretrained 可作为当前主线候选；C1 pretrained 可作为边界/结构质量证据。
结构损失 C4 没有带来 Dice 提升，暂时不建议作为主模型，只可作为 clDice 指标补充。
```

后续最紧急的是对 C3 pretrained 做统计和复现验证：

1. 对 C3 pretrained、Ours green MSE pretrained、TransUNet pretrained 做 paired bootstrap / Wilcoxon。
2. 补 C3 pretrained 的 seed 43/44；如果多 seed 仍稳定高于 0.7583，才适合作为论文主线。
3. 做 C1/C3 的可视化案例，重点展示边界更平滑、细长血管更完整、误检减少。
4. 暂时不必补 C4 scratch，除非组会需要完整表；它优先级低于 C3 多 seed。

### 4.5 scratch 多 seed 复现实验

如果从“提升程度”而不是“单次最高指标”来讲，scratch 口径更值得重点观察。已补充 seed 43/44 的同口径对照，并与 seed 42 合并统计：

运行结果文件：

```text
results/scratch_delta_multiseed_20260710/metrics_summary.csv
results/unified_eval_scratch_delta_multiseed_20260710
```

逐 seed Dice 与相对提升：

| Seed | TransUNet scratch | Ours green MSE scratch | C3 scratch | Ours - TransUNet | C3 - TransUNet | C3 - Ours |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7522 | 0.7571 | 0.7583 | +0.0050 | +0.0061 | +0.0011 |
| 43 | 0.7524 | 0.7588 | 0.7549 | +0.0064 | +0.0025 | -0.0039 |
| 44 | 0.7507 | 0.7593 | 0.7597 | +0.0086 | +0.0089 | +0.0004 |

三 seed 均值与标准差：

| 组别 | Dice mean ± std | HD95 mean ± std | clDice mean ± std | Boundary F1 mean ± std | 判断 |
|---|---:|---:|---:|---:|---|
| TransUNet scratch | 0.7518 ± 0.0009 | 23.64 ± 0.44 | 0.8403 ± 0.0005 | 0.6373 ± 0.0034 | scratch baseline |
| Ours green MSE scratch | 0.7584 ± 0.0011 | 23.42 ± 0.14 | 0.8467 ± 0.0014 | 0.6468 ± 0.0041 | 当前 scratch 口径最稳 |
| C3 anisotropic + no BN + no intensity aug scratch | 0.7576 ± 0.0025 | 22.62 ± 0.53 | 0.8461 ± 0.0021 | 0.6453 ± 0.0027 | HD95 更好，但 Dice 不稳定 |

阶段性结论：

```text
scratch 多 seed 结果支持“green-channel prior 稳定有效”，但不支持“C3 在 scratch 口径下稳定超过旧 Ours”。
旧 Ours green MSE 的平均 Dice 为 0.7584，高于 C3 的 0.7576，且方差更小。
C3 的优势主要体现在 HD95：三 seed 平均 HD95 为 22.62，明显好于旧 Ours 的 23.42，说明各向异性增强和强度先验保护可能改善边界/空间误差，但 Dice 收益还不稳定。
因此组会中应把 C3 表述为“pretrained 单次最好、scratch 下结构误差有改善、但额外 Dice 提升需继续验证”，而不是已经确定的新主线。
```

与 pretrained 口径放在一起看：

| 口径 | TransUNet | Ours green MSE | C3 anisotropic + no BN + no intensity aug | 主要解释 |
|---|---:|---:|---:|---|
| scratch mean, seed 42/43/44 | 0.7518 | 0.7584 | 0.7576 | 旧 Ours 提升更稳定，C3 的 Dice 波动较大 |
| pretrained seed 42 | 0.7567 | 0.7583 | 0.7606 | C3 单次最高，但还缺 pretrained 多 seed |

当前最合理的组会说法：

```text
从零训练口径下，green prior 相对 TransUNet 的提升约 +0.0066 Dice，比较稳定；
新 C3 组合相对 TransUNet 仍有 +0.0058 Dice，但没有稳定超过旧 Ours。
不过 C3 在 pretrained seed42 下达到当前最高 Dice 0.7606，并且 scratch 下 HD95 更好，因此它仍是值得继续验证的候选方向。
下一步优先补 pretrained 多 seed，而不是继续堆新模块。
```

## 5. 组会汇报建议

关于预训练口径可以这样说：

```text
ImageNet21k 预训练会显著增强 TransUNet baseline，因此 Ours 的相对提升会被压小。
但医学图像任务是否必须用自然图像预训练并没有唯一标准，所以本轮同时保留 scratch 和 pretrained 两套口径。
如果 scratch 口径下方法提升更明显，论文主表可以采用从零训练设置；
pretrained 结果作为强 baseline 补充，说明方法在强初始化下是否仍有收益。
```

如果 E2 或 E3 有明显提升：

```text
下一轮主线建议从 image-level green prior distillation 转向 feature-level morphology prior learning。
这比单纯 MSE 对齐更接近血管拓扑结构建模，也更符合 Morphology-aware distillation 和 TransFuse 的文献逻辑。
```

如果 E1 或 E4 有提升，但 E2/E3 没有提升：

```text
说明提升可能主要来自 morphology-aware enhancer 或 intensity-prior preservation。
论文叙事可以从“复杂融合结构”转向“面向甲襞细长血管的形态学增强与强度先验保护”。
```

如果 5 个探针都没有明显提升：

```text
当前 TransUNet pretrained 可能已经接近本数据集单模型上限。
下一步应优先补 nnU-Net/MedNeXt 等强 baseline，以及多 seed、统计显著性和失败案例分析。
```

## 6. 下一步待办

1. 补 pretrained 多 seed：`TransUNet pretrained`、`Ours green MSE pretrained`、`C3 pretrained` 均补 seed 43/44，用同 seed 比较确认 C3 的 0.7606 是否可复现。
2. 对 C3、Ours green MSE、TransUNet 做 paired bootstrap / Wilcoxon，优先报告 Dice、HD95、clDice、Boundary F1 的 paired delta。
3. 做 C1/C3 的可视化案例，重点看细长血管末端、边界断裂、低对比区域和误检减少。
4. 组会中暂不把 C3 定为最终主线；表述为“pretrained 单次最好、scratch 下 HD95 更好，但 Dice 稳定性仍需验证”。
5. 如果 pretrained 多 seed 也不稳定，组会后应转向强 baseline、数据上限分析、阈值策略和失败案例分层。
