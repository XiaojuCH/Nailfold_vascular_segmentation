# 2026-07-10 组会汇报草稿：四个创新点探针实验

更新日期：2026-07-10  
当前目标：围绕甲襞毛细血管形态学先验，验证 4 个可能带来更大提升的方向。

## 1. 本周核心问题

前期实验已经说明 green-channel prior 有正向信号，但不能只看指标绝对值，必须看相对同口径 baseline 的提升。下面以 `pretrained seed42` 为例，`ΔHD95` 为负数表示更好：

| 模型 | Dice | ΔDice vs TransUNet | ΔDice vs Ours green MSE | HD95 | ΔHD95 vs TransUNet | Boundary F1 | ΔBoundary F1 vs TransUNet | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TransUNet pretrained | 0.7567 | baseline | - | 22.69 | baseline | 0.6336 | baseline | 当前主要强 baseline |
| Ours green MSE pretrained | 0.7583 | +0.0016 | baseline | 22.05 | -0.64 | 0.6414 | +0.0078 | green prior 有稳定正信号，但 Dice 提升小 |
| Ours clDice+Boundary old | 0.7567 | +0.0000 | -0.0016 | 23.55 | +0.86 | 0.6519 | +0.0183 | 结构指标最好，但 Dice/HD95 不占优 |
| C3 anisotropic + no BN + no intensity aug pretrained | 0.7606 | +0.0039 | +0.0023 | 22.31 | -0.38 | 0.6463 | +0.0127 | 单次 Dice 最好，需多 seed 验证 |

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

组会可引用的起点文献：

| 文献 | 链接 | 用途 |
|---|---|---|
| Morphology-aware distillation | https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2026.1825518/full | 支持“血管分割蒸馏应关注 morphology/decoder 信息” |
| TransFuse | https://arxiv.org/abs/2102.08005 | 支持 CNN 局部特征与 Transformer 全局特征互补 |
| Contrast-enhanced KD reliability | https://www.sciencedirect.com/science/article/pii/S1361841525002245 | 支持“增强/对比度先验知识需要可靠性建模” |
| Vessel-aware structure preservation | https://openaccess.thecvf.com/content/WACV2026W/P2P/papers/Dong_VAOT_Vessel-Aware_Optimal_Transport_for_Retinal_Fundus_Enhancement_WACVW_2026_paper.pdf | 支持血管增强/结构保持比单纯像素一致更重要 |

## 3. 已实现的探针实验

所有实验先分成两套口径：

| 口径 | 含义 | 汇报定位 |
|---|---|---|
| `scratch` | 不加载 ImageNet21k 权重，从 0 开始训练 TransUNet | 更符合部分医学图像实验习惯，可能更能体现方法带来的相对增益 |
| `pretrained` | 加载 `R50+ViT-B_16.npz` ImageNet21k 预训练 | 更强 baseline 口径，适合证明方法在强初始化下仍有效 |

注意：两套口径不能混在同一个主表里直接比较。最终论文应选择一个主口径，另一个作为补充分析。

本次组会汇报口径：

```text
不要只报 Dice/HD95 的绝对值，而是优先报相对提升：
1. 新方法 vs 同口径 TransUNet baseline，说明相对强 baseline 是否有效。
2. 新方法 vs 旧 Ours green MSE，说明新增模块是否真的比原主线更值得继续。
3. HD95 越低越好，所以 ΔHD95 为负数代表改善。
```

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

### 3.1 本周代码改动

本周代码不是只改模型本体，还同步改了训练入口、评估入口和实验脚本，保证每个探针都能在同一数据集、同一阈值、同一指标口径下比较。

| 文件 | 具体改动 | 对应实验/用途 | 是否影响旧默认行为 |
|---|---|---|---|
| `models/joint_framework.py` | 新增 `_norm_layer(channels, norm_type)`，支持 `bn` 与 `none`；`Enhancer/MultiScaleEnhancer/AnisotropicEnhancer/DualFusion` 都可选择是否使用 BN | E4a、C2、C3 的“去 BN/强度先验保护”实验 | 默认仍为 `bn`，旧 `basic/v1` 行为不变 |
| `models/joint_framework.py` | 新增 `AnisotropicEnhancer`：`3x3 local branch + 1x7/7x1 strip branch + 1x21/21x1 strip branch`，最后 residual 输出增强图 | E1、C1、C3，用于甲襞细长血管的方向性形态增强 | 只有 `--enhancer anisotropic` 时启用 |
| `models/joint_framework.py` | 新增 `JointModel_DecoderDistill`：同一 TransUNet 对 enhanced RGB 与 green prior view 的 decoder features 做一致性 | E2 初版 decoder 蒸馏探针 | 只有 `--joint_model decoder_distill` 时启用 |
| `models/joint_framework.py` | 新增 `JointModel_DecoderDistillV2`：独立 frozen direct-green teacher，student 对 enhanced RGB，teacher 对 green prior，decoder feature 经 `1x1 projection` 后对齐 | 组会前最后一次更规范的 decoder 蒸馏 V2 | 只有 `--joint_model decoder_distill_v2` 时启用 |
| `models/joint_framework.py` | 新增 `JointModel_DualFusion`：浅层 CNN 走原始 RGB，TransUNet 走 enhanced image，最后 decoder feature 做 residual fusion | E3 CNN + TransUNet 双路融合探针 | 只有 `--joint_model dual_fusion` 时启用 |
| `datasets/dataset_vessel.py` | 增加 `intensity_aug` 开关；几何增强同步作用于 image/mask/teacher，亮度/对比度增强只作用于 image，且可关闭 | E4b、C1、C2、C3 的强度先验保护实验 | 默认 `intensity_aug=True`，旧训练默认不变 |
| `losses/joint_loss.py` | `JointDecoderDistillationLoss` 增加 `decoder_distill_mode`，支持 `mse/normalized_mse/cosine/cosine_mse`；支持 dict payload，兼容 V1/V2 feature 输入 | E2 与 V2 decoder 蒸馏 | 默认 `mse`，旧 E2 行为不变 |
| `train_unified.py` | 新增参数 `--enhancer anisotropic`、`--enhancer_norm bn/none`、`--joint_model decoder_distill/decoder_distill_v2/dual_fusion`、`--intensity_aug on/off` | 所有四个创新点探针 | 默认仍为 `basic + bn + v1 + intensity_aug on` |
| `train_unified.py` | 新增 `--decoder_teacher_weight`、`--decoder_teacher_pretrained`、`--decoder_distill_mode`；V2 会加载 independent teacher 并冻结；optimizer 只更新 `requires_grad=True` 参数 | decoder distill V2 | 只影响 V2，避免 teacher 被误更新 |
| `evaluate_all.py` | 统一评估支持 `anisotropic`、`enhancer_norm`、`decoder_distill_v2`、`dual_fusion`，并把 decoder 参数写入 aggregate CSV/XLSX | 统一复评所有新模型 | 旧模型评估不变 |
| `evaluate_all.py` | aggregate 输出保留 Dice、IoU、Recall、Precision、Specificity、Accuracy、HD95、clDice、Boundary F1，per-image CSV 保留逐图指标 | 后续 paired bootstrap/Wilcoxon 与失败案例筛选 | 指标口径统一 |
| `scripts/run_four_innovation_probes_20260710.ps1` | 串联 E1-E4b 的训练与统一评估，支持 `-PretrainMode scratch/pretrained/both` | 第一轮四方向探针 | 新脚本，不影响旧脚本 |
| `scripts/run_morph_intensity_combo_20260710.ps1` | 串联 C1-C4 组合实验，支持结构损失与阈值选择补充分析 | 形态学增强 + 强度先验保护组合实验 | 新脚本 |
| `scripts/run_scratch_delta_multiseed_20260710.ps1` | 固定 TransUNet、Ours green MSE、C3 三组，多 seed 跑 scratch/pretrained，并记录相对 delta | 多 seed 稳定性和相对提升分析 | 新脚本 |
| `scripts/run_decoder_distill_v2_20260710.ps1` | 训练 V2 后自动统一评估，并写入 `metrics_summary.csv` | 组会前 decoder 蒸馏 V2 收尾实验 | 新脚本 |

按创新点归纳如下：

| 模块/参数 | 改动 | 汇报定位 |
|---|---|---|
| `AnisotropicEnhancer` | 在 Enhancer 中加入 `1x7/7x1` 与 `1x21/21x1` strip convolution | 利用甲襞毛细血管细长、方向性强的形态先验 |
| `enhancer_norm=bn/none` | 支持关闭 Enhancer 内 BN | 验证归一化是否削弱 green-channel 强度/对比度先验 |
| `intensity_aug=on/off` | 训练时可关闭亮度/对比度 jitter | 验证强度增强是否干扰 green prior |
| `decoder_distill` | 对同一 TransUNet 在 enhanced image 与 green prior view 上的 decoder features 做一致性约束 | 当前作为失败探针，后续需改成更规范的 teacher-student 蒸馏 |
| `decoder_distill_v2` | 新增独立 frozen direct-green teacher、final decoder layer 蒸馏、projection 与 `cosine_mse` | 作为组会前最后一次“真正 teacher-student decoder distillation”探针 |
| `dual_fusion` | 浅层 CNN 走 RGB，TransUNet 走 enhanced image，最后 decoder feature 做 residual refinement | 当前末端融合效果不理想，提示需要更早层级/多尺度交互 |
| 统一评估与脚本 | `evaluate_all.py` 与实验脚本统一输出 Dice、IoU、Recall、Precision、Specificity、Accuracy、HD95、clDice、Boundary F1 | 保证所有模型同一测试集、阈值和指标口径 |

### 3.2 低级错误核查

| 检查项 | 结果 | 结论 |
|---|---|---|
| 数据集划分 | 训练、验证、测试仍使用 `dataset_all_filtered/train`、`val`、`test` | 没有把 test 当成全数据集重新划分 |
| 测试集口径 | unified eval 使用 `dataset_all_filtered/test`，测试图数为 436 | 与前期统一复评口径一致 |
| teacher prior 对齐 | `train/images` 与 `train/teacher_priors_green_only` 文件名排序一致 | 未发现 teacher 文件错位 |
| green-only teacher | green-only prior 是 3 通道复制的绿通道图 | 符合当前 enhancer 输出 3 通道的 MSE 约束设计 |
| 数据增强同步 | 翻转/旋转同步作用于 image、mask、teacher；亮度/对比度增强只作用于 image | 符合“teacher prior 不被强度增强破坏”的设计 |
| unified eval 日志 | 评估时会先构建模型再加载完整训练权重；日志里的“未加载预训练权重”只是构建阶段提示 | 不代表评估没有加载训练好的 `best_model.pth` |
| decoder distill | 未发现路径错误、shape mismatch 或 teacher 图像错位 | 低分更可能来自蒸馏设计本身，而不是低级工程错误 |
| decoder distill V2 smoke test | 随机输入 `[1,3,256,256]` 下 logits 为 `[1,1,256,256]`，student/teacher final decoder feature 均为 `[1,16,256,256]` | V2 训练前向、teacher 加载、feature 对齐和 loss 计算均已跑通 |

### 3.3 Decoder 蒸馏失败原因核查

当前 `decoder_distill` 的实现不是严格意义上的“固定强 teacher 指导 student”，而是同一 `segmentor` 对两个输入视图做 decoder feature consistency：

```text
student view: enhanced RGB image -> TransUNet decoder features
teacher view: green prior image -> 同一个 TransUNet，临时 eval + no_grad -> decoder features detach
loss: decoder layer 2,3 直接做 MSE，lambda_decoder_distill = 1.0
```

这解释了为什么 E2 可能变差：

| 问题 | 影响 |
|---|---|
| teacher 与 student 共享同一个 segmentor 权重 | 不是独立/frozen teacher，teacher-view feature 本身未必是更强监督 |
| student 分支为 train mode，teacher-view 临时 eval mode | BN/Dropout 行为不同，特征分布可能天然不一致 |
| 直接对 decoder layer `2,3` 做 MSE | 这两层较浅且高分辨率，可能约束低层纹理而不是稳定血管形态 |
| `lambda_decoder_distill=1.0` | 日志显示 decoder loss 在 scratch 版多次达到 `0.1~0.45`，相对 segmentation loss 不轻，可能扰乱优化 |
| green prior 不是人工标注或强 teacher logits | 不能默认 green-view feature 总是优于 enhanced RGB feature |

因此组会中建议这样表述：

```text
我们验证了 decoder feature consistency 的初版设计，但它并没有带来提升。
代码核查未发现路径或数据错位问题，失败更可能来自蒸馏目标设计：当前是同网络双视图 feature MSE，而不是独立 teacher-student 的 morphology-aware distillation。
下周如果继续做 decoder 蒸馏，应改为 frozen teacher、低权重、final layer/projection 对齐，或蒸馏结构图/logits，而不是裸 feature MSE。
```

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

绝对指标表：

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

相对提升表。`pretrained` 口径对照 `TransUNet pretrained seed42 = 0.7567` 与 `Ours green MSE pretrained seed42 = 0.7583`；`scratch` 口径对照 `TransUNet scratch seed42 = 0.7522` 与 `Ours green MSE scratch seed42 = 0.7571`。`ΔHD95` 为负数表示更好。

| 口径 | 实验 | ΔDice vs TransUNet | ΔDice vs Ours green MSE | ΔHD95 vs TransUNet | ΔHD95 vs Ours green MSE | ΔBoundary F1 vs TransUNet | ΔBoundary F1 vs Ours green MSE | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pretrained | E1 anisotropic enhancer | +0.0009 | -0.0007 | +0.44 | +1.08 | +0.0178 | +0.0100 | 边界收益明显，但 Dice/HD95 不如旧 Ours |
| pretrained | E2 decoder distill | -0.0065 | -0.0081 | +2.38 | +3.02 | -0.0076 | -0.0154 | 明确失败 |
| pretrained | E3 dual fusion | -0.0021 | -0.0037 | +0.69 | +1.33 | +0.0082 | +0.0004 | Dice 下降，不作为主线 |
| pretrained | E4a no enhancer BN | +0.0006 | -0.0010 | +0.45 | +1.09 | +0.0098 | +0.0020 | 弱信号 |
| pretrained | E4b no intensity aug | +0.0014 | -0.0002 | +0.12 | +0.76 | +0.0128 | +0.0050 | 支持强度先验消融 |
| scratch | E1 anisotropic enhancer | +0.0032 | -0.0017 | -0.80 | -0.20 | +0.0093 | -0.0011 | 相对 TransUNet 有提升，但不如旧 Ours |
| scratch | E2 decoder distill | +0.0026 | -0.0022 | -1.49 | -0.89 | +0.0000 | -0.0104 | HD95 有改善但 Dice/结构不够 |
| scratch | E3 dual fusion | +0.0024 | -0.0025 | -1.35 | -0.75 | +0.0116 | +0.0012 | 结构略好但 Dice 不够 |
| scratch | E4a no enhancer BN | +0.0055 | +0.0007 | -1.20 | -0.60 | +0.0018 | -0.0086 | scratch 单次弱阳性 |
| scratch | E4b no intensity aug | +0.0046 | -0.0003 | -0.21 | +0.39 | +0.0016 | -0.0088 | 相对 TransUNet 有提升，不如旧 Ours |

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

绝对指标表：

| 口径 | 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained | C1 anisotropic + no intensity aug | 0.7592 | 0.6227 | 0.7740 | 0.7651 | 22.40 | 0.8496 | 0.6542 | 结构与 Dice 均有提升 |
| pretrained | C2 no BN + no intensity aug | 0.7488 | 0.6114 | 0.8302 | 0.7003 | 24.70 | 0.8398 | 0.6190 | 暂停 |
| pretrained | C3 anisotropic + no BN + no intensity aug | 0.7606 | 0.6241 | 0.7872 | 0.7544 | 22.31 | 0.8494 | 0.6463 | 当前主线候选 |
| pretrained | C4 C1 + clDice/Boundary loss | 0.7559 | 0.6190 | 0.7991 | 0.7366 | 22.60 | 0.8542 | 0.6461 | 结构指标高，但 Dice 下降 |
| scratch | C1 anisotropic + no intensity aug | 0.7561 | 0.6179 | 0.8003 | 0.7347 | 24.26 | 0.8435 | 0.6451 | 暂停/观察 |
| scratch | C2 no BN + no intensity aug | 0.7573 | 0.6193 | 0.7720 | 0.7621 | 23.99 | 0.8449 | 0.6484 | 观察 |
| scratch | C3 anisotropic + no BN + no intensity aug | 0.7583 | 0.6204 | 0.7842 | 0.7517 | 22.11 | 0.8466 | 0.6481 | scratch 最好 |

相对提升表。`pretrained` 对照同 4.0；`scratch` 对照 `TransUNet scratch seed42 = 0.7522` 与 `Ours green MSE scratch seed42 = 0.7571`。`ΔHD95` 为负数表示更好。

| 口径 | 实验 | ΔDice vs TransUNet | ΔDice vs Ours green MSE | ΔHD95 vs TransUNet | ΔHD95 vs Ours green MSE | ΔclDice vs Ours green MSE | ΔBoundary F1 vs Ours green MSE | 解释 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pretrained | C1 anisotropic + no intensity aug | +0.0025 | +0.0009 | -0.29 | +0.35 | +0.0031 | +0.0128 | 边界收益最明显，适合作为形态学增强证据 |
| pretrained | C2 no BN + no intensity aug | -0.0079 | -0.0095 | +2.01 | +2.65 | -0.0067 | -0.0224 | 去 BN + 关增强在 basic 上失效 |
| pretrained | C3 anisotropic + no BN + no intensity aug | +0.0039 | +0.0023 | -0.38 | +0.26 | +0.0029 | +0.0049 | 单次当前最好，Precision 提升明显，Recall 有下降 |
| pretrained | C4 C1 + clDice/Boundary loss | -0.0008 | -0.0024 | -0.09 | +0.55 | +0.0077 | +0.0047 | clDice 高，但 Dice 不如旧 Ours |
| scratch | C1 anisotropic + no intensity aug | +0.0039 | -0.0010 | +0.15 | +0.75 | -0.0016 | -0.0015 | scratch 下不如旧 Ours |
| scratch | C2 no BN + no intensity aug | +0.0051 | +0.0002 | -0.12 | +0.48 | -0.0002 | +0.0018 | 去 BN + 关增强有弱阳性 |
| scratch | C3 anisotropic + no BN + no intensity aug | +0.0061 | +0.0012 | -2.00 | -1.40 | +0.0015 | +0.0015 | scratch seed42 同时改善 Dice 和 HD95 |

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

三 seed 平均相对提升：

| 对比 | ΔDice | ΔHD95 | ΔclDice | ΔBoundary F1 | 解释 |
|---|---:|---:|---:|---:|---|
| Ours green MSE scratch vs TransUNet scratch | +0.0066 | -0.22 | +0.0064 | +0.0095 | green prior 在 scratch 口径下提升最稳定 |
| C3 scratch vs TransUNet scratch | +0.0058 | -1.02 | +0.0058 | +0.0080 | C3 对 HD95/空间误差改善更明显 |
| C3 scratch vs Ours green MSE scratch | -0.0008 | -0.80 | -0.0006 | -0.0015 | C3 没稳定超过旧 Ours 的 Dice，但 HD95 更好 |

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

### 4.6 pretrained 多 seed 复现实验

周五凌晨已补完 pretrained seed 43/44，用于确认 C3 的 `0.7606` 是否稳定：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_scratch_delta_multiseed_20260710.ps1 -PretrainMode pretrained -Seeds "43,44" -SkipExisting
```

运行结果文件：

```text
results/scratch_delta_multiseed_20260710/metrics_summary.csv
results/unified_eval_scratch_delta_multiseed_20260710
```

逐 seed Dice 与相对提升：

| Seed | TransUNet pretrained | Ours green MSE pretrained | C3 pretrained | Ours - TransUNet | C3 - TransUNet | C3 - Ours |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7567 | 0.7583 | 0.7606 | +0.0017 | +0.0040 | +0.0023 |
| 43 | 0.7583 | 0.7549 | 0.7603 | -0.0034 | +0.0020 | +0.0054 |
| 44 | 0.7559 | 0.7602 | 0.7566 | +0.0043 | +0.0008 | -0.0035 |

三 seed 均值与标准差：

| 组别 | Dice mean ± std | HD95 mean ± std | clDice mean ± std | Boundary F1 mean ± std | 判断 |
|---|---:|---:|---:|---:|---|
| TransUNet pretrained | 0.7569 ± 0.0012 | 22.95 ± 0.24 | 0.8494 ± 0.0006 | 0.6369 ± 0.0030 | 强 baseline |
| Ours green MSE pretrained | 0.7578 ± 0.0027 | 22.04 ± 0.44 | 0.8481 ± 0.0042 | 0.6370 ± 0.0055 | HD95 最好，但 Dice 波动较大 |
| C3 pretrained | 0.7592 ± 0.0022 | 22.45 ± 0.14 | 0.8499 ± 0.0032 | 0.6455 ± 0.0078 | 平均 Dice 与 Boundary F1 最好，但 seed44 不稳定 |

三 seed 平均相对提升：

| 对比 | ΔDice | ΔHD95 | ΔclDice | ΔBoundary F1 | 解释 |
|---|---:|---:|---:|---:|---|
| Ours green MSE pretrained vs TransUNet pretrained | +0.0009 | -0.91 | -0.0013 | +0.0001 | 预训练强 baseline 下 Dice 收益被压小，但 HD95 最好 |
| C3 pretrained vs TransUNet pretrained | +0.0023 | -0.50 | +0.0005 | +0.0086 | C3 的 Dice 和 Boundary F1 平均最好 |
| C3 pretrained vs Ours green MSE pretrained | +0.0014 | +0.41 | +0.0018 | +0.0085 | C3 更偏边界/结构收益，HD95 不如旧 Ours |

阶段性结论：

```text
pretrained 三 seed 下，C3 的平均 Dice 最高：0.7592，高于 TransUNet pretrained 的 0.7569 和 Ours green MSE pretrained 的 0.7578。
C3 的 Boundary F1 也最高：0.6455，说明形态学增强 + 强度先验保护对边界/结构质量有正向信号。
但 C3 在 seed44 低于 Ours green MSE，说明它还不是非常稳定的最终主线，只能作为“当前最有希望的候选方向”。
```

组会中建议这样说：

```text
本周最稳定的结论是 green-channel prior 确实有效；C3 组合在 pretrained 三 seed 平均 Dice 和 Boundary F1 上最好，但提升仍只有约 +0.0023 vs TransUNet、+0.0014 vs old Ours。
因此当前结果足以支撑继续推进该方向，但还不足以单独支撑 SCI 2-3 区，需要统计显著性、可视化和进一步优化。
```

### 4.7 Decoder 蒸馏 V2 组会前补充实验

由于原版 E2 的问题更像是“同网络双视图裸 feature MSE”而不是真正 teacher-student distillation，组会前最后补一个更规范的 V2 探针：

| 项目 | V1 原版 | V2 新版 |
|---|---|---|
| Teacher | 与 student 共享同一个 TransUNet | 独立 frozen TransUNet direct-green teacher |
| Teacher 输入 | green prior view | green prior view |
| Teacher 权重 | 无独立 teacher 权重 | `direct_green_baseline_20260620/0620_0119/best_model.pth` |
| 蒸馏层 | decoder layer `2,3` | final decoder layer `3` |
| 蒸馏损失 | raw MSE | projection 后 `cosine_mse` |
| 权重 | `lambda=1.0` | `lambda=0.1` |

代码层面这次改动包括：

| 文件 | 改动 |
|---|---|
| `models/joint_framework.py` | 新增 `JointModel_DecoderDistillV2`，student 使用 enhanced RGB，teacher 使用 green prior，并冻结 teacher 参数 |
| `losses/joint_loss.py` | `JointDecoderDistillationLoss` 支持 `mse`、`normalized_mse`、`cosine`、`cosine_mse` 四种 decoder feature loss |
| `train_unified.py` | 新增 `--joint_model decoder_distill_v2`、`--decoder_teacher_weight`、`--decoder_distill_mode`，optimizer 只更新 `requires_grad=True` 参数 |
| `evaluate_all.py` | 支持加载和统一评估 `decoder_distill_v2` |
| `scripts/run_decoder_distill_v2_20260710.ps1` | 训练后自动用 `evaluate_all.py` 复评，并写入 `metrics_summary.csv` |

工程 smoke test 已完成：

```text
logits shape: [1, 1, 256, 256]
student final decoder feature: [1, 16, 256, 256]
teacher final decoder feature: [1, 16, 256, 256]
decoder loss mode: cosine_mse
lambda_decoder_distill: 0.1
```

运行脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_decoder_distill_v2_20260710.ps1
```

运行结果：

```text
Metrics: results\decoder_distill_v2_20260710\metrics_summary.csv
Eval:    results\unified_eval_decoder_distill_v2_20260710\20260710_101910\aggregate_results.csv
Weight:  results\experiments\all_filtered\decoder_distill_v2_direct_green_teacher_pretrained_seed42_20260710\0710_0812\best_model.pth
```

| 实验 | Dice | IoU | Recall | Precision | Specificity | Accuracy | HD95 | clDice | Boundary F1 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DecoderDistillV2 direct-green teacher pretrained seed42 | 0.7543 | 0.6162 | 0.8122 | 0.7220 | 0.9619 | 0.9465 | 23.44 | 0.8438 | 0.6290 | 仍未超过主线，暂停 |

与关键对照相比：

| 对照 | Dice 差值 | HD95 差值 | Boundary F1 差值 | 结论 |
|---|---:|---:|---:|---|
| vs TransUNet pretrained seed42 | -0.0024 | +0.75 | -0.0046 | 不如强 baseline |
| vs Ours green MSE pretrained seed42 | -0.0040 | +1.39 | -0.0124 | 不如当前 green prior 主线 |
| vs C3 pretrained seed42 | -0.0063 | +1.13 | -0.0173 | 不如当前 C3 候选 |

阶段性结论：

```text
V2 修正了原 E2 的主要工程/设计问题：独立 frozen teacher、只蒸馏 final decoder layer、低权重、projection + cosine_mse。
但结果仍为 Dice 0.7543，低于 TransUNet pretrained、Ours green MSE pretrained 和 C3 pretrained。
因此 decoder 蒸馏方向目前不建议继续作为组会后的主线；它可以作为“已系统排查但暂不成立”的负结果。
下周主线更应收敛到 morphology-aware enhancer / intensity-prior preservation，并补统计显著性、可视化和强 baseline。
```

## 5. 组会汇报建议

关于预训练口径可以这样说：

```text
ImageNet21k 预训练会显著增强 TransUNet baseline，因此 Ours 的相对提升会被压小。
但医学图像任务是否必须用自然图像预训练并没有唯一标准，所以本轮同时保留 scratch 和 pretrained 两套口径。
如果 scratch 口径下方法提升更明显，论文主表可以采用从零训练设置；
pretrained 结果作为强 baseline 补充，说明方法在强初始化下是否仍有收益。
```

如果汇报时需要解释 E2/E3：

```text
本周我们没有只凭直觉放弃 decoder 蒸馏，而是先做了原版 E2，再做了更规范的 V2。
V2 已经改成独立 frozen teacher、final layer、低权重和 projection/cosine_mse，但仍未超过 baseline。
这说明当前 direct-green teacher 的 decoder feature 对 RGB/enhanced student 可能不是可靠知识源，至少在现有数据和 TransUNet 架构下不适合作为主线。
双路融合 E3 也没有提升，提示“复杂结构融合”不能简单照搬，当前更可信的信号仍来自形态学增强和强度先验保护。
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

1. 组会中明确：decoder distill V2 已跑完但仍无提升，因此暂停 decoder 蒸馏方向。
2. 对 C3、Ours green MSE、TransUNet 做 paired bootstrap / Wilcoxon，优先报告 Dice、HD95、clDice、Boundary F1 的 paired delta。
3. 做 C1/C3 的可视化案例，重点看细长血管末端、边界断裂、低对比区域和误检减少。
4. 组会中暂不把 C3 定为最终主线；表述为“pretrained 三 seed 平均最好，但提升仍小且 seed44 不稳定”。
5. 组会后应转向 C3 稳定性优化、强 baseline、数据上限分析、阈值策略和失败案例分层。
