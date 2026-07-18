# 2026-07-17 组会准备：代码审计、结果边界与下一轮主线

## 0. 本文档覆盖范围：两周成果，不只是 7 月 14 日审计

本文用于 7 月 17 日组会，覆盖 7 月 3 日组会之后到 7 月 14 日下午的两周工作。它包含 7 月 10 日组会文档中的四个创新点探针、形态/强度组合实验、scratch 与 pretrained 多 seed、decoder distillation V2，以及 7 月 14 日的代码与数据审计。

| 时间 | 已完成工作 | 对组会应呈现的结论 |
|---|---|---|
| 7 月 6-7 日 | E1-E4 四个创新点探针：各向异性 enhancer、decoder distillation、late dual fusion、BN/强度增强消融 | 形态和强度先验有弱阳性；当前 decoder 蒸馏与末端融合为负结果 |
| 7 月 8 日 | C1-C4 形态/强度组合和结构损失探针 | C3 单次 Dice 最好，C1 边界指标最好；结构 loss 未带来稳定 Dice 收益 |
| 7 月 9-10 日 | scratch 与 pretrained 各 3 seeds；decoder distillation V2 | green prior 的 scratch 增益最稳定；C3 仅是结构方向候选，不是最终赢家；V2 仍失败 |
| 7 月 14 日 | 数据划分、teacher 对齐、训练/评估代码审计 | 无 train/test 样本泄漏；但训练选模与增强实现需要在下一轮修正 |

### 0.1 7 月 10 日四个创新点探针完整记录

对照：pretrained `TransUNet=0.7567`，scratch `TransUNet=0.7522`。以下为 seed42 的统一 test 复评；仅用于呈现探索轨迹，不作为最终统计结论。

| 探针 | pretrained Dice | 相对 pretrained baseline | scratch Dice | 相对 scratch baseline | 本轮判断 |
|---|---:|---:|---:|---:|---|
| E1 当前 anisotropic enhancer | 0.7576 | +0.0009 | 0.7554 | +0.0033 | 有弱结构信号，但实现不是实际方向卷积 |
| E2 decoder feature consistency | 0.7502 | -0.0065 | 0.7549 | +0.0027 | 暂停 |
| E3 final-layer CNN residual fusion | 0.7546 | -0.0021 | 0.7546 | +0.0024 | 暂停；不能代表真正多尺度双路融合 |
| E4a basic enhancer, no BN | 0.7573 | +0.0006 | 0.7578 | +0.0056 | 仅弱信号 |
| E4b basic enhancer, no intensity aug | 0.7581 | +0.0014 | 0.7568 | +0.0047 | 有信号，但需修正增强实现后复验 |

### 0.2 7 月 10 日组合实验完整记录

| 组合 | pretrained Dice | pretrained Boundary F1 | scratch Dice | scratch HD95 | 本轮判断 |
|---|---:|---:|---:|---:|---|
| C1：当前 anisotropic + no intensity aug | 0.7592 | 0.6542 | 0.7561 | 24.26 | seed42 的边界信号最好，值得保留为形态学线索 |
| C2：basic + no BN + no intensity aug | 0.7488 | 0.6190 | 0.7573 | 23.99 | pretrained 明显失败，排除“简单去 BN”作为主线 |
| C3：当前 anisotropic + no BN + no intensity aug | 0.7606 | 0.6463 | 0.7583 | 22.11 | 单次 Dice 最高；后续三 seed 显示 Dice 不够稳定 |
| C4：C1 + soft clDice/Boundary loss | 0.7559 | 0.6461 | 未完成 | 未完成 | clDice 提高但 Dice 下降，停止继续调主损失 |

7 月 10 日的完整绝对指标、相对提升、工程实现说明保留在 `docs/group_meeting_summary_20260710.md`；本文将其压缩为两周汇报所需的结论，并补入多 seed 和审计后的修正口径。

## 1. 本周结论先行

当前项目不是“所有实验都跑错了”，但也还不能把 C3 或 decoder distillation 当作已成立的 SCI 主创新。

可以保留的最可靠结论是：green-channel prior 在从头训练下对 TransUNet 有稳定正增益；在 ImageNet21k 预训练下，Dice 增益被明显压缩，但仍可观察到 HD95、Boundary F1 等结构指标收益。当前提升幅度不足以仅凭单一内部测试集结果支撑 SCI 2-3 区，需要先修复训练选择口径、冻结测试集，并把“绿色先验”从图像级 MSE 约束升级成可解释的特征级方向/形态融合。

下一轮不建议继续重跑原 decoder distillation，也不建议继续做大量 lambda 或损失函数网格搜索。推荐主线为：

```text
RGB 主分支 + green-prior 轻量分支
    -> 平行的水平/垂直多尺度方向卷积
    -> 在 decoder 多尺度处进行门控融合
    -> 保留 RGB 的全局语义，同时用 green 分支补充局部细长结构
```

## 10. 2026-07-15 的评估协议修订与今晚实验

### 10.1 旧结果的正确解释

旧版 `train_unified.py` 在每个 epoch 都使用 **val Dice** 保存 `best_model.pth`，并不是使用 test 选 checkpoint；但训练结束后会自动输出一次 test 指标。由于随后多轮结构、loss 和预处理选择都参考过这批 test 结果，当前 `dataset_all_filtered/test` 在本项目中应诚实地称为 **development test**，而不是未参与研发的 final test。

这不否定旧结果的探索价值，也不意味着不能报告 test。它意味着：从现在开始，每一批预先固定的候选模型完成后再统一评估一次 development test；最终论文主结果需要补患者级外层 5-fold CV，或保留一份从未用于模型选择的 final holdout。

### 10.2 修正后的 F0 结果（只用 val 选权重）

| 方法 | 训练模式 | seed | best val Dice | best val epoch | HD95 at best Dice | 备注 |
|---|---|---:|---:|---:|---:|---|
| F0 corrected TransUNet | scratch, BCE-Dice, intensity aug on | 42 | 0.7897 | 20 | 19.98 | 逐图 val 汇总；无自动 test |
| 旧 green image-MSE v1 | scratch, MSE10/Grad0 | 42 | 0.7900 | 24 | 20.72 | 相对 F0 Dice `+0.0003`，HD95 `+0.74`；不再作为新 feature-fusion 主线 |

因此，当前最严谨的结论是：修正训练/验证统计口径后，旧图像级 green-MSE 在 seed42 上尚未显示足够的 val 改善。它不能被混称为下面的 feature-level green prior 消融。

### 10.3 新增的真实方向先验融合实现

新增 `models/green_prior_fusion.py`，不再使用 enhancer、teacher PNG 或 image-level MSE。RGB 仍输入 TransUNet 主分支；模型内部从原始 RGB 的 green channel (`x[:, 1:2]`) 提取 prior feature，并在 decoder 中以残差门控方式注入。所有 fusion residual 的 `alpha` 初始化为 0，因此初始化时严格退化为原 TransUNet，训练后才由数据决定是否使用 prior。

| 编号 | 代码配置 | 唯一新增变量 | 作用 |
|---|---|---|---|
| F1 | `plain_single` | 普通单尺度 green branch | 检验 green feature 本身是否有效 |
| F2 | `directional_single` | F1 + 并行 `3x3/1x7/7x1/1x21/21x1` depthwise strip branches + softmax direction gate | 检验真实水平/垂直、短/长尺度方向建模 |
| F3 | `directional_multiscale` | F2 + decoder 64/128/256 三尺度 gated fusion | 检验多尺度注入是否优于只在最终尺度融合 |

注意：旧 C1/C3 的 `1x7 -> 7x1` 串联实现是可分解二维大核，不等价于 F2 的并行方向分支；二者不能混作“同一个各向异性卷积实验”。

### 10.4 今晚预注册配置与判定

固定训练配置：`dataset_all_filtered`、patient-level split、scratch、seed42、`BCE-Dice`、`intensity_aug=on`、50 epochs、patience20、batch4、learning rate `1e-4`。F1-F3 均只由 val Dice 选择权重。

三者全部训练完成后，脚本才将 F0/F1/F2/F3 一起评估当前 436 张 development test，并输出相对 F0 的 Dice、HD95、clDice 和 Boundary F1 delta。development test 结果用于决定下一批方向，但不表述为最终独立泛化结论。

继续门槛：某候选相对 F0 同时满足 val 与 development-test Dice 正增益，且 development-test Dice `>= +0.006`、HD95 不变差、或 Boundary F1 `>= +0.008`。达到门槛后，下一晚仅补 F0 与胜出模型的 seed43；未达到则停止堆叠 fusion 模块，转向数据/标注上限、nnU-Net/MedNeXt 强基线和外层 CV。

### 10.5 F1-F3 seed42 结果（2026-07-15）

所有模型均使用同一 scratch、BCE-Dice、seed42 和 patient-level split；checkpoint 由 val Dice 选择。随后将已完成的 F0-F3 一起在 436 张 current development test 上统一评估，阈值固定为 0.5。

| 方法 | best val Dice | delta vs F0 | dev-test Dice | delta vs F0 | HD95 | delta vs F0 | clDice delta | Boundary F1 delta | 本轮判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F0 corrected TransUNet | 0.7897 | 0.0000 | 0.7535 | 0.0000 | 24.16 | 0.00 | 0.0000 | 0.0000 | baseline |
| F1 plain green single | 0.7897 | 0.0000 | 0.7463 | -0.0072 | 25.50 | +1.33 | -0.0050 | +0.0028 | 停止：普通 green feature 分支降低泛化 Dice |
| F2 directional green single | 0.7896 | -0.0001 | 0.7517 | -0.0018 | 24.21 | +0.05 | -0.0012 | +0.0045 | 停止：单尺度方向分支未超过 F0 |
| F3 directional green multiscale | 0.7938 | +0.0041 | 0.7536 | +0.0001 | 23.70 | -0.47 | +0.0025 | +0.0089 | 保留为结构指标候选，需 seed43 复现 |

解释：F3 未达到预设的 `dev-test Dice >= +0.006`，所以不能说它在区域重叠指标上胜出；但它是唯一同时保持 Dice 不降、改善 HD95、clDice，并使 Boundary F1 超过 `+0.008` 的配置。下一轮仅值得复现 F0/F3 seed43，不能继续给 F1/F2 堆叠更多模块。若 seed43 未复现 Boundary/HD95 的方向性改善，则停止该方向融合线。

实现诊断也支持这个结论：F2 训练后方向 gate 大量偏向 horizontal 21 分支，而 F3 的多尺度 fusion alpha 在 64/128 尺度为轻微负残差、最终尺度为正残差；说明模型确实使用了方向和尺度分支，而不是所有新增模块保持未激活。这个观察只能作为机制证据，不能替代多 seed 统计。

### 10.6 预训练是否削弱相对改进：待完成的严格配对

本轮 F1-F3 均为 scratch，未加载 ImageNet21k。历史旧 image-MSE 主线的三 seed 记录显示：scratch 相对 TransUNet Dice 约 `+0.0066`，ImageNet21k 预训练下约 `+0.0009`；这说明强预训练 backbone 可能已经吸收了部分低层纹理/对比度信息，使附加 prior 的边际收益缩小。但该结论不能直接外推至新 F3。

为回答新 F3 的实际情况，只补以下同 seed、同修正口径的配对实验：

| 试验 | 初始化 | 目的 |
|---|---|---|
| P0 | ImageNet21k pretrained TransUNet | 预训练 F0 baseline |
| P3 | ImageNet21k pretrained F3 directional multiscale | 计算 F3 在预训练下相对 P0 的真实增量 |

两者仍使用 `seed42`、BCE-Dice、intensity augmentation on、val checkpoint selection；完成后一起评估 current development test。只比较 `P3 - P0` 与 `F3 - F0` 的 delta，不能以 scratch F3 和 pretrained F0 的绝对值直接比较。

配对实验已完成。训练日志均明确显示成功加载 `R50+ViT-B_16.npz`；评估阶段先构建无初始化网络再加载训练后的完整 `best_model.pth`，因此 evaluator 打印的“未加载预训练权重”不表示漏载最终权重。

| 初始化 | 方法 | best val Dice | dev-test Dice | Dice delta | HD95 | HD95 delta | clDice delta | Boundary F1 delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| scratch | F0 TransUNet | 0.7897 | 0.7535 | baseline | 24.16 | baseline | baseline | baseline |
| scratch | F3 directional multiscale | 0.7938 | 0.7536 | +0.0001 | 23.70 | -0.47 | +0.0025 | +0.0089 |
| ImageNet21k | P0 TransUNet | 0.7932 | 0.7571 | baseline | 21.72 | baseline | baseline | baseline |
| ImageNet21k | P3 directional multiscale | 0.7920 | 0.7542 | -0.0029 | 22.32 | +0.60 | -0.0005 | -0.0112 |

结论：ImageNet21k 将 baseline 的绝对 dev-test Dice 从 0.7535 提高到 0.7571，并明显改善 HD95；但它没有增强 F3 的边际贡献。F3 在 scratch 下保留结构收益，在预训练下则 Dice、HD95、clDice 和 Boundary F1 相对同初始化 baseline 全部变差。当前新结构应以 scratch 作为主实验设置，预训练结果作为“强初始化削弱显式 green morphology prior 边际收益”的消融证据，而不能为了追求最高绝对值选择 pretrained F3。

探索性患者级 paired bootstrap（19 位 current development-test 患者）进一步显示：scratch F3 的患者平均 Boundary F1 delta 为 `+0.0129`，95% CI `[+0.0027, +0.0234]`；HD95 delta 为 `-1.50`，95% CI `[-2.85, -0.26]`。Dice CI 仍跨 0，因此当前只能说结构/边界指标出现阳性信号，不能说 Dice 显著提升。预训练 F3 的患者平均 Dice delta 为 `-0.0034`，Boundary F1 delta 为 `-0.0105`，未显示正向收益。该统计仍属于已参与研发的 development test 探索，论文最终统计应在外层 CV 或新 final holdout 上重做。

该主线暂称为“方向自适应绿色先验融合”（direction-aware green-prior fusion）。它比当前的图像 MSE 蒸馏更符合甲襞毛细血管的细长、方向多变、低对比度特点，也能把已有的 green prior、形态学动机和失败的 decoder distillation 收束为一个连贯故事。

## 2. 数据与评估审计：没有发现 train/test 混淆

### 2.1 实测数据划分

| split | 图像数 | image-mask 文件名对应 | 可恢复患者数 | 与其他 split 患者重叠 |
|---|---:|---|---:|---:|
| `train` | 1,838 | 1,838/1,838，0 个错配 | 81 | 0 |
| `val` | 449 | 449/449，0 个错配 | 19 | 0 |
| `test` | 436 | 436/436，0 个错配 | 19 | 0 |

审计方式：由 `third_party/ANFC_OURS_All_dataset/backup_original_names/rename_mapping.txt` 恢复每个 `ANFC_*.png` 的原始患者 ID，再与 `dataset_all_filtered` 三个 split 对照。三组图像文件名和患者 ID 均无重叠。

结论：当前 `dataset_all_filtered/test` 没有被重新划入训练，也不是把整个数据集重新划分后误当测试集。此前汇报的 436 张测试图口径正确。

### 2.2 green teacher 对齐

`dataset_all_filtered/train/images`、`train/masks`、`train/teacher_priors_green_only` 均为 1,838 个文件，文件名完全一致。`scripts/generate_teacher.py` 使用原图 BGR 的 G 通道，并复制为三通道 teacher 图；不存在 RGB/BGR 通道读取错误。

统一复评 `evaluate_all.py` 使用固定 `dataset_all_filtered/test`、`img_size=256`、`threshold=0.5`，并按单张图计算 Dice、IoU、Recall、Precision、Specificity、Accuracy、surface HD95、clDice、Boundary F1。因此结果表的横向比较口径是统一的。

## 3. 当前结果应该怎样解释

### 3.1 多 seed 结果比单次最高值更重要

| 训练设定（3 seeds: 42/43/44） | 方法 | Dice | 相对 TransUNet | HD95 | clDice | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|
| scratch | TransUNet | 0.7518 +/- 0.0009 | baseline | 23.64 | 0.8403 | 0.6373 |
| scratch | Ours green MSE10 Grad0 | 0.7584 +/- 0.0011 | +0.0066 | 23.42 | 0.8467 | 0.6468 |
| scratch | C3 (current implementation) | 0.7576 +/- 0.0025 | +0.0058 | 22.62 | 0.8461 | 0.6453 |
| pretrained | TransUNet | 0.7569 +/- 0.0012 | baseline | 22.95 | 0.8494 | 0.6369 |
| pretrained | Ours green MSE10 Grad0 | 0.7578 +/- 0.0027 | +0.0009 | 22.04 | 0.8481 | 0.6370 |
| pretrained | C3 (current implementation) | 0.7592 +/- 0.0022 | +0.0023 | 22.45 | 0.8499 | 0.6455 |

说明：HD95 越小越好；以上为 per-image metric 的 test 均值，再对三次训练取均值。原始记录在 `results/scratch_delta_multiseed_20260710/metrics_summary.csv`。

组会建议口径：

```text
从头训练下，green prior 对 Dice 有约 +0.0066 的稳定收益；
预训练下，C3 的平均 Dice 仅高 +0.0023，但 Boundary F1 高 +0.0086。
因此“green prior 有效”成立；“C3 已经显著优于强 baseline”尚不成立。
```

### 3.2 已有方向的正确判断

| 方向 | 当前证据 | 正确结论 |
|---|---|---|
| green-only image MSE prior | scratch 三 seed Dice +0.0066，clDice/Boundary F1 也提高 | 保留为已验证基础发现 |
| 不使用亮度/对比度 jitter | seed42 pretrained Dice 0.7581，结构指标也改善 | 有信号，但需在修正增强实现后复验 |
| C3 组合 | pretrained 三 seed Dice/Boundary F1 最好，但 seed44 不稳定 | 候选方向，不是最终模型 |
| soft clDice + Boundary loss | clDice 有时提高，但 Dice/Precision/Boundary F1 不稳定甚至下降 | 不再作为主优化手段 |
| Frangi/black-hat teacher | 未超过原始 green prior | 暂停 |
| decoder distillation V1/V2 | V1 0.7502；V2 0.7543，均低于主线 | 暂停，不把负结果解释成“蒸馏无效” |
| late CNN fusion probe | 仅在最终 decoder feature 做 residual fusion，未超过主线 | 不能据此否定真正的双路多尺度融合 |

## 4. 代码审计发现：下一轮正式训练前必须处理

### A. 验证指标按 batch 平均，可能选错 checkpoint

位置：`train_unified.py:373`、`train_unified.py:384`。

目前验证过程先对每个 batch 求平均 Dice，再除以 batch 数。验证集有 449 张图，batch size=4 时最后一个 batch 只有 1 张图，却与前面每个 4 张图的 batch 权重相同。这会影响 early stopping 和 `best_model.pth` 的选择。

影响范围：统一复评是逐图平均，已报告的 test 表可以继续使用；但旧模型的最佳 epoch 不一定是严格的 val Dice 最佳 epoch。后续所有正式训练应改为收集每张图的指标后再取全体均值。

### B. 测试集已经被反复观察，不能再称为完全独立的最终 test

位置：`train_unified.py:205`、`train_unified.py:418`。

每次训练结束都会对 test 做一次评估，随后又根据 test 结果决定下一个模块、loss 和配置。代码没有把 test 样本加入训练，但研究决策已经反复看过 test，这属于开发阶段的 test exposure / 多重试验风险。

处理原则：

1. 旧 test 结果改称“内部开发集结果”，保留全部记录，不删除也不篡改。
2. 从下一轮开始训练脚本默认只看 train/val；test 只由单独的最终评估脚本运行。
3. 论文正式结果建议采用 patient-level 5-fold cross-validation，或保留一组从未参与选模的患者级 final holdout。若没有外部队列，必须在局限性中说明。

### C. 当前“各向异性卷积”并不是真正的方向卷积

位置：`models/joint_framework.py:96` 至 `models/joint_framework.py:110`。

现在的 `strip7` 是 `1x7 -> 7x1` 串联，`strip21` 是 `1x21 -> 21x1` 串联。串联后本质上接近因式分解的大二维 `7x7/21x21` 感受野，不是互相独立的水平条带和垂直条带响应。因此现有 C1/C3 可表述为“大感受野因式分解增强”探针，不能严谨地表述为“方向感知血管卷积”。

下一版应把 `1x7`、`7x1`、`1x21`、`21x1` 作为四个并行分支，再用方向选择门控或注意力进行自适应融合。这才与甲襞血管的多方向细长形态相对应。

### D. intensity augmentation 的实现会改变像素绝对值关系

位置：`datasets/dataset_vessel.py:45` 至 `datasets/dataset_vessel.py:48`。

目前使用 `cv2.convertScaleAbs(image, alpha, beta)`。`convertScaleAbs` 会执行绝对值变换；尽管当前 alpha 为正、beta 范围不大，低亮度像素仍可能被非物理地翻转。更关键的是：image 被 brightness/contrast jitter，teacher 保持原始 green prior，训练目标变成“从受扰 RGB 恢复未扰动 teacher”，而不是单纯的强度先验保持。

这不是致命错误，但会直接影响“关闭 intensity augmentation 有效”的物理解释。应改为 float 域的线性变换或 gamma 变换后 clip 到 `[0,255]`，并在论文中准确描述 image/teacher 是否共享光度变换。

### E. decoder distillation V1/V2 的失败有清楚原因

V1 位置：`models/joint_framework.py:149` 至 `models/joint_framework.py:177`。

V1 是同一个 segmentor 对 enhanced RGB 与 green view 输出 decoder feature，然后以裸 MSE 约束一致；它不是固定强 teacher 指导 student。student 处于 train mode，而 teacher view 暂时处于 eval mode，BN 分布也不同，且 layer `2,3` 是高分辨率特征，容易被低层纹理差异主导。

V2 位置：`models/joint_framework.py:180` 至 `models/joint_framework.py:232`，`losses/joint_loss.py:371` 至 `losses/joint_loss.py:402`。

V2 已修正为独立 frozen direct-green teacher、projection、final decoder layer、`cosine_mse` 和低权重 0.1；但其 teacher 权重来自 scratch 的 direct-green baseline（内部 test Dice 约 0.7530），本身不强于 pretrained RGB student baseline（0.7567）。而且 V2 在 payload 中产生了 `teacher_logits`，损失函数实际只读取 feature，没有 logit/KL/结构图蒸馏。V2 结果 Dice=0.7543、HD95=23.44、Boundary F1=0.6290，低于 baseline 和主线。

结论：V2 没有发现路径、shape、teacher 文件错位等低级错误；负结果更符合“teacher 不够强 + 跨视图 decoder feature 不可直接对齐 + 缺少可靠性/结构目标”的机制性问题。此方向暂时停止，不值得原样重跑。

### F. 汇总脚本中的 delta 字段不可用于多 seed 统计

位置：`scripts/run_scratch_delta_multiseed_20260710.ps1:250` 至 `scripts/run_scratch_delta_multiseed_20260710.ps1:252`。

脚本中 seed43/44 的 `delta_vs_transunet_seed42` 与 `delta_vs_ours_seed42` 都固定减去 seed42 的常数，不是同 seed paired delta。原始 Dice/HD95/clDice/Boundary F1 数据正常，但这两列不应放进论文或组会统计。多 seed 对比应使用每个 seed 内同 seed 相减，再汇报 mean +/- std、bootstrap CI 和 Wilcoxon signed-rank 结果。

### G. 次要兼容问题

`select_threshold_on_val.py` 的 `--joint_model` choices 没有 `decoder_distill_v2`，且未透传 V2 teacher 配置。当前 V2 已暂停，未影响主表；但以后若重新启用，必须同步修复阈值选择脚本。

## 5. 下一轮创新主线：方向自适应绿色先验融合

### 5.1 为什么不是继续做图像 MSE 或 decoder MSE

图像级 MSE 已证明能提供弱但稳定的引导；裸 decoder MSE 则把不同视图、不同域的所有 feature 都强行拉近，反而损伤分割。我们需要的不是“RGB feature 像 green feature”，而是“只在血管结构需要时，green contrast prior 帮助 RGB decoder 恢复细长连续边界”。

### 5.2 建议的结构

```text
RGB image ----------> pretrained TransUNet encoder/decoder ----------> segmentation logits
                             ^               ^               ^
                             |               |               |
green channel -> light prior encoder -> directional features -> gated cross-scale fusion
                               |-- 1x7  (horizontal, independent)
                               |-- 7x1  (vertical, independent)
                               |-- 1x21 (horizontal, independent)
                               `-- 21x1 (vertical, independent)
```

核心设计约束：

1. green 分支只需轻量 CNN，不再复制一个完整 TransUNet。
2. 四个方向/尺度分支必须并行，避免当前串联假各向异性卷积的问题。
3. 融合发生在 decoder 的 2-3 个尺度，通过 gate 决定 green prior 是否介入；不要仅在最末层加一个 CNN residual。
4. 第一轮只使用 BCE-Dice，先隔离结构贡献；不同时叠加 clDice、Boundary loss、KD、BN 消融等多个变量。
5. 若方向融合有效，再增加轻量 boundary/centerline auxiliary head，作为第二阶段，而不是和主结构同时上线。

### 5.3 论文叙事

```text
甲襞毛细血管具有细长、弯曲、局部低对比度、方向多变的特点。
RGB Transformer 能建模全局上下文，却不能保证在弱对比细血管处保留局部方向连续性；
green channel 含有可重复验证的局部对比先验，但直接输入或图像级蒸馏都不足。
因此，我们通过方向自适应、跨尺度、门控式融合，将 green prior 仅在需要的 decoder 区域转化为结构补充，
从而同时提升区域重叠、中心线连续性与边界精度。
```

这比“给 TransUNet 加一些 strip conv 和 MSE”更像一个完整方法；但前提是严格消融、强 baseline、患者级统计和可视化必须补齐。

## 6. 推荐实验顺序（不是今晚全部开跑）

### Phase 0：先修口径，再训练

| 编号 | 工作 | 目的 | 是否必须 |
|---|---|---|---|
| P0-1 | 修正 train/val 逐图平均、取消训练末尾自动 test | 让 checkpoint 选择和 test 隔离正确 | 必须 |
| P0-2 | 修正 intensity augmentation 的 float/gamma 实现，并记录 image/teacher 光度关系 | 重新检验强度先验结论 | 必须 |
| P0-3 | 固定 patient split manifest、训练配置 JSON、random seed 与权重路径 | 可复现与论文方法部分 | 必须 |
| P0-4 | 对旧 TransUNet、green MSE、C3 的 per-image CSV 做 paired bootstrap/Wilcoxon | 先判断已有 green prior 是否真的显著 | 必须 |

注意：P0 修正后，旧结果可作为“探索期结果”保留，不能和新训练结果混成一个正式主表。

### Phase 1：最小、可判别的方向融合探针

固定：同一 patient split、同一预训练设置、BCE-Dice、统一增强、seed42、只用 val 选 checkpoint；测试集在预注册的全部 Phase 1 方案结束后统一评估一次。

| 编号 | 方法 | 关键问题 |
|---|---|---|
| F0 | 修正后的 TransUNet | 新训练/评估口径的 baseline |
| F1 | RGB + green lightweight branch，无方向卷积 | 绿分支本身是否有效 |
| F2 | F1 + 并行 `1x7/7x1/1x21/21x1` | 方向和尺度是否有效 |
| F3 | F2 + decoder 三级 gated fusion | 跨尺度门控是否优于简单拼接 |
| F4 | F3 去掉 green 分支或方向 gate | 对完整模型做必要消融 |

决策门槛：

```text
继续：F3 相对 F0 在 seed42 Dice >= +0.006，且 HD95 不变差、Boundary F1 >= +0.008。
复现：达到继续门槛后，F0/F3 立即补 seed43/44，并做患者级/图像级配对统计。
停止：F3 不超过 F1 或仅有 <= +0.002 的 Dice 波动，停止继续堆融合模块。
```

### Phase 2：结构辅助分支，只在 Phase 1 阳性后进行

若 F3 有阳性信号，增加一个从 decoder feature 输出的 boundary/centerline auxiliary head，GT 边界由 mask 生成，中心线在离线预处理中生成。主 segmentation logits 仍只用 BCE-Dice；辅助损失建议从 0.05、0.10、0.20 三档开始。这样检验的是多任务结构监督，而不是把 topology loss 硬加到主 logits 上。

### Phase 3：强 baseline 与发表级验证

1. 跑 2D nnU-Net v2（同一患者级 split 或 5-fold）作为必备强 baseline。
2. 跑 MedNeXt-S 或 U-Mamba/UMamba 之一，而不是再跑多个 SMP 小模型。
3. 若强 baseline 明显超过 TransUNet，再将 green-prior fusion 移植到最强可用 backbone，或诚实地把 TransUNet 作为其中一个 baseline。
4. 最终报告 patient-level 5-fold CV 或一个从未用于方法选择的 final holdout；补 paired bootstrap 95% CI、Wilcoxon、成功/失败案例、参数量、FLOPs 和速度。

## 6A. 7 月 14-16 日三晚冲刺安排

这三晚不适合把 nnU-Net、MedNeXt、方向融合、结构辅助头都正式跑完。更高收益的策略是先把新口径建立好，再让一个真正有论文叙事的模型完成首轮探针。否则会得到更多不可比较的 test 数字，而不是可用的结论。

| 晚上 | 任务 | 运行内容 | 交付物 | 不做什么 |
|---|---|---|---|---|
| 7 月 14 日晚 | 修正与基线 | 修正逐图 val metric、关闭训练末尾自动 test、修正 intensity augmentation；完成 1 epoch smoke；启动 F0 TransUNet scratch 和 F1 green-prior branch scratch | 正确的训练配置、val 曲线、可用权重 | 不跑 test，不跑 decoder KD |
| 7 月 15 日晚 | 核心创新探针 | 运行 F2 真并行方向分支、F3 多尺度 gate；用同一 seed42 和同一训练口径 | F0-F3 的 val 结果与模块消融 | 不加 clDice/Boundary loss，不再加 no-BN 变量 |
| 7 月 16 日晚 | 复现与组会材料 | 选择最优候选与 F0 补 seed43；统一 final eval 一次；做 bootstrap、案例可视化、参数/FLOPs/速度表 | 7 月 17 日可汇报的相对提升、置信区间和 6-10 张案例 | 不临时切换大 backbone；nnU-Net 只做环境准备 |

### 6A.1 三晚的最小实验矩阵

| 名称 | 模型与变量 | 设置 |
|---|---|---|
| F0 | 修正后的 TransUNet baseline | scratch，BCE-Dice，seed42 |
| F1 | F0 + 单尺度 green-prior gate | 仅验证 green feature 分支是否有增益 |
| F2 | F1 + 真并行 `1x7/7x1/1x21/21x1` directional prior | 仅验证方向/尺度建模 |
| F3 | F2 + decoder 三尺度 gated fusion | 验证多尺度融合，不添加结构 loss |
| R1 | F0 与最优 F1/F2/F3 的 seed43 | 只在 seed42 达到门槛后运行 |

所有 F0-F3 必须预先固定：数据 split、增强、epoch、patience、learning rate、batch size、pretrain mode、val selection metric。由于旧 test 已被用于探索，三晚内的模型选择必须只使用 val。若时间不足，F2 优先于 F3；F3 优先于结构辅助头。

### 6A.2 是否需要重跑旧实验

| 实验 | 是否重跑 | 原因 |
|---|---|---|
| 旧 TransUNet / Ours green MSE / C3 | 不作为原配置重跑 | 结果可保留为探索期证据；下一轮由新 F0/F1 在修正口径下重新建立基线 |
| E2 decoder distill V1/V2 | 不重跑 | 两版均为负结果，继续原样重跑没有信息增益 |
| E3 late dual fusion | 不重跑 | 已知融合太晚；应替换成 F3 的多尺度 gate |
| C4 soft clDice + Boundary | 不重跑 | 结果显示结构 loss 与 Dice/Precision 存在冲突；下一阶段改为独立辅助头 |
| C1/C3 当前 anisotropic | 不重跑原版 | 当前串联写法不是真方向卷积，应以 F2 的平行实现替代 |
| direct green/CLAHE input | 暂不重跑 | 已充分证明替换 RGB 输入不如 RGB 主分支；F1/F2 是更有信息量的对照 |
| nnU-Net | 不建议塞入三晚正式结果 | 值得做，但要单独配好预处理、交叉验证和评估，不能匆忙得到一个不可比数字 |

### 6A.3 代码层面还可优化的点

1. 训练脚本把 val/test 评估改为逐图汇总；训练结束只保存权重和 val 指标，不自动读 test。
2. 数据集增加可配置的 photometric policy：`none`、float linear、gamma；所有配置写入 JSON。不要再用 `convertScaleAbs`。
3. 记录一个固定的 `split_manifest.json`：文件名、患者 ID、split、生成版本。论文与复现都以它为准。
4. 每个 run 保存 `config.json`、git commit/文件版本、best epoch、val per-image CSV；最终评估再生成 test per-image CSV。
5. 方向分支使用 group/pointwise 轻量卷积，避免 21 长核造成不必要参数暴涨；门控初值建议让 prior 的初始贡献接近 0，保证 F0 是稳定退化情形。
6. 训练日志同时记录 mask loss、prior/gate 正则、每层 gate 均值；若性能下降，能判断是 prior 没被使用还是过度干预。

## 7. 文献阅读清单与借鉴点

以下先作为组会阅读清单。网络元数据接口本次检索不稳定，正式写论文前需逐篇下载 PDF 并核对题名、版本、DOI、期刊分区和实验设置，不能只引用网页摘要。

| 文献/资源 | 需要借鉴的不是“照抄模块”，而是 | 对本项目的实际启发 |
|---|---|---|
| ANFC dataset/pipeline, arXiv:2312.05930 | 甲襞任务定义、数据分割和临床意义 | 用作任务/数据背景；明确自身数据患者级切分 |
| TransUNet, arXiv:2102.04306 | CNN-Transformer 的互补性与 decoder skip 结构 | 作为现有 backbone，不要假设它天然就是最佳 baseline |
| TransFuse, arXiv:2102.08005 | 多尺度、双路、反复交互融合 | 当前 E3 仅末层残差，不足以代表 TransFuse；新模型应做多尺度 gate |
| clDice, arXiv:2003.07311 | 管状结构应评估中心线连续性 | clDice 作为核心补充指标，不代表必须把 soft-clDice 强加进主损失 |
| Boundary DoU Loss, arXiv:2308.00220 | 边界误差在小目标中需独立刻画 | 优先做 boundary auxiliary head 或 boundary-aware 选择指标 |
| nnU-Net, Nature Methods 2021 | 强、可复现的自动配置 baseline | 必须补，避免论文只与 TransUNet/SMP 轻量模型比较 |
| MedNeXt, arXiv:2303.09975 | 大卷积核 biomedical baseline | 用于检验“大感受野”是否比当前 Transformer/假条带卷积更适合任务 |
| U-Mamba, arXiv:2401.04722 | 选择性状态空间的长程建模 | P2 候选，不应在主线尚未验证前投入大量时间 |
| Morphology-aware distillation for lightweight retinal vessel segmentation across fundus photography and OCT angiography（用户提供 Frontiers 链接） | teacher 可靠性、形态结构重建，而非裸 feature MSE | 解释 decoder V1/V2 为什么失败；若未来恢复 KD，应先训练质量显著更强的 teacher，再蒸馏结构图/logits |

## 8. 对 SCI 2-3 区目标的诚实判断

当前状态：有值得继续做的任务先验和完整的探索记录，但尚不满足“凭现有数字即可投稿”的状态。

| 维度 | 当前情况 | 投稿前最低补强 |
|---|---|---|
| Dice 相对提升 | scratch +0.0066；pretrained 最好均值 +0.0023 | 新主线要形成 >= +0.006 到 +0.010 的多 seed 稳定收益，或明确的结构收益和临床解释 |
| 统计 | 仅部分三 seed；未做 paired CI/显著性 | 3-5 seeds + per-image paired bootstrap/Wilcoxon |
| 强 baseline | 现有 SMP 均弱于 TransUNet | nnU-Net + MedNeXt/UMamba 至少两个强对照 |
| 测试独立性 | 当前 test 被反复用于实验决策 | 5-fold patient CV 或未使用 final holdout |
| 创新完整性 | green image prior 有效；C3/decoder/fusion未闭环 | 完成方向自适应先验融合和必要消融 |
| 临床/结构价值 | 已有 HD95、clDice、Boundary F1 | 可视化细血管连续性、末端、断裂和误检病例，最好加临床形态学下游指标 |

因此，SCI 2-3 区不是“不可能”，但不能靠把 Dice `0.7592` 写成“显著提升”达成。最合理的目标是：先构建一条可信、统计稳健、能解释为何改善细长毛细血管结构的主线，再决定是否以 Dice 或结构/临床指标作为文章主卖点。

## 10. 2026-07-17 双专家蒸馏：已完成准备与当日实验

### 10.1 新的可检验假设

单个 ImageNet21k 预训练 TransUNet（F0）在区域重叠上更强，而从头训练的方向性 green-prior F3 在部分病例的细长结构、边界和远距离误差上具有互补性。当前不再用同一网络对 RGB/green view 做裸 decoder feature MSE；改为将两个已经训练好的异构专家的**输出概率**平均，作为单模型学生的离线 soft target：

```text
p_ensemble = 0.5 * sigmoid(F0 pretrained RGB) + 0.5 * sigmoid(F3 scratch directional green prior)
L = BCE-Dice(student, GT) + lambda_kd * soft-BCE(student, p_ensemble)
```

学生仍是单个 RGB TransUNet，初始化为 F0 的完整训练权重；因此推理阶段不保留第二教师或 green 分支，避免把 ensemble 上限误报为单模型结果。

### 10.2 双教师复现与数据审计（已完成）

固定教师权重：

- F0：`results/experiments/all_filtered/f0_transunet_corrected_pretrained_seed42_20260715/0715_1907/best_model.pth`
- F3：`results/experiments/all_filtered/f3_directional_green_multiscale_scratch_seed42_20260715/0715_1556/best_model.pth`

离线工具先单独运行 F0 并缓存 float32 概率，释放 GPU 后再运行 F3；两套完整 TransUNet 不会同时驻留 RTX 4060 8GB 显存。最终 KD 文件是 float16 `.npy`，不使用会量化 soft probability 的 PNG。train/val/test 的 ensemble probability 与 disagreement 文件数分别为 `1838/449/436`，均与图像数一一对齐，且 shape 为 `256 x 256`、值域为 `[0,1]`。

| development-test (436 images) | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 pretrained RGB | 0.7571 | 0.6199 | 0.8099 | 0.7293 | 21.72 | 0.8468 | 0.6346 |
| F3 scratch directional green prior | 0.7536 | 0.6156 | 0.8013 | 0.7290 | 23.70 | 0.8429 | 0.6384 |
| Fixed 0.5 probability ensemble | **0.7636** | **0.6280** | 0.8053 | **0.7432** | **20.97** | **0.8542** | **0.6483** |
| Ensemble delta vs F0 | **+0.0064** | +0.0081 | -0.0046 | +0.0138 | -0.74 | +0.0074 | +0.0136 |

该 ensemble 是当前的 development-test 上限与教师有效性证据，不是论文中的单模型主结果。此前的患者级 bootstrap 也显示 Dice 与 Boundary F1 的方向性增益为正；正式统计仍需在最终患者级 CV/final holdout 上重新做。

### 10.3 本轮代码改动

| 文件/组件 | 改动 | 防止的问题 |
|---|---|---|
| `generate_dual_teacher_targets.py` | 顺序生成 F0/F3 概率、ensemble soft target、disagreement 和逐图指标 | 双教师 OOM、权重/文件错位、8-bit 量化 |
| `datasets/dataset_vessel.py` | 支持 `.npy` soft target/disagreement；flip/rotate 与 image/mask 同步，photometric jitter 仅作用于 image | 几何错位和强度先验错误增强 |
| `losses/joint_loss.py` | 新增 output-level `BCE-Dice + soft-BCE` KD；agreement 模式权重为 `clamp(1-disagreement, 0.25, 1)` | 重复 decoder feature MSE 的跨 view 分布冲突 |
| `train_unified.py` | 新增 `soft_kd`、`init_weight`、soft-target/KD 参数；仍只用 val Dice 选 checkpoint | 在线双教师占显存、训练中暴露 test |
| `scripts/run_dual_teacher_kd_20260717.ps1` | K0/K1/K2 顺序训练、统一 development-test、自动决策、`-SkipExisting` 续跑 | 中断后错把部分权重��为完成 |

### 10.4 7 月 17 日执行矩阵

统一：F0 完整权重初始化、seed42、BCE-Dice、batch4、lr `3e-5`、30 epoch、patience10、intensity augmentation on、仅 val Dice 保存 best checkpoint。训练结束后才对 development-test 做一次统一评估。

| 编号 | KD | 作用 |
|---|---:|---|
| K0 | `lambda_kd=0` | fine-tune control，排除额外训练轮数的影响 |
| K1 | uniform soft target，`lambda_kd=0.3` | 温和向双教师 ensemble 蒸馏 |
| K2 | uniform soft target，`lambda_kd=1.0` | 测试较强蒸馏约束 |

判定：若 K1/K2 最优者相对 K0 的 Dice `>= +0.001` 且绝对 Dice `>=0.7615`，再补 K0 和获胜 KD 的 seed43/44；若未通过，则只运行 K3 agreement KD (`0.3`) 和 K4 uniform KD (`0.1`)，不再继续堆 decoder loss 或结构 loss。强阳性标准为 Dice `>=0.7630` 且 HD95、Boundary F1 不差于 K0。

运行命令：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_dual_teacher_kd_20260717.ps1 -SkipExisting
```

### 10.5 双教师 KD 完整结果与判定（2026-07-18 更新）

第一阶段 K0--K2 完成后，最优 K2 的 Dice 为 `0.7609`。它相对 K0 有 `+0.0047` 的提升，但未达到预先固定的 `0.7615` 单模型保留阈值，因此按预注册流程运行了 K3/K4，而不是继续围绕 K2 任意调参。所有学生均从同一 F0 完整权重初始化，使用同一 seed、训练轮数、增强和 val-Dice checkpoint 选择；development-test 只在每一批候选训练完成后统一评估一次。

| 模型（seed42） | KD 设置 | Dice | 相对 K0 Dice | 相对原始 F0 Dice | HD95 | clDice | Boundary F1 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 原始 F0 pretrained | 无额外微调 | 0.7571 | +0.0010 | 0.0000 | 21.72 | 0.8468 | 0.6346 | 原始单模型基线 |
| K0 fine-tune control | `lambda=0` | 0.7561 | 0.0000 | -0.0010 | 21.90 | 0.8490 | 0.6451 | 排除额外训练轮数 |
| K1 uniform KD | `lambda=0.3` | 0.7585 | +0.0024 | +0.0014 | 21.92 | 0.8496 | 0.6448 | 弱正向，但不及 K2 |
| K2 uniform KD | `lambda=1.0` | **0.7609** | **+0.0047** | **+0.0037** | **21.35** | **0.8537** | 0.6451 | 当前 KD 最优单模型 |
| K3 agreement KD | `lambda=0.3`，`1-disagreement` 加权 | 0.7575 | +0.0014 | +0.0004 | 22.61 | 0.8481 | **0.6465** | Dice/HD95 不足，不保留 |
| K4 uniform KD | `lambda=0.1` | 0.7558 | -0.0003 | -0.0013 | 22.47 | 0.8498 | 0.6436 | 退化，不保留 |
| 双教师 0.5 概率 ensemble | F0 + F3，双模型推理 | **0.7636** | +0.0074 | +0.0064 | **20.97** | **0.8542** | **0.6483** | 教师上限，不能作为单模型主结果 |

K2 的改善主要体现为：Recall 相对 K0 `+0.0260`、clDice `+0.0047`、HD95 `-0.55`；但 Precision `-0.0153`，Boundary F1 基本持平。因此它说明双教师软标签可以把一部分 ensemble 知识迁移给单模型，却尚未完整保留 ensemble 的边界收益。K3 虽使 Boundary F1 相对 K0 增加 `+0.0014`，但 HD95 变差 `+0.71`，不能把这一点解释为可靠改进；K4 同时降低 Dice 和 Boundary F1，应停止。

患者级 paired bootstrap（19 名患者，10,000 次）进一步说明结果边界：

| 比较 | 患者平均 Dice delta（95% CI） | 其他可靠变化 | 解释 |
|---|---:|---|---|
| K2 vs K0 | `+0.00265` (`+0.00035`, `+0.00517`)；14/19 患者改善 | clDice `+0.00421` (`+0.00155`, `+0.00730`)；Recall 上升，但 Precision 下降 | KD 对额外微调 control 有小而正的患者级信号 |
| K2 vs 原始 F0 | `+0.00520` (`+0.00118`, `+0.00950`)；14/19 患者改善 | Boundary F1 `+0.01600` (`+0.00745`, `+0.02565`)；clDice `+0.01083` (`+0.00123`, `+0.02473`) | 有开发期正向证据，但同一 development-test 已参与选模，不能作为最终论文显著性 |
| K3 vs K0 | `+0.00135` (`-0.00159`, `+0.00441`)；10/19 患者改善 | Dice、clDice、Boundary F1 的 CI 均跨 0 | agreement weighting 没有获得稳定收益 |

本轮停止条件已经触发：K1--K4 中只有 K2 有明确正向信号，但未达到预设单模型 Dice `0.7615` 保留线，也未达到 `0.7630` 的强阳性标准。因此不再在当前 development-test 上继续搜索 KD 系数、教师权重或阈值。后续若推进双专家方向，应在未参与当前调参的患者级 outer-CV/final holdout 上验证 K2，并以 K0 为对照；若不复现，就将其保留为探索性结果，而不是论文主方法。

### 10.6 本轮收尾代码审计与修复

运行 K3/K4 时发现汇总工具默认总是写入 `metrics_summary.csv` 和 `first_night_decision.json`，会覆盖第一阶段 K0--K2 的汇总文件。该问题只影响汇总命名，不影响训练权重、统一评估或 soft target；现已修复为可传入输出文件名，并已从两份原始 `aggregate_results.csv` 重建结果：

| 文件 | 最终用途 |
|---|---|
| `results/dual_teacher_kd_20260717/metrics_summary.csv` | K0--K2 第一阶段汇总 |
| `results/dual_teacher_kd_20260717/first_night_decision.json` | K2 未达阈值，进入 K3/K4 的原始决策 |
| `results/dual_teacher_kd_20260717/fallback_metrics_summary.csv` | K0、K3、K4 回退阶段汇总 |
| `results/dual_teacher_kd_20260717/fallback_decision.json` | K3/K4 无稳定胜出者的记录 |
| `results/dual_teacher_kd_20260717/K2_vs_K0_patient_bootstrap.csv` | K2 对 K0 的患者级 CI |
| `results/dual_teacher_kd_20260717/K2_vs_original_F0_patient_bootstrap.csv` | K2 对原始 F0 的患者级 CI |

这也提示最终论文实验必须把每一阶段 manifest、评价 CSV、随机种子、检查点路径和决策规则固定归档；不能只保留最后一个汇总表。

## 9. 7 月 17 日组会建议讲法（约 3 分钟）

```text
这周我先对已有实验做了代码和数据审计。数据仍然是患者级 train/val/test，
测试集 436 张图没有进入训练，green teacher 和图像文件也完全对齐。

从结果看，最稳定的发现仍是 green prior：从头训练下平均 Dice 提升约 0.0066；
预训练下 Dice 收益缩小，但 C3 的 Boundary F1 有正向信号。

不过审计也发现两点需要先处理：训练时验证 Dice 是 batch 平均，可能影响最优 checkpoint；
此外测试集已经在多轮调参中被观察，因此后续会改为 val 选模、test 最后一次评估，并补患者级交叉验证。

我重新检查了四个创新点。decoder 蒸馏并非路径错误，而是 teacher 不够强、跨视图裸 feature 对齐不合理；
当前所谓各向异性卷积其实是串联的大二维卷积，也需要改成真正平行的水平/垂直方向分支。

所以下一阶段不再继续堆 loss，而是做 RGB 主分支和 green 轻量方向分支的多尺度门控融合，
先验证方向建模和绿色先验各自的贡献，再补 nnU-Net/MedNeXt 强 baseline、统计显著性和结构可视化。
```
