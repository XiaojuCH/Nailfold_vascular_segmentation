# 2026-07-03 组会汇报模板：文献启发与下一轮模型改进方向

更新日期：2026-07-01  
组会时间：2026-07-03 晚上  
当前阶段目标：先通过文献和现有实验结果重新确定下一轮模型改进方向，再进入代码实现和训练。

## 1. 本次组会主线

本周汇报建议围绕一个核心问题展开：

```text
当前 green-channel prior 已经验证有正向信号，但 Dice 提升幅度仍小。
下一步是否应该从“简单蒸馏/预处理”升级为“物理先验引导的特征融合 + 管状结构保真分割”？
```

上一阶段的结论不是失败，而是把方向收窄了：

- 直接把 green/CLAHE 图像作为输入，效果不如 Ours，说明提升不是简单预处理带来的。
- MSE consistency 是目前最稳定的 green prior 使用方式，但相对强 baseline 的 Dice 增益仍偏小。
- clDice + Boundary 分支能改善结构和边界指标，但没有显著拉高 Dice。
- SMP 系列强 baseline 没有超过 TransUNet，说明短期不必完全抛弃 TransUNet，但需要补更现代 backbone 作为中期对照。

因此，下周的重点不建议继续微调小 loss，而是优先尝试更有方法贡献的三条线：

1. **Prior-guided feature fusion**：让 green prior 参与 skip/decoder 特征选择，而不是只做输出一致性。
2. **Topology/boundary auxiliary branch**：把中心线和边界作为辅助监督，避免结构 loss 直接伤害主分割 Dice。
3. **Modern backbone sanity check**：用 nnU-Net / MedNeXt / Mamba-UNet 类方法判断当前上限是不是被 TransUNet 限制。

## 2. 当前结果快速复盘

| 类别 | 模型 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | TransUNet old | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 | 旧正式 baseline |
| Baseline | TransUNet pretrained | 0.7567 | 0.6188 | 0.8016 | 0.7331 | 22.69 | **0.8489** | 0.6336 | 预训练使 baseline 变强 |
| Ours | Ours green MSE pretrained | **0.7583** | **0.6208** | 0.8031 | 0.7347 | **22.05** | 0.8465 | 0.6414 | 当前 Dice/HD95 最好 |
| Ours | Ours clDice+Boundary old | 0.7567 | 0.6183 | 0.7928 | 0.7421 | 23.55 | **0.8533** | **0.6519** | 当前结构指标最好 |
| Strong baseline | SMP Unet++ EfficientNet-B3 | 0.7514 | 0.6130 | 0.7830 | 0.7419 | 24.48 | 0.8392 | 0.6391 | 未超过 TransUNet |

需要在组会中诚实说明：

```text
相对 TransUNet old，当前最好 Dice 提升约 +0.0061；
相对 TransUNet pretrained，只提升约 +0.0017。
如果论文目标是 SCI 2-3 区，目前还不能只靠“大幅 Dice 提升”来讲故事。
```

更合适的论文叙事应调整为：

```text
green-channel physical prior guided topology- and boundary-preserving
nailfold capillary segmentation.
```

也就是：不是单纯追求一个 Dice 冠军，而是强调甲襞毛细血管这种低对比、细管状结构任务中，物理先验如何帮助模型保留连续性、细小分支和边界。

## 3. 文献矩阵：优先阅读清单

### 3.1 甲襞/毛细血管任务背景

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 组会怎么讲 |
|---|---|---|---|---|
| P0 | ANFC: A Comprehensive Dataset and Automated Pipeline for Nailfold Capillary Analysis, 2023. https://arxiv.org/abs/2312.05930 | 构建甲襞毛细血管数据集和自动分析 pipeline，包含图像、视频、临床报告和专家标注。 | 说明甲襞自动分析是有临床和工程价值的，但公开研究仍相对少。 | 我们的任务不是普通自然图像分割，而是面向甲襞微循环的细结构分割。 |
| P1 | CapillaryNet, 2021. https://arxiv.org/abs/2104.11574 | 用深度学习和传统视觉结合自动分析皮肤微血管密度和红细胞速度。 | 支持“微血管分析需要自动化和结构量化”，但不是甲襞静态分割的直接 baseline。 | 可作为应用背景补充，不作为主方法对照。 |
| P1 | IFCIS-155 capillary segmentation benchmark, 2022. https://arxiv.org/abs/2207.06861 | 针对免疫荧光毛细血管图像构建分割 benchmark。 | 说明毛细血管/细管结构分割常面临数据少、结构细、边界复杂的问题。 | 用于支撑“细管状结构指标比单一 Dice 更重要”。 |

### 3.2 TransUNet 与医学 Transformer 改进

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 可转化实验 |
|---|---|---|---|---|
| P0 | TransUNet, 2021. https://arxiv.org/abs/2102.04306 | CNN 提供局部细节，Transformer 建模全局上下文，decoder 恢复空间分辨率。 | 当前 baseline 合理，但它的 skip/decoder 对低对比细血管未必最优。 | 继续保留 TransUNet pretrained 作为主 baseline。 |
| P0 | UCTransNet, 2021. https://arxiv.org/abs/2109.04335 | 重新设计 U-Net skip connection，用 channel-wise Transformer 减少 encoder-decoder 语义 gap。 | 我们不应该只在输出端做 MSE，而应让 green prior 参与 skip feature 选择。 | Prior-guided channel/spatial skip attention。 |
| P0 | TransFuse, 2021. https://arxiv.org/abs/2102.08005 | CNN 分支保留局部细节，Transformer 分支建模全局上下文，再用 BiFusion 融合。 | 甲襞血管同时需要局部细线和全局走向，可用双分支思路：RGB 主分支 + green prior 分支。 | RGB/green 双分支轻量融合，而不是替换输入。 |
| P1 | MISSFormer, 2021. https://arxiv.org/abs/2109.07162 | 分层 Transformer encoder-decoder，并用 context bridge 融合多尺度上下文。 | 如果继续改 TransUNet，可优先改 decoder/context bridge，而不是全换模型。 | Multi-scale prior bridge。 |
| P1 | Swin-Unet, 2021. https://arxiv.org/abs/2105.05537 | 纯 Transformer U-Net，用 shifted window 同时捕获局部和全局信息。 | 可作为中期替代 baseline，但实现成本和调参成本高于当前短期目标。 | 暂列为 baseline 候选，不作为本周 P0。 |

### 3.3 现代医学分割强 baseline

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 可转化实验 |
|---|---|---|---|---|
| P0 | nnU-Net, 2018/2021. https://arxiv.org/abs/1809.10486 | 强调自动配置、预处理、训练策略和推理设置对医学分割性能影响很大。 | 如果我们只比较手写 TransUNet，审稿人可能质疑 baseline 不够强。 | 跑 2D nnU-Net 或至少借鉴其训练策略：patch/augmentation/TTA/ensemble。 |
| P1 | MedNeXt, 2023. https://arxiv.org/abs/2303.09975 | Transformer 思路启发的现代 ConvNet，大核、可扩展，适合医学小数据。 | 如果 TransUNet 已接近上限，现代 ConvNet 可能更稳。 | 作为下一轮强 backbone 探针。 |
| P1 | U-Mamba, 2024. https://arxiv.org/abs/2401.04722 | CNN + state space model，提升长程依赖建模，面向 biomedical segmentation。 | 对长条状血管连通性有潜在帮助，但实现和训练风险较高。 | 中期 baseline，不建议本周立刻主攻。 |
| P1 | VM-UNet, 2024. https://arxiv.org/abs/2402.02491 | 基于 Vision Mamba 的 U-shaped medical segmentation baseline。 | 可作为“不要吊死在 TransUNet 上”的方向储备。 | 若导师要求换 baseline，可优先调研开源实现。 |

### 3.4 管状结构、边界与指标

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 可转化实验 |
|---|---|---|---|---|
| P0 | clDice, 2020/2021. https://arxiv.org/abs/2003.07311 | 针对血管、神经等管状结构，提出中心线感知的拓扑指标/损失。 | clDice 是甲襞毛细血管非常适合的主补充指标。 | 保留 clDice 指标；尝试中心线辅助 head，而不是只把 clDice 加进主 loss。 |
| P0 | Boundary loss, 2018/2019. https://arxiv.org/abs/1812.07032 | 针对类别不平衡分割，用边界距离指导训练。 | 甲襞血管前景少，边界细，传统 Dice/BCE 容易忽略边界。 | 边界辅助 head 或 boundary refinement。 |
| P0 | Boundary DoU Loss, 2023. https://arxiv.org/abs/2308.00220 | 专门强调边界区域的医学分割 loss，并在 UNet/TransUNet/Swin-Unet 上验证。 | 比普通 boundary loss 更容易包装为“边界保真”分支。 | 可作为下一轮 loss/auxiliary branch 候选。 |
| P0 | Metrics Reloaded, 2022/2024. https://arxiv.org/abs/2206.01653 | 强调根据任务属性选择合适指标，避免只看单一分数。 | 支持我们用 Dice + HD95 + clDice + Boundary F1 的组合。 | 论文中解释为什么要报告结构/边界指标。 |
| P0 | Retinal vessel evaluation inconsistency, 2021. https://arxiv.org/abs/2111.03853 | 指出血管分割论文中评价协议不一致会导致不可比。 | 支持我们强调统一 test、统一 threshold、统一指标实现。 | 写入实验设置，提升可信度。 |

### 3.5 Green channel / CLAHE / 血管物理先验

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 可转化实验 |
|---|---|---|---|---|
| P0 | Supervised Segmentation of Retinal Vessel Structures Using ANN, 2020. https://arxiv.org/abs/2001.05549 | 视网膜血管分割中使用 RGB green channel，理由是血管在 green channel 上更清晰；并使用 CLAHE、形态学操作等预处理。 | 支持 green channel 作为血管/微血管物理先验，而不是任意工程技巧。 | 将 green prior 从输出 MSE 升级为 feature-level prior。 |
| P1 | SA-UNet, 2020. https://arxiv.org/abs/2004.03696 | 视网膜血管分割中使用空间注意力增强小样本血管分割。 | 对我们启发更偏模块：green prior 可生成 spatial attention。 | Green-prior spatial attention gate。 |
| P1 | CAR-UNet, 2020. https://arxiv.org/abs/2004.03702 | 将 channel attention 用于 skip connections，提升视网膜血管分割。 | 和 UCTransNet 一起支持“skip connection 不是简单 concat，应该被注意力调制”。 | Prior-guided channel attention。 |
| P1 | DR-VNet, 2021. https://arxiv.org/abs/2111.04739 | 面向 retinal thin/tiny vessels，使用 residual dense 与 squeeze-excitation。 | 说明小血管提升常来自局部细节、注意力和结构保留。 | 轻量 SE/attention，不作为单独大创新。 |

### 3.6 Foundation model / SAM 方向

| 优先级 | 文献 | 核心内容 | 对我们的启发 | 可转化实验 |
|---|---|---|---|---|
| P2 | MedSAM, 2023/2024. https://arxiv.org/abs/2304.12306 | 大规模医学图像 mask 数据训练的通用医学分割 foundation model。 | 可作为未来强 baseline 或伪标签工具，但短期实现成本较高。 | 用 MedSAM/SAM-Med2D 做辅助标注或外部对照。 |
| P2 | SAM-Med2D, 2023. https://arxiv.org/abs/2308.16184 | 针对 2D 医学图像微调 SAM，讨论自然图像 SAM 到医学图像的 domain gap。 | 如果我们有未标注甲襞图像，可考虑 pseudo-label 或 prompt-based refinement。 | 中期探索，不建议本周主攻。 |

## 4. 从文献导出的下一轮实验方向

### 方向 A：Prior-Guided Skip Fusion（最推荐）

**动机**  
目前的 green prior 主要通过 enhancer 输出和 teacher 做 MSE consistency。这个约束太靠后，可能只能轻微改变输出，不足以明显改变分割特征。UCTransNet、TransFuse、CAR-UNet 都提示：skip/feature fusion 对医学分割尤其重要。

**方法设想**  

```text
RGB image -> TransUNet encoder
green/CLAHE prior -> lightweight prior encoder
prior feature -> channel/spatial attention
attention modulates TransUNet skip features or decoder features
```

**为什么论文叙事能讲通**  
甲襞毛细血管在 green channel 上对比更明显，但 direct input 实验证明“只换输入”不够。因此我们把 green prior 作为物理引导信号，作用于 feature selection，而不是粗暴替换 RGB。

**预期收益**  
比 MSE-only 更可能带来 Dice 提升，因为它直接影响 encoder-decoder 特征流；如果 Dice 不涨，也可能改善 clDice/Boundary F1。

**下周可跑版本**  

1. `RGB + green prior attention gate`：只在 decoder 高分辨率 skip 加 gate。
2. `RGB + green + CLAHE prior attention gate`：比较 green 和 CLAHE 是否互补。
3. 消融：无 prior gate / channel gate / spatial gate / channel+spatial gate。

### 方向 B：RGB + Prior 多通道输入（低成本、必须补）

**动机**  
我们已经跑过 direct green/CLAHE，但那是“用预处理图替代 RGB”。还没充分验证“RGB 保留原始信息 + prior 作为额外通道”。

**方法设想**  

```text
Input = [RGB, green, CLAHE(green)]
或 Input = [RGB, green, CLAHE(green), blackhat(green)]
```

第一层卷积/patch embedding 需要适配多通道，可用 RGB 权重均值或复制初始化。

**为什么值得做**  
这是最容易被审稿人问到的控制实验：既然说 green prior 有用，为什么不是直接 concat 输入就行？这个实验能回答。

**风险**  
如果直接 concat 就超过 Ours，说明当前蒸馏设计不够好；但这并不是坏事，可以把论文方法转向“physical-prior augmented input + feature fusion”。

### 方向 C：Topology/Boundary Auxiliary Head（结构指标主线）

**动机**  
已有 `clDice+Boundary` 结果说明结构指标能上去，但 Dice 没明显上去。直接把结构 loss 加到主 mask 上，可能造成 Recall/Precision 平衡被破坏。

**方法设想**  

```text
main head: vessel mask
aux head 1: boundary map
aux head 2: skeleton/centerline map

loss = BCE-Dice(main)
     + lambda_boundary * BCE/Dice(boundary)
     + lambda_skeleton * BCE/Dice(skeleton)
     + optional soft-clDice(main)
```

**为什么论文叙事能讲通**  
临床上毛细血管形态、连续性、分叉和边界比单纯像素重叠更有意义。辅助 head 让网络显式学习结构，而不是只靠主 mask loss 间接学习。

**下周可跑版本**  

1. main mask + boundary head。
2. main mask + skeleton head。
3. main mask + boundary + skeleton head。

### 方向 D：nnU-Net / MedNeXt / Mamba Baseline（中期补强）

**动机**  
如果目标是 SCI 2-3 区，审稿人可能会问：为什么不用更强的医学分割 baseline？SMP 系列没超过 TransUNet，但 SMP 不等于当前医学分割最强路线。

**建议顺序**

1. 先跑 `2D nnU-Net`，验证强训练策略和 TTA/ensemble 能否抬高上限。
2. 再考虑 `MedNeXt`，它适合小数据医学分割，工程上比 Mamba 稳。
3. 最后考虑 `U-Mamba / VM-UNet`，作为新颖 baseline 或未来工作，不建议本周立刻投入大量时间。

**组会表述**

```text
短期仍以 TransUNet pretrained 为主 baseline，因为它在当前实验中强于 SMP。
但为了避免论文只依赖一个 baseline，下一轮会补 nnU-Net/MedNeXt/Mamba 系列探针。
```

### 方向 E：TTA / Ensemble（冲指标上限，但不作为主创新）

**动机**  
如果单模型一直卡在 0.758 左右，可以用 TTA 或 ensemble 看任务上限。

**建议**

- TTA：horizontal/vertical flip、small scale。
- Ensemble：TransUNet pretrained + Ours green MSE + Ours clDice+Boundary。

**论文定位**

只能作为工程增强或 supplementary，不能当核心创新。

## 5. 下周实验优先级建议

### P0：最应该先做

| 实验 | 目的 | 成功标准 |
|---|---|---|
| RGB + green/CLAHE 多通道输入 | 验证 prior 作为额外通道是否优于 direct input 和 MSE-only | Dice 比 TransUNet pretrained 高至少 +0.005 |
| Prior-guided skip/decoder attention | 把 green prior 从输出约束升级到特征融合 | Dice 接近或超过 0.762；或 Boundary F1/clDice 明显提升 |
| main mask + boundary/skeleton auxiliary head | 把结构指标提升转化为可解释方法贡献 | clDice/Boundary F1 继续领先，Dice 不明显下降 |

### P1：有时间再做

| 实验 | 目的 | 成功标准 |
|---|---|---|
| 2D nnU-Net | 看强训练策略是否提高任务上限 | Dice >= 0.760 则需要纳入主表 |
| MedNeXt 2D 探针 | 检查现代 ConvNet 是否比 TransUNet 更适合 | Dice >= TransUNet pretrained |
| TTA/ensemble | 看 test 上限 | Ensemble Dice >= 0.765 可用于 upper-bound 分析 |

### P2：暂不建议本周主攻

| 实验 | 原因 |
|---|---|
| MedSAM/SAM-Med2D fine-tuning | 实现链路长，短期可能消耗时间且不一定适合细小血管 |
| 大量继续调 lambda | 目前小 loss tweak 已经证明收益有限 |
| 继续 blackhat/frangi teacher 单独尝试 | 已有结果显示没有明显 Dice 增益 |

## 6. 7 月 3 日组会汇报结构

### Slide 1：本周汇报目的

标题建议：

```text
甲襞毛细血管分割：从 green prior 验证到下一轮结构化改进
```

要讲的话：

```text
上周我们已经验证了 green-channel prior 有一定正向作用，但当前提升幅度不足以支撑“显著性能提升”的论文叙事。
所以本周主要汇报两件事：第一，文献里有哪些可迁移的改进思路；第二，下一轮模型应该怎么改才更可能带来实质提升。
```

### Slide 2：当前实验结论

放当前关键表格，只放 4-5 个模型：

- TransUNet old
- TransUNet pretrained
- Ours green MSE pretrained
- Ours clDice+Boundary old
- SMP Unet++ EfficientNet-B3

强调：

```text
当前最好 Dice 是 0.7583；
相对旧 TransUNet 提升 +0.0061；
相对 pretrained TransUNet 提升 +0.0017；
结构指标最好的是 clDice+Boundary 分支。
```

### Slide 3：问题与瓶颈

建议写三点：

1. Dice 提升小，不能只讲“大幅超过 baseline”。
2. 直接输入 green/CLAHE 不如 Ours，说明 green prior 需要更合理融合。
3. 单纯 loss tweak 和结构 teacher 已经遇到瓶颈，需要把 prior 融入特征层。

### Slide 4：文献启发一：医学 Transformer 的 skip/fusion 改进

放 TransUNet、UCTransNet、TransFuse、MISSFormer。

要讲的话：

```text
这些工作共同提示：医学分割不是只有 encoder 强就够，encoder-decoder 的特征融合方式非常关键。
我们的 green prior 目前主要约束输出，下一步可以让它去调制 skip 或 decoder 特征。
```

### Slide 5：文献启发二：管状结构指标与边界保真

放 clDice、Boundary loss、Boundary DoU、Metrics Reloaded。

要讲的话：

```text
甲襞毛细血管是典型细管状结构，临床关心的不只是像素重叠，还包括连续性、断裂、边界和细小分支。
所以我们后续会把 clDice、Boundary F1、HD95 作为核心补充指标，而不是只报告 Dice。
```

### Slide 6：文献启发三：green channel 物理先验

放 retinal vessel green channel / CLAHE 相关工作。

要讲的话：

```text
视网膜血管分割里 green channel 常被用于增强血管与背景对比。
甲襞图像中毛细血管同样是低对比红色细结构，因此 green prior 有物理和成像基础。
但我们 direct input 实验证明，简单预处理不是最优，需要 feature-level fusion。
```

### Slide 7：下一轮模型方案

建议画一个简单框图：

```text
RGB image -----------------> TransUNet encoder/decoder -----> mask
                              ^       ^       ^
green/CLAHE prior -> prior encoder -> attention/gate on skips

optional:
decoder feature -> boundary head
decoder feature -> skeleton head
```

汇报命名可以先暂定：

```text
PGF-TransUNet: Physical-prior Guided Fusion TransUNet
```

或者中文：

```text
物理先验引导的特征融合 TransUNet
```

### Slide 8：实验计划与导师需要拍板的问题

建议直接列：

1. 是否认可下一步从 MSE-only 转向 prior-guided feature fusion？
2. 是否把 clDice/Boundary F1 作为论文重要结构指标，而不是只盯 Dice？
3. 是否需要同步补 nnU-Net/MedNeXt/Mamba baseline，避免只依赖 TransUNet？

## 7. 可以对导师说的“谨慎但积极”的版本

```text
目前我们已经验证 green prior 有效，但它带来的 Dice 增益还比较小。
我不想继续只调 lambda 或堆小模块，因为这很可能只能带来 0.001 级别波动。
我查了一些医学 Transformer、视网膜血管分割和管状结构分割的文献，感觉下一步更合理的是：
把 green prior 从输出端一致性约束，前移到 encoder-decoder 特征融合阶段，
同时用 boundary/skeleton 辅助分支强化细管状结构。
这样即使 Dice 提升不是特别大，也能在 clDice、Boundary F1、HD95 和可视化上形成更完整的论文叙事。
```

## 8. 风险判断

### 对 SCI 2-3 区的积极因素

- 数据任务相对专门：甲襞毛细血管分割公开工作少。
- 已有统一评估和较完整消融。
- green prior 有物理依据，direct input 对照能证明不是简单预处理。
- clDice/Boundary/HD95 能支撑管状结构保真叙事。

### 当前主要风险

- Dice 提升仍小。
- 目前多为单 seed，统计显著性不够。
- 尚缺外部验证或 cross-validation。
- 如果不补强 baseline，审稿人可能质疑 TransUNet 是否足够代表当前水平。

### 解决思路

```text
短期：prior-guided fusion + auxiliary structure head，争取更大单模型提升。
中期：补 nnU-Net/MedNeXt/Mamba baseline，确认任务上限。
论文阶段：多 seed + paired statistics + 结构可视化，避免只靠单次 Dice。
```

## 9. 下周之后建议执行顺序

1. 先实现并跑 `RGB + green/CLAHE prior 多通道输入`，这是最直接的补充对照。
2. 再实现 `prior-guided skip/decoder attention`，作为下一轮主创新候选。
3. 同步准备 `boundary/skeleton auxiliary head`，如果 Dice 不涨，至少能稳住结构指标叙事。
4. 若模型改进后 Dice 仍小于 0.762，优先补 `2D nnU-Net` 或 `MedNeXt` 强 baseline。
5. 找 10-20 张典型图做成功/失败可视化，辅助判断模型到底改善了哪里。

## 10. 本周阅读优先级

如果时间有限，建议先读这 8 篇：

1. ANFC nailfold dataset/pipeline: https://arxiv.org/abs/2312.05930
2. TransUNet: https://arxiv.org/abs/2102.04306
3. UCTransNet: https://arxiv.org/abs/2109.04335
4. TransFuse: https://arxiv.org/abs/2102.08005
5. clDice: https://arxiv.org/abs/2003.07311
6. Boundary DoU Loss: https://arxiv.org/abs/2308.00220
7. Metrics Reloaded: https://arxiv.org/abs/2206.01653
8. Retinal green channel/CLAHE example: https://arxiv.org/abs/2001.05549

这 8 篇足够支撑下一次组会的核心逻辑：

```text
甲襞任务有需求 -> TransUNet 是合理 baseline -> skip/fusion 是医学分割关键 ->
green channel 有血管成像依据 -> 管状结构需要拓扑/边界指标 ->
下一步方法应升级为 physical-prior guided feature fusion。
```

