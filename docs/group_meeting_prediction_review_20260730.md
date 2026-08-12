# 组会汇报：从 development-test 图像复盘模型瓶颈

## 本周任务

导师要求从 `dataset_all_filtered/test` 的真实预测图出发，判断当前提升小的原因，而不是继续仅比较总指标。

本轮将该集合称为 **development-test**：它已参与过方法选择，用于诊断；论文最终结果仍需患者级 outer-CV 或未参与研发的 final holdout。

## 复盘设置

- 固定 development-test：436 张，阈值 0.5，图像尺寸 256。
- 已保存 9 组代表模型的全部预测，共 3924 张二值预测图。
- 每张图生成 RGB、GT 与各模型 TP/FP/FN 对照：绿色 TP，红色 FP，蓝色 FN。
- 自动筛选：baseline 最好/最差、最大漏检、最大误检、最大 HD95、方法改善/退化最大、模型分歧最大病例。

入口与结果：

- `results/prediction_error_review_20260730/index.html`：全部病例可视化总览。
- `results/prediction_error_review_20260730/analysis_summary.md`：完整指标和样本排序。
- `results/prediction_error_review_20260730/all_cases`：436 张横向对照图。

## 核心结果

| 对照 | Dice 变化 | 改善 / 退化图数 | 结构变化 | 解读 |
|---|---:|---:|---:|---|
| green MSE vs scratch TransUNet | +0.0050 | 222 / 214 | HD95 -0.83；Boundary F1 +0.0073 | 有效，但不是所有图都改善 |
| F3 directional vs scratch TransUNet | +0.0014 | 234 / 202 | HD95 -0.41；Boundary F1 -0.0021 | 主要是 Recall 增加，Precision 下降 |
| K2 soft KD vs pretrained F0 | +0.0037 | 280 / 156 | HD95 -0.37；Boundary F1 +0.0105 | 最好的单模型 KD 信号，但幅度有限 |
| S2 structure vs S0 | +0.0010 | 227 / 209 | HD95 -1.13；Boundary F1 +0.0007 | 结构距离改善伴随过分割风险 |
| decoder KD V2 vs pretrained F0 | -0.0029 | 213 / 223 | HD95 +1.72；Boundary F1 -0.0056 | 失败，过分割更明显 |
| F0+F3 ensemble vs pretrained F0 | +0.0064 | 299 / 137 | HD95 -0.74；Boundary F1 +0.0136 | 双模型互补明确，但尚非单模型方案 |

## 从图像看到的瓶颈

### 1. green MSE 的作用是修复“低召回困难图”，不是普遍抬高上限

- baseline Recall < 0.6 的 37 张图中，green MSE 有 30 张改善，平均 Dice `+0.0459`。
- baseline Recall/Precision 均 >= 0.6 的 353 张图中，green MSE 平均 Dice 变化仅 `+0.00005`。
- 因此 green prior 的叙事应是：**补充局部低对比、断裂或暗细血管的可见性**，而不是声称在清晰图上全面超过 RGB 模型。

### 2. 当前误差并非单一的“漏检问题”

- 低对比或大面积/连续血管图：模型常出现大范围蓝色 FN，表现为 Recall 低、HD95 大。例如 `ANFC_001302.png`、`ANFC_001303.png`、`ANFC_001343.png`。
- 背景纹理或反光接近血管的图：模型会产生大块红色 FP，表现为 Recall 不低但 Precision 很低。例如 `ANFC_000389.png`。
- 这两类错误需要不同策略：前者需要可见性/连续性先验；后者需要抑制伪影与不确定区域。单纯提高结构损失或 Recall 会扩大 FP。

### 3. 困难病例存在患者聚集，提示数据域差异与标注上限需要排查

baseline 最难的患者包括：`8_84237`（Dice 0.6730）、`8_55896`（0.6796）、`9_60031`（0.6993）、`8_92229`（0.6996）。其中 `8_92229` 有 43 张图且占多数极端漏检案例。

这说明平均 Dice 的局限：需要进一步逐患者检查采集条件、对焦/曝光、染色、病理形态与标注一致性，确认是否存在 domain shift 或低质量标注子集。

### 4. 为什么复杂模块没有带来大提升

- F3 和 S2 往往通过提高 Recall 得到较小的结构指标改善，但同步降低 Precision，说明模型更容易“把背景当血管”。
- decoder KD V2 的可视化显示红色 FP 增多，和其 Precision 0.7220、Boundary F1 0.6290 的退化一致。
- ensemble 能同时补 F0 的局部漏检、减少部分误检，但其增益没有被 K2 单模型完全保留，说明当前的全图 uniform soft-KD 会把两位教师的局部冲突平均掉。

## 当前结论与下一步

1. 当前最可信的正结论是：green prior 对低召回困难图具有补偿价值；双专家概率融合证明 RGB 语义和 green morphology 存在互补。
2. 继续堆 strip convolution、裸 decoder feature MSE 或提高结构损失权重，不是优先方向；它们容易以 Precision 为代价换取 Recall。
3. 官方 MMSegmentation DeepLabV3+ 已在固定 patient-level split 和统一标签/指标下完成复现，未超过现有 TransUNet/green prior；因此短期不再把“换成官方 DeepLabV3+”作为主线。
4. 后续资源优先投入错误子群/标注审计、强 baseline 与可靠性融合；同时向师妹索取 checkpoint 和原始评估代码，厘清其 HD95 15.x 的数据与指标口径。

## 官方 MMSegmentation DeepLabV3+ 复现结果（2026-07-31）

为排除“已有 SMP DeepLabV3+ 不是官方实现”这一可能，本轮使用师妹提供的 MMSegmentation 1.2.2 源码，完成官方结构的独立复现：

- 模型：`MobileNetV2 + DepthwiseSeparableASPPHead`，2 类 softmax。
- 训练：scratch、CrossEntropy + `2 x Dice`、AdamW（lr `1e-3`）、10000 iterations、seed 42；水平/垂直翻转、正负 45 度旋转和 `PhotoMetricDistortion`。
- 数据：固定 `dataset_all_filtered` 的 patient-level train/val/test；原始 mask 按项目既有规则 `mask > 127` 转为 MMSeg 所需 0/1 标签，不改动原始数据。
- 选权重：仅在 val 上按 MMSeg `mDice` 选取最佳 checkpoint；最终为 `best_mDice_iter_10000.pth`，val `mDice=87.79`。注意此 mDice 含背景类，不与论文前景 Dice 并排比较。
- 测试：训练完成后才在 436 张 development-test 上评估；统一采用本项目逐图前景 Dice、IoU、Recall、Precision、Specificity、Accuracy、surface HD95、clDice、Boundary F1。

| 模型 | Dice | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Scratch TransUNet | 0.7522 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 |
| Green MSE | 0.7571 | 0.7960 | 0.7398 | 23.28 | 0.8451 | 0.6477 |
| Official DeepLabV3+ | 0.7382 | 0.7196 | 0.7809 | 23.34 | 0.8221 | 0.6214 |
| F0+F3 ensemble | 0.7636 | 0.8053 | 0.7432 | 20.98 | 0.8542 | 0.6483 |

相对 scratch TransUNet，官方 DeepLabV3+ 的 Dice 为 `-0.0140`、Recall 为 `-0.0641`，但 Precision 为 `+0.0381`、HD95 为 `-0.77`。其预测更保守，减少部分误检，却漏掉更多低对比和细小血管；因此没有解决当前“难图低召回”的核心瓶颈，也未超过 green MSE 主线。

该结果证明：在当前统一复核口径下，官方 DeepLabV3+ 并非比 TransUNet 更强；但它**不等同于判定师妹结果错误**，因为她的 split、checkpoint、mask 阈值、后处理和 HD95 实现仍未知。

## 师妹 DeepLabV3+ 与我们现有实现的审计

两边都叫 DeepLabV3+，但不是同一个实验：

| 项目 | 我们已有 SMP 版本 | 师妹提供的 MMSeg 版本 |
|---|---|---|
| 框架 | `segmentation_models_pytorch` | OpenMMLab MMSegmentation 1.2.2 |
| backbone | ResNet34，已汇报结果使用 ImageNet 预训练 | MobileNetV2，配置写明 scratch |
| 输出 | 1 通道 sigmoid | 2 通道 softmax/argmax |
| loss | BCE + Dice | CrossEntropy + 2 x Dice |
| 训练长度 | 50 epochs，patience 20 | 10000 iterations，每 1000 iter 验证 |
| 优化 | AdamW，lr 1e-4，epoch cosine | AdamW，lr 1e-3，iteration cosine |
| 增强 | 水平翻转、正负 15 度、线性亮度/对比度 | 水平+垂直翻转、正负 45 度、PhotoMetricDistortion |
| 选权重 | val 逐图前景 Dice | MMSeg `mDice`，按全验证集/类别聚合且包含背景类 |
| test mask | `>127` | 自定义评估脚本为 `>0` |
| HD95 | 当前统一 surface HD95 | 引用未提供的 `metrics.py`，实现未知 |
| 数据 split | 已核对 `dataset_all_filtered` patient split | 绝对路径指向她电脑的 `UNet_test/train_data`，文件清单未提供 |

我们已有 SMP DeepLabV3+（ResNet34 ImageNet）在当前统一 development-test 上为：Dice `0.7456`、HD95 `23.58`、clDice `0.8341`、Boundary F1 `0.6247`。这只能说明“当前 SMP 配置表现一般”，不能用来否定 MobileNetV2 MMSeg 配置。

师妹的 HD95 15.x 暂时也不能判为错误，但目前不可复核，原因有三：

1. 未提供训练权重、原始 `metrics.py` 和数据文件列表。
2. 当前项目 mask 含 0--255 的抗锯齿灰度边缘；直接交给 MMSeg 二分类训练会产生非法类别，必须明确她是否预先转换成了 0/1。
3. 测试代码以 `mask > 0` 二值化，而我们固定为 `mask > 127`；HD95 对边界定义很敏感，阈值与实现不同会直接改变数值。

因此组会上建议表述为：**我们的统一评估是当前可复核口径；师妹结果是重要线索，但在取得她的 split 文件清单、0/1 mask、checkpoint 和 HD95 函数前，不能认定她一定更优，也不能认定她算错。**

官方 10k 训练已完成。下一步优先向师妹索取四项材料：`train/val/test` 文件名、实际训练 mask 的 unique values、best checkpoint、原始 `metrics.py`。拿到后先用我们的 436 张 development-test 和统一 metrics 直接复评她的 checkpoint，信息量高于继续重复训练。

## 一句话汇报

> 可视化说明 green prior 的提升主要来自修复低对比困难图的漏检，而非均匀提升全部样本；官方 DeepLabV3+ 同口径复现未超过 TransUNet，进一步说明当前核心矛盾是低可见性漏检与纹理伪影误检并存。下一步优先复评师妹 checkpoint、审计困难患者和探索可靠性融合，而不是继续单纯替换分割框架。
