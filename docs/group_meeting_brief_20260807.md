# 2026-08-07 组会简要汇报：从测试图像复盘模型瓶颈与评估口径

## 一、本周完成工作

1. 对固定 `dataset_all_filtered/test` 的 436 张图像完成 9 个代表性模型的逐图预测、TP/FP/FN 可视化与排序复盘。
2. 使用官方 MMSegmentation 1.2.2 的 DeepLabV3+ 独立训练并按本项目统一协议复评。
3. 审计师妹提供的 DeepLabV3+ 配置和评估脚本，定位到数据路径、标签二值化和 HD95 实现不可复核等关键差异。

说明：该 436 张集合已参与多轮研发决策，以下称为 **development-test**；它适合诊断和方向筛选，论文最终主表仍需外层患者级 CV 或从未参与研发的 holdout。

## 二、从图像中看到的模型瓶颈

代表图已单独整理到 `results/group_meeting_20260807_figures/`，可直接用于制作 PPT。图中绿色为 TP、红色为 FP、蓝色为 FN。若只讲最佳单模型，请优先使用 `04_k2_soft_kd_focus/`：它只保留 `RGB / GT / Pretrained F0 / K2 soft KD` 四列，并同时给出 K2 的改善、退化和仍未解决的困难图。

| 观察到的问题 | 代表图 | 结论 |
|---|---|---|
| 低对比、暗细或连续血管漏检 | `01_error_patterns/01_low_contrast_fn_ANFC_001302.jpg` | 大范围 FN 同时拉低 Recall 和 HD95；这是当前最核心的困难子群。 |
| 纹理、反光和非血管边缘误检 | `01_error_patterns/02_texture_reflection_fp_ANFC_000389.jpg` | 模型不能只追求 Recall；提高结构/召回往往会以 Precision 下降为代价。 |
| green prior 的典型成功样例 | `02_green_mse/01_green_gain_ANFC_001324.jpg` | baseline Dice 0.388 到 Green MSE 0.678，说明绿色强度先验对低可见性血管有补偿价值。 |
| green prior 的退化样例 | `02_green_mse/02_green_regression_ANFC_001161.jpg` | baseline Dice 约 0.675 到 Green MSE 约 0.399，说明该先验并非每张图稳定有效。 |
| 双专家互补 | `03_method_tradeoffs/01_ensemble_complementarity_ANFC_000449.jpg` | RGB 语义和 green morphology 有局部互补，但 ensemble 仍是双模型推理。 |
| decoder KD V2 退化 | `03_method_tradeoffs/02_decoder_kd_regression_ANFC_001305.jpg` | 当前裸 decoder feature 一致性会产生明显 FP，不宜继续叠加该设计。 |

补充量化结论：在 baseline Recall 小于 0.6 的 37 张困难图中，Green MSE 有 30 张改善，平均 Dice 增益为 `+0.0459`；在其余 353 张相对容易图中平均 Dice 变化约为 `+0.00005`。因此，green prior 的准确叙事是“补偿低可见性困难样本”，而不是“对所有图像均匀提升”。

## 三、统一口径的核心结果

固定：patient-level split、256 x 256、标签 `mask > 127`、阈值 0.5、逐图前景 macro 平均、surface-HD95、clDice、Boundary F1。

| 模型 | Dice | Recall | Precision | HD95 | clDice | Boundary F1 | 相对 Scratch TransUNet Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch TransUNet | 0.7522 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 | baseline |
| Green MSE (seed 42) | 0.7571 | 0.7960 | 0.7398 | 23.28 | 0.8451 | 0.6477 | +0.0050 |
| Green MSE (scratch, 3 seeds) | 0.7584 +/- 0.0011 | - | - | 23.42 | 0.8467 | 0.6468 | +0.0066 |
| Official DeepLabV3+ | 0.7382 | 0.7196 | 0.7809 | 23.34 | 0.8221 | 0.6214 | -0.0140 |
| F0+F3 probability ensemble | 0.7636 | 0.8053 | 0.7432 | 20.98 | 0.8542 | 0.6483 | 不适用，双模型 |

结论：Green MSE 是当前最稳定、可解释的单模型小幅改善；F0+F3 集成展示了互补上限，但还不是最终可部署单模型。当前要解决的是“难图漏检”和“纹理误检”并存，而非继续无差别堆叠模块。

## 四、官方完整 DeepLabV3+ 复现

为排除原先 SMP DeepLabV3+ 不是官方实现的影响，本轮使用官方 MMSegmentation 1.2.2 实现：MobileNetV2 + `DepthwiseSeparableASPPHead`、二类 softmax、从头训练、CrossEntropy + 2 x Dice、AdamW、10000 iterations、seed 42。仅按 val 的 `mDice` 选择 checkpoint，训练完成后才在 436 张 development-test 上评估。

| Dice | IoU | Recall | Precision | Specificity | Accuracy | HD95 | clDice | Boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7382 | 0.5956 | 0.7196 | 0.7809 | 0.9755 | 0.9488 | 23.34 | 0.8221 | 0.6214 |

相对 Scratch TransUNet：Dice `-0.0140`、Recall `-0.0641`，但 Precision `+0.0381`、HD95 `-0.77`。即官方 DeepLabV3+ 更保守，减少部分 FP，但会漏掉更多细小或低对比血管；在当前统一协议下未超过 TransUNet 或 Green MSE。

注意：MMSeg val `mDice=87.79` 包含背景类，不能与表中逐图前景 Dice 并排比较。

## 五、师妹代码与本项目的审计对比

| 项目 | 本项目统一协议 | 师妹提供代码 | 当前判断 |
|---|---|---|---|
| 数据与 split | `dataset_all_filtered` patient-level split，文件数和患者无交叉已核验 | 使用绝对路径 `C:\Users\33101\Desktop\UNet_test\train_data`，未提供文件清单 | 不知道是否为同一份数据或同一 split。 |
| 标签二值化 | 原始灰度 mask 统一按 `>127` 转为 0/1 | 测试脚本按 `mask > 0` 转为 0/1 | 两种边界定义不同；当前数据有抗锯齿灰度边缘，指标会明显改变。 |
| 模型输出 | TransUNet 为单通道 sigmoid | DeepLabV3+ 为二通道 softmax/argmax | 输出形式不同本身合理，但必须在同一 GT 和指标下比较。 |
| 最佳权重选择 | val 的逐图前景 Dice | MMSeg val `mDice`，包含背景类 | 选择规则不同，不能把训练日志数值直接横比。 |
| HD95 | 本项目 `utils/metrics.py` 的 surface-HD95 | 调用未提供的 `metrics.py` | 她报告的 HD95 15.x 目前无法复核。 |
| 测试流程 | 统一入口 `evaluate_all.py`，固定阈值与逐图 macro | CPU 手工 `encode_decode`、手工归一化和 argmax | 不等于错误，但和本项目协议不同。 |

严谨结论：不能说“我们的数值绝对正确、师妹一定错”，也不能把她的 HD95 15.x 直接并入当前表格。本项目当前的结果是 **内部一致、可复核** 的；师妹的结果是重要线索，但复现信息尚不充分。

需要向师妹索取：`train/val/test` 文件名清单、实际训练 mask 的 unique values 或处理脚本、best checkpoint、原始 `metrics.py`、是否使用后处理或阈值优化。拿到后优先用本项目 436 张 development-test 和统一指标复评其 checkpoint。

## 六、组会结论与下一步

1. 先做困难患者和标注质量审计，确认低对比漏检是否存在采集条件、染色、曝光或标注一致性问题。
2. 获取师妹 checkpoint 与完整评估信息，先统一复评，再讨论 HD95 15.x 的来源。
3. 后续强 baseline 应选择 nnU-Net v2 或 MedNeXt，并坚持相同 split、标签与统一评估；不再为了“换模型”重复无效训练。
4. 方法优化聚焦可靠性：对低对比区域增强召回、对纹理/反光区域抑制 FP，而不是继续堆叠普通注意力、裸特征蒸馏或高权重结构损失。

## 可直接口头汇报

> 这周我们没有只看平均指标，而是把 436 张 development-test 的预测逐图拆开看。结论是当前瓶颈同时包括低对比连续血管的漏检和纹理反光的误检；green prior 对前一类困难图有明确补偿，但不是每张图都会提高。官方 DeepLabV3+ 在统一口径下没有超过 TransUNet，师妹的 HD95 15.x 目前不能直接比较，因为数据划分、标签阈值和 HD95 实现还没对齐。下一步优先统一复评她的 checkpoint，并从困难患者和标注上限入手，而不是盲目继续加模块。
