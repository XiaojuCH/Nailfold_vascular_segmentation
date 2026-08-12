# 2026-08-07 组会代表性可视化图

本目录从 `results/prediction_error_review_20260730/all_cases/` 的 436 张 development-test 多模型对照图中筛选。每张面板均包含原始 RGB、GT 与各模型的 TP/FP/FN 叠加：绿色为 TP，红色为 FP，蓝色为 FN。

## 01_error_patterns

| 文件 | 组会要点 | 建议讲法 |
|---|---|---|
| `01_low_contrast_fn_ANFC_001302.jpg` | 低对比、大范围连续血管漏检 | Scratch baseline Dice 0.257，Green MSE 提升到 0.414，但仍有大面积 FN。当前上限首先受可见性和连续性制约。 |
| `02_texture_reflection_fp_ANFC_000389.jpg` | 背景纹理或反光被误判为血管 | Baseline Dice 0.348，Precision 0.222、Recall 0.812；模型并非只会漏检，也会把强纹理当成血管。 |

## 02_green_mse

| 文件 | 组会要点 | 建议讲法 |
|---|---|---|
| `01_green_gain_ANFC_001324.jpg` | green prior 的典型正收益 | Scratch baseline Dice 0.388，Green MSE Dice 0.678；绿色强度先验可恢复部分暗细血管。 |
| `02_green_regression_ANFC_001161.jpg` | green prior 并非逐图稳定获益 | Scratch baseline Dice 约 0.675，Green MSE 约 0.399；必须同时展示成功和失败样例，避免把平均增益误读为所有病例均改善。 |

## 03_method_tradeoffs

| 文件 | 组会要点 | 建议讲法 |
|---|---|---|
| `01_ensemble_complementarity_ANFC_000449.jpg` | RGB 语义和绿色形态先验存在互补 | F0+F3 ensemble Dice 0.854；这是双专家互补的直观证据，但不是可直接部署的单模型结果。 |
| `02_decoder_kd_regression_ANFC_001305.jpg` | 当前 decoder KD V2 产生大块误检 | F0 Dice 0.908，Decoder KD V2 Dice 0.642；裸 decoder feature 一致性会放大 FP，因此已停止作为主线。 |

## PPT 使用建议

1. 一页放 `01_error_patterns` 两张，说明漏检和误检是两类不同问题。
2. 一页并列 `02_green_mse` 两张，讲清 green prior 的收益边界。
3. 一页放 `03_method_tradeoffs` 两张，说明为什么保留双专家互补线索、停止当前 decoder KD。
4. 图像原始总览仍保留在 `results/prediction_error_review_20260730/index.html`，本目录只用于汇报选图。
