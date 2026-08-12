# 甲襞分割预测可视化复盘说明

## 目的

本轮不继续调模型，而是对已有代表模型做统一的 development-test 全量预测，回答：

1. 模型在哪些图像上共同失败，困难是否来自低对比度、血管面积小、断裂或背景伪影。
2. green prior 的提升来自修复漏检、减少误检，还是只改善少量病例。
3. F3、结构辅助、decoder distillation 和双教师 ensemble 分别改变了什么错误模式。
4. 后续应优先改模型、数据增强、后处理，还是检查标注上限。

## 代表模型

配置固定在 `docs/prediction_visualization_manifest_20260730.json`。默认包含：

- scratch TransUNet 与 green-MSE Ours：复核最稳定的 green prior 增益。
- scratch F3 directional：复核 HD95/Boundary F1 信号来自哪些图像。
- pretrained F0、K2 soft-KD 与 F0+F3 ensemble：观察双专家互补和单模型蒸馏损失了什么。
- S0/S2：直接观察 boundary+centerline 是否导致过分割。
- DecoderDistillV2：作为负结果，观察其退化是普遍变差还是集中失败。

不同训练协议的模型不直接按绝对 Dice 排名。正式结论只使用 manifest 中明确写出的 control/candidate 配对。

## 运行

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_prediction_error_review_20260730.ps1
```

若推理已经完成、只需要重新生成图表和报告：

```powershell
.\scripts\run_prediction_error_review_20260730.ps1 -ReusePredictions
```

如果暂时不想生成 436 张横向对照大图：

```powershell
.\scripts\run_prediction_error_review_20260730.ps1 -ReusePredictions -SkipAllCasePanels
```

## 输出

输出目录为 `results/prediction_error_review_20260730`：

| 路径 | 内容 |
|---|---|
| `index.html` | 可视化总览、典型病例画廊、全部 436 张病例索引 |
| `analysis_summary.md` | 可直接用于组会或下一轮实验讨论的中文摘要 |
| `predictions/<model>` | 每个模型对全部测试图的二值预测 PNG |
| `probability_cache` | float16 概率缓存，用于无须重推理地重画报告 |
| `all_cases` | 每张图的 RGB、GT 和代表模型 TP/FP/FN 横向对照 |
| `rankings` | 最好/最差、FP/FN、HD95、模型改善/退化和分歧病例 CSV |
| `per_image_metrics.csv` | 所有模型逐图完整指标与图像诊断特征 |
| `patient_metrics.csv` | 按患者聚合的指标 |

误差图颜色固定：绿色为 TP，红色为 FP，蓝色为 FN；GT 面板使用青色轮廓。

## 解读原则

- 同时检查 `wins` 和 `losses`，不能只挑 Ours 更好的图。
- S2 若主要表现为 Recall 上升、红色 FP 增多，说明结构辅助在当前协议下倾向过分割。
- ensemble 若能在 F0 漏检处补出连续血管、而 K2 没有保留，下一步应研究可靠性融合或局部不确定性蒸馏，而不是继续做裸 feature MSE。
- 若所有模型在同一批图上共同失败，应优先检查原图可见性和标注一致性；这类病例可能代表数据/标注上限，而不是单纯网络容量不足。
- 当前 `test` 是 development-test。论文主表仍需患者级 outer-CV 或未参与研发的 final holdout。
