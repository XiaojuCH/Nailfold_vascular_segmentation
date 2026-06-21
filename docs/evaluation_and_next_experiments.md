# 统一复评与下一轮实验说明

更新日期：2026-06-20  
主数据集：`dataset_all_filtered/test`  
统一口径：`img_size=256`，`threshold=0.5`，同一套 `utils/metrics.py` 指标实现。

## 1. 当前结论

这次重训后的 TransUNet baseline 已经恢复到合理水平，可以作为后续论文表格的 baseline：

| 模型 | 权重 | Dice | IoU | Recall | Precision | Specificity | Accuracy | HD95 | clDice | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransUNet retrain | `results/experiments/all_filtered/baseline_retrain_20260619/0619_0232/best_model.pth` | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 0.9670 | 0.9482 | 24.11 | 0.8403 | 0.6405 |
| Ours green only | `results/experiments/all_filtered/ours_green_only/best_model.pth` | 0.7567 | 0.6184 | 0.8156 | 0.7230 | 0.9622 | 0.9473 | 23.26 | 0.8440 | 0.6348 |

相对重训 TransUNet，当前主线 `Ours green only` 的变化是：

| 指标 | 差值 | 判断 |
|---|---:|---|
| Dice | +0.0045 | 小幅提升 |
| IoU | +0.0045 | 小幅提升 |
| Recall | +0.0318 | 明显提升，说明更少漏检细血管 |
| Precision | -0.0199 | 下降，说明误检增加 |
| Specificity | -0.0048 | 下降，背景抑制略弱 |
| Accuracy | -0.0009 | 基本持平，略低 |
| HD95 | -0.85 | 略好，越低越好 |
| clDice | +0.0037 | 小幅提升 |
| Boundary F1 | -0.0056 | 略低 |

核心判断：当前方法确实有收益，但收益主要来自“召回更多血管”和“结构指标略好”，不是全面碾压 baseline。论文叙事应避免写成所有指标都优于 baseline，更适合写成：green prior distillation 让模型更偏向保留细小血管结构，提升 Recall、Dice、IoU、clDice 和 HD95，但会带来一定 Precision/Specificity/Boundary F1 代价。

## 2. 剔除异常权重后的横向结果

下面这张表剔除了两个已确认异常的旧权重：旧 `TransUNet baseline` 和旧 `Ours green+CLAHE`。

| 实验 | Dice | 相对新 TransUNet Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours green only multiscale | 0.7569 | +0.0048 | 0.6192 | 0.7742 | 0.7589 | 23.28 | 0.8459 | 0.6489 |
| Ours green only | 0.7567 | +0.0045 | 0.6184 | 0.8156 | 0.7230 | 23.26 | 0.8440 | 0.6348 |
| Ours CLAHE only | 0.7552 | +0.0030 | 0.6165 | 0.8063 | 0.7297 | 24.03 | 0.8427 | 0.6354 |
| Ours green only learnable loss | 0.7552 | +0.0031 | 0.6166 | 0.8053 | 0.7283 | 23.67 | 0.8414 | 0.6337 |
| Ours green only gated | 0.7544 | +0.0022 | 0.6160 | 0.8166 | 0.7209 | 24.52 | 0.8418 | 0.6348 |
| Ours green only inverse attention | 0.7531 | +0.0009 | 0.6150 | 0.8116 | 0.7207 | 24.63 | 0.8390 | 0.6301 |
| TransUNet retrain | 0.7522 | +0.0000 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 |
| UNet++ | 0.7484 | -0.0038 | 0.6082 | 0.7291 | 0.7897 | 23.49 | 0.8391 | 0.6456 |
| UNet | 0.7374 | -0.0147 | 0.5962 | 0.7189 | 0.7851 | 24.58 | 0.8231 | 0.6351 |

注意：`Ours green only multiscale` 的 Dice、IoU、clDice、Boundary F1 目前略高，但它只比主线 `Ours green only` 高 `0.0003 Dice`。这个差距太小，不建议马上把 multiscale 改成主线。更稳妥的说法是：multiscale 是一个候选分支，需要二次复现或多 seed 后再决定是否替换主线。

## 3. 逐图配对分析

配对对象：

- Baseline：`TransUNet_retrain_20260619_per_image.csv`
- Ours：`Ours_green_only_per_image.csv`
- 测试图数：436

逐图差值采用 `Ours green only - TransUNet retrain`。HD95 越低越好，所以 HD95 差值为负代表 Ours 更好。

| 指标 | 平均差值 | 近似 95% CI | 中位差值 | 提升图数 | 退化图数 | 解读 |
|---|---:|---:|---:|---:|---:|---|
| Dice | +0.00448 | [0.00018, 0.00878] | +0.00024 | 220 | 216 | 均值略升，但逐图胜负几乎对半 |
| IoU | +0.00445 | [-0.00021, 0.00912] | +0.00032 | 220 | 216 | 趋势略好，但不够强 |
| Recall | +0.03182 | [0.02600, 0.03764] | +0.02277 | 334 | 101 | 最稳定的正向收益 |
| Precision | -0.01988 | [-0.02490, -0.01486] | -0.01924 | 122 | 314 | 误检增加很明确 |
| Specificity | -0.00477 | [-0.00586, -0.00369] | -0.00358 | 106 | 330 | 背景像素误检增加 |
| Accuracy | -0.00093 | [-0.00179, -0.00007] | -0.00114 | 182 | 252 | 略低 |
| HD95 | -0.85244 | [-2.54511, 0.84023] | 0.00000 | 203 | 201 | 平均略好，但波动大 |
| clDice | +0.00371 | [-0.00132, 0.00875] | +0.00180 | 226 | 210 | 略好，但不稳 |
| Boundary F1 | -0.00565 | [-0.01068, -0.00062] | -0.00754 | 183 | 253 | 边界贴合略差 |

这个结果说明：主线方法不是在每张图上都明显更好，而是稳定提高 Recall，同时牺牲 Precision。后续论文图例应该重点找两类样本：

1. 主线成功召回 baseline 漏掉的细血管样本，用来支撑方法动机。
2. 主线误检变多或边界变粗的失败样本，用来解释 Precision/Boundary F1 下降。

## 4. 已确认异常权重

以下两个旧权重不要进入正式论文主表：

| 权重 | 旧日志结果 | 统一复评结果 | 判断 |
|---|---:|---:|---|
| `results/experiments/all_filtered/baseline/best_model.pth` | Dice 0.7531 | Dice 0.5928 | 不可信，已由重训 baseline 替代 |
| `results/experiments/all_filtered/ours/best_model.pth` | Dice 0.7537 | Dice 0.5295 | 不可信，若需要 green+CLAHE 必须重训 |

已经做过的排查：

1. `evaluate_all.py` 单独复评旧 TransUNet，仍为 Dice 0.5928。
2. `evaluate_with_cldice.py` 单独复评同一旧 TransUNet，仍为 Dice 0.5928。
3. 旧权重可以 strict load，没有 missing/unexpected key。
4. 未发现明显的 TransUNet 文件覆盖证据。
5. 旧 `Ours green+CLAHE` 同样无法复现旧日志结果。

当前处理策略：保留异常记录，但正式结果以 `docs/unified_eval_manifest_all_filtered.json` 中的新权重 registry 为准。

## 5. 统一复评入口

当前固定权重 registry：

`docs/unified_eval_manifest_all_filtered.json`

推荐复评命令：

```powershell
D:\anaconda3\envs\pytorch\python.exe evaluate_all.py --manifest docs\unified_eval_manifest_all_filtered.json --dataset all_filtered --split test --threshold 0.5 --batch_size 4
```

输出目录：

`results/unified_eval/<运行时间>/`

输出文件：

| 文件 | 作用 |
|---|---|
| `aggregate_results.csv` | 所有模型统一指标总表 |
| `aggregate_results.xlsx` | Excel 版本总表，方便写论文表格 |
| `<实验名>_per_image.csv` | 每张图的逐图指标，用于 paired delta、bootstrap CI、失败样例筛选 |

指标包括：

`Dice/F1, IoU, Recall/Sensitivity, Precision, Specificity, Accuracy, surface-HD95, clDice, Boundary F1`

代码层面已经注意两点：

1. `evaluate_all.py` 的完整摘要会打印所有核心指标。
2. per-image 文件名对 `UNet++` 这类名字做了唯一化处理，避免 `UNet` 与 `UNet++` 文件名碰撞。

## 6. 下一步实验优先级

### P0：先补 direct input 对照

最重要的问题不是继续堆模块，而是证明当前提升不是“直接把 green/CLAHE 图喂给 TransUNet”就能得到。

建议新增训练入口：

- `--input_variant original|green_only|clahe_only|green_clahe`
- 默认 `original`，保持现有训练不变。
- direct input baseline 只训练普通 TransUNet，不启用 enhancer/distillation。
- mask 仍使用原始 `masks`。

需要跑的 direct input 对照：

| 实验 | 目的 |
|---|---|
| TransUNet direct green input | 验证绿通道直接输入是否已经足够 |
| TransUNet direct CLAHE input | 验证 CLAHE 直接输入是否已经足够 |
| TransUNet direct green+CLAHE input | 对照当前 prior/distillation 设计 |

如果 direct input 已经接近或超过 `Ours green only`，那当前创新故事会变弱；如果 direct input 不如当前方法，就能支撑“不是简单预处理，而是先验蒸馏/联合学习有效”。

### P1：补 loss 消融

当前主线是：

```text
teacher_mode: green_only
loss_weighting: fixed
lambda_mse: 10.0
lambda_grad: 30.0
```

必须补的消融：

| 实验 | 参数 | 目的 |
|---|---|---|
| MSE only | `lambda_mse=10, lambda_grad=0` | 验证 gradient consistency 是否有额外贡献 |
| lower grad | `lambda_mse=10, lambda_grad=20` | 看是否能减少误检、改善 Precision/Boundary F1 |
| higher grad | `lambda_mse=10, lambda_grad=40` | 看是否进一步改善 HD95/clDice |

可直接跑的命令：

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 0.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --exp_name all_filtered/ours_green_only_mse_only
```

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 20.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --exp_name all_filtered/ours_green_only_mse10_grad20
```

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green_only --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 40.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --exp_name all_filtered/ours_green_only_mse10_grad40
```

训练后用单模型复评：

```powershell
D:\anaconda3\envs\pytorch\python.exe evaluate_all.py --name ours_green_only_mse_only --model_type ours --weight results\experiments\all_filtered\ours_green_only_mse_only\<timestamp>\best_model.pth --dataset all_filtered --split test --threshold 0.5 --teacher_mode green_only --enhancer basic --joint_model v1 --lambda_mse 10.0 --lambda_grad 0.0
```

### P1：是否重训 green+CLAHE

旧 `Ours green+CLAHE` 权重不可信。如果论文消融表需要 `green+CLAHE teacher`，建议重训一版：

```powershell
D:\anaconda3\envs\pytorch\python.exe train_unified.py --mode ours --dataset all_filtered --teacher_mode green+clahe --joint_model v1 --enhancer basic --loss_weighting fixed --lambda_mse 10.0 --lambda_grad 30.0 --epochs 50 --patience 20 --batch_size 4 --lr 1e-4 --exp_name all_filtered/ours_green_clahe_retrain_20260620
```

优先级低于 direct input 和 loss 消融，因为现有 `CLAHE only` 与 `green only` 已经能说明一些 teacher prior 差异。

### P2：阈值和结构损失

如果前面实验后提升仍然很小，再考虑：

| 实验 | 说明 |
|---|---|
| validation threshold selection | 在 val 上统一选阈值，再固定到 test；不能在 test 上调阈值 |
| soft-clDice loss branch | 作为结构损失分支尝试，不替换当前主线 |
| 多 seed 或重训主线 | 用于判断 +0.0045 Dice 是否稳定 |

## 7. 当前推荐路线

1. 用更新后的 manifest 重新跑一次完整复评，得到包含新 TransUNet baseline 的正式总表。
2. 实现 direct input baseline 入口，先跑 direct green、direct CLAHE、direct green+CLAHE。
3. 跑 `MSE only`、`grad20`、`grad40` 三个 loss 消融。
4. 从 per-image CSV 里筛选主线相对 baseline 改善最大/退化最大的病例，准备论文可视化。
5. 如果 direct input 对照能证明当前方法确实不是简单预处理，再考虑是否重训 green+CLAHE 或进一步尝试 soft-clDice。
6. 暂时不要继续优先投入 gated、inverse attention、learnable loss 这几条，它们目前没有显示出足够稳定的收益。

## 8. 论文表述建议

目前可以写的方向：

- green-channel physical prior distillation improves the detection of subtle capillary structures.
- The method increases vessel sensitivity and slightly improves Dice/IoU/clDice/HD95 under a unified evaluation protocol.
- The gain is modest and comes with a precision trade-off, so direct preprocessing controls and loss ablations are necessary.

目前不建议写的方向：

- 不建议写“全面优于所有 baseline”。
- 不建议使用旧 `baseline/best_model.pth` 或旧 `ours/best_model.pth` 的结果。
- 不建议把 `multiscale` 直接定为最终主线，除非再做复现实验。
