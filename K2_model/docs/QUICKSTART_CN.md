# K2 快速使用说明

适用对象：需要把 K2 用于**新的、已经完成患者级划分**的血管分割数据集的同学。完整背景、历史指标和风险见根目录 [README](../README.md) 与 [K2_handoff.md](K2_handoff.md)。

## 1. 数据准备

```text
<DATA_ROOT>/
  train/images/  train/masks/
  val/images/    val/masks/
  test/images/   test/masks/
```

- 同一 split 的 image 和 mask 必须同名同扩展名，支持 `png/jpg/jpeg/bmp`。
- 标签固定为 `mask > 127` 是血管前景；不要改为 `> 0`。
- 每个 split 的文件名去掉扩展名后也必须唯一，例如不能同时出现 `001.png` 和 `001.jpg`。
- 训练只依据 `val` 选最佳权重，`test` 只在方案固定后评估一次。

先运行审计；成功后再训练：

```powershell
cd D:\Projects_\JiaBi_new\K2_model
& "D:\anaconda3\envs\pytorch\python.exe" .\code\audit_dataset.py --data_dir "D:\YourDataset"
```

## 2. 一条命令跑完整 K2

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd D:\Projects_\JiaBi_new\K2_model
.\scripts\run_k2_pipeline.ps1 `
  -DataDir "D:\YourDataset" `
  -OutputRoot "outputs\your_dataset_seed42" `
  -IncludeK0Control
```

流程固定为 `F0 RGB 教师 -> F3 绿色形态教师 -> 双教师 soft target -> K0 对照（可选） -> K2 学生 -> val 评估`。训练日志放在 `<OutputRoot>/logs/`；每一阶段自己的 `training_log.txt` 在对应的 `F0_seed42/`、`F3_seed42/`、`K0_seed42/`、`K2_seed42/` 目录。PowerShell 会把 stdout/stderr 分开保存再合并到主日志，避免 `tqdm` 进度条写到 stderr 被误当作训练失败。

## 3. 常用参数

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `-DataDir` | 必填 | 数据集根目录。 |
| `-OutputRoot` | `outputs` | 本次实验所有输出；相对路径固定相对 `K2_model`。每个 seed 建议单独目录。 |
| `-Seed` | `42` | 随机种子。固定方案后再补 `43,44`。 |
| `-BatchSize` | `4` | 显存不足可改为 `2`。 |
| `-F0Epochs` / `-F3Epochs` | `50` / `50` | 两名教师的最大 epoch。 |
| `-K2Epochs` | `30` | K0/K2 微调最大 epoch。 |
| `-Patience` / `-K2Patience` | `20` / `10` | val Dice 无提升的早停轮数。 |
| `-IncludeK0Control` | 关闭 | 增加无 KD 的 K0；正式比较 KD 贡献时应开启。 |
| `-EvaluateTest` | 关闭 | 最终才评 test；开发阶段保持关闭。 |
| `-SkipExisting` | 关闭 | 中断后安全续跑：仅跳过完整阶段，遇到半成品会停止而不是覆盖。 |

示例：显存不足、只做 val 开发评估：

```powershell
.\scripts\run_k2_pipeline.ps1 -DataDir "D:\YourDataset" -OutputRoot "outputs\seed42_bs2" -BatchSize 2 -IncludeK0Control
```

## 4. 参考 K2 复评（只针对本项目甲襞数据）

```powershell
.\scripts\evaluate_reference_k2.ps1 `
  -DataDir "D:\Projects_\JiaBi_new\dataset_all_filtered" `
  -Split test
```

这个命令应接近 Dice `0.7609`、HD95 `21.35`。`reference_weights/` 内的 F0/F3 是甲襞数据训练得到的教师，**新数据集不得直接拿它们生成 K2 soft target**；必须在新数据上先重训 F0 与 F3。

## 5. 最小提交物

每个实验请保留：`config.json`、`best_model.pth`、`training_log.txt`、`val_per_image.csv`、`dual_teacher_targets/metadata.json`、最终 `evaluation/*/aggregate_metrics.csv` 与 `per_image_metrics.csv`。报告 K2 改进时，优先报告 `K2 - K0`，而不只报告 `K2 - scratch baseline`。
