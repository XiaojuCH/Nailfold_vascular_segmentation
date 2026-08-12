# K2 快速使用说明

适用对象：需要把 K2 用于**新的、已经完成患者级划分**的血管分割数据集的同学。完整方法和注意事项见根目录 [README](../README.md)。

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
conda activate pytorch
cd D:\Projects_\JiaBi_new\K2_model
& "D:\anaconda3\envs\pytorch\python.exe" .\code\audit_dataset.py --data_dir "D:\YourDataset"
```

## 2. 一条命令跑完整 K2

```powershell
conda activate pytorch
cd D:\Projects_\JiaBi_new\K2_model
python .\run_k2_pipeline.py `
  --data_dir "D:\YourDataset" `
  --output_root "outputs\your_dataset_seed42" `
  --include_k0_control
```

PS: 以上命令是使用vscode的情况下，如果你们用的是pycharm等，可能会有差异

流程固定为 `F0 RGB 教师 -> F3 绿色形态教师 -> 双教师 soft target -> K0 对照（可选） -> K2 学生 -> val 评估`。训练日志在 `<output_root>/logs/`；每一阶段自己的 `training_log.txt` 在对应的 `F0_seed42/`、`F3_seed42/`、`K0_seed42/`、`K2_seed42/` 目录。Python 会将 stdout 与 stderr 合并写入日志，不依赖 PowerShell。

## 3. 常用参数

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `--data_dir` | 必填 | 数据集根目录。 |
| `--output_root` | `outputs` | 本次实验所有输出；相对路径固定相对 `K2_model`。每个 seed 建议单独目录。 |
| `--seed` | `42` | 随机种子。固定方案后再补 `43,44`。 |
| `--batch_size` | `4` | 显存不足可改为 `2`。 |
| `--f0_epochs` / `--f3_epochs` | `50` / `50` | 两名教师的最大 epoch。 |
| `--k2_epochs` | `30` | K0/K2 微调最大 epoch。 |
| `--patience` / `--k2_patience` | `20` / `10` | val Dice 无提升的早停轮数。 |
| `--include_k0_control` | 关闭 | 增加无 KD 的 K0；正式比较 KD 贡献时应开启。 |
| `--evaluate_test` | 关闭 | 最终才评 test；开发阶段保持关闭。 |
| `--skip_existing` | 关闭 | 中断后安全续跑：仅跳过完整阶段，遇到半成品会停止而不是覆盖。 |

示例：显存不足、只做 val 开发评估：

```powershell
python .\run_k2_pipeline.py --data_dir "D:\YourDataset" --output_root "outputs\seed42_bs2" --batch_size 2 --include_k0_control
```

## 4. 最小提交物

每个实验请保留：`config.json`、`best_model.pth`、`training_log.txt`、`val_per_image.csv`、`dual_teacher_targets/metadata.json`、最终 `evaluation/*/aggregate_metrics.csv` 与 `per_image_metrics.csv`。报告 K2 改进时，优先报告 `K2 - K0`，而不只报告 `K2 - scratch baseline`。本包不含任何已训练的血管分割权重，必须先在自己的数据集训练 F0 和 F3。
