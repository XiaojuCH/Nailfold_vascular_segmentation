# 统一训练脚本使用指南

## 功能

`train_unified.py` 是一个统一的训练脚本，支持两种模式：

### 1. Baseline 模式
纯 TransUNet，不使用任何增强模块

### 2. Ours 模式
完整方法：Enhancer + TransUNet + 联合蒸馏

## 使用方法

### Baseline 训练
```bash
python train_unified.py --mode baseline --batch_size 2 --epochs 50
```

### Ours 训练
```bash
python train_unified.py --mode ours --batch_size 2 --epochs 50
```

### 调整蒸馏权重（仅 ours 模式）
```bash
python train_unified.py --mode ours --lambda_mse 10.0 --lambda_grad 30.0
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | baseline | 训练模式：baseline 或 ours |
| `--data_dir` | ./dataset_raw_split | 数据集路径 |
| `--save_dir` | ./results/experiments | 结果保存路径 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 2 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--lambda_mse` | 10.0 | MSE 蒸馏权重 |
| `--lambda_grad` | 30.0 | 梯度蒸馏权重 |
| `--pretrained` | model/vit_checkpoint/... | 预训练权重路径 |

## 输出

结果保存在 `results/experiments/{mode}_transunet/`：
- `best_model.pth` - 最优模型权重
- `training_log.txt` - 训练日志

## 消融实验示例

```bash
# 1. Baseline
python train_unified.py --mode baseline

# 2. Ours (完整方法)
python train_unified.py --mode ours

# 3. 不同权重消融
python train_unified.py --mode ours --lambda_grad 20.0
python train_unified.py --mode ours --lambda_grad 40.0
```
