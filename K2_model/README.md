# K2 双教师蒸馏

本目录用于在新数据集上从零复现 K2 训练流程。K2 是一个 **输出级双教师蒸馏** 方法：以 RGB 语义教师 F0 和绿色形态教师 F3 的平均概率图，监督一个由 F0 完整权重初始化的 TransUNet 学生。最终推理只保留 K2 学生，教师不参与推理。

首次上手请直接看 [中文快速使用说明](docs/QUICKSTART_CN.md)。推荐入口是纯 Python 的 `run_k2_pipeline.py`，不依赖 PowerShell。

## 1. 方法与边界

```text
RGB image --> F0: pretrained RGB TransUNet (frozen) --------> p_rgb --+
RGB image --> F3: directional green-prior TransUNet (frozen) -> p_green -+--> p_teacher
RGB image --> Student: TransUNet initialized from F0 --------> p_student

p_teacher = 0.5 * p_rgb + 0.5 * p_green
L = BCE-Dice(p_student, GT) + 1.0 * Soft-BCE(p_student, p_teacher)
```

- **F0**：ImageNet21k R50-ViT-B_16 初始化的 RGB TransUNet。
- **F3**：从头训练的 `directional_multiscale` green-prior fusion TransUNet；其输入仍是 RGB，内部取绿通道并用并行 `1x7 / 7x1 / 1x21 / 21x1` 方向卷积分支提取形态先验，在 decoder 三个尺度门控融合。
- **K2**：从 F0 的完整训练权重继续微调，使用 F0+F3 的离线 float16 soft targets。
- **不是**：Green MSE 的 enhancer 图像级先验；也不是之前失败的 decoder feature MSE。

本包只保留 ImageNet21k 的 TransUNet 初始化权重；不包含任何甲襞数据训练得到的 F0、F3 或 K2 权重。每个新数据集都必须重训 F0、F3 和 K2。

## 2. 目录

```text
K2_model/
  code/                 K2 专用模型、训练、软标签、评估与数据审计代码
  reference_weights/    ImageNet21k 初始化权重
  outputs/              本地训练、软标签、日志和评估输出（默认不附带结果）
  docs/                 方法和使用说明
```

## 3. 环境

当前已验证环境：Python `3.8.20`、PyTorch `2.4.1+cu118`、CUDA `11.8`、NumPy `1.24.4`、OpenCV `4.13.0`、SciPy `1.10.1`、scikit-image `0.21.0`、tqdm `4.67.1`。

**请勿使用 Python 3.12+ 或 NumPy 2.x**：`third_party/TransUNet` 官方代码与 SciPy 的兼容性只在 NumPy 1.x 下验证过，NumPy 2.x 会导致部分指标计算报错。

在已安装匹配 CUDA 的 PyTorch 环境中执行：

```powershell
pip install -r .\code\requirements-k2.txt
```

运行前先进入自己的 PyTorch 环境，例如 `conda activate pytorch`。随后所有推荐命令都只需要 `python`，不依赖 PowerShell 执行策略。

如需重新安装 PyTorch，请按本机 CUDA 驱动从 PyTorch 官方安装页选择版本；不要盲目安装 `third_party/TransUNet/requirements.txt` 中已过时的 `torch==1.4.0`。

## 4. 数据格式与强制检查

数据根目录必须严格为：

```text
<DATA_ROOT>/
  train/images/  train/masks/
  val/images/    val/masks/
  test/images/   test/masks/
```

- 同一 split 中 image 和 mask 必须有**完全相同的文件名和扩展名**，支持 `.png/.jpg/.jpeg/.bmp`。
- 图像按 RGB 读取、resize 到 `256 x 256`、归一化到 `[0, 1]`。
- 统一标签规则：`mask > 127` 为前景，其他为背景。原始灰度抗锯齿 mask、`mask > 0`、以及不同的 HD95 实现，都会改变结果，禁止混用。
- 当前指标为逐图前景 macro 平均：Dice、IoU、Recall/Sensitivity、Precision、Specificity、Accuracy、surface-HD95、clDice、Boundary F1。HD95 和 Boundary F1 的单位是 256x256 resize 后的像素。
- 在新数据上请先完成患者级 train/val/test 划分；训练期只能用 val 选择 checkpoint，不要逐 epoch 查看 test。若 test 已参与选方向，应在报告中称为 development-test。

先审计数据：

```powershell
python .\code\audit_dataset.py --data_dir "D:\YourDataset"
```

审计输出中 `masks_with_gray_values_1_to_254` 不为零不一定是错误，但表示必须使用本包的 `>127` 规则，不能自行换成 `>0`。

## 5. 新数据集的完整复现

从 `K2_model` 目录执行：

```powershell
python .\run_k2_pipeline.py --data_dir "D:\YourDataset" --output_root "outputs\your_dataset_seed42" --include_k0_control
```

脚本顺序固定：

1. 审计数据和标签；
2. 训练 F0 RGB 教师，ImageNet21k 初始化；
3. 从头训练 F3 directional multiscale 绿色形态教师；
4. 依次生成 F0/F3 概率图，保存 `train/val` 的 float16 `.npy` soft target；
5. 可选训练 K0 control：从 F0 继续训练，但 `lambda_kd=0`；
6. 训练 K2：从 F0 初始化，`lambda_kd=1.0`；
7. 默认只在 val 上评估。确认模型方案后，增加 `--evaluate_test` 才评 test 一次。

若中途因断电、显存或网络问题中断，在确认已有阶段完整后可在原命令末尾增加 `--skip_existing`。脚本只会跳过同时具备权重、配置和 val 逐图结果的训练阶段；半成品目录不会被自动覆盖。

**重要**：`reference_weights/` 中只有 ImageNet 初始化文件，不是现成的血管分割教师。新数据集必须重训 F0 与 F3，并重新生成 soft targets。

## 6. 分配给师弟时的交付清单

1. 给每人一份独立 `train/val/test` 文件名清单，确保患者不跨 split。
2. 先让其运行 `audit_dataset.py`，把 JSON 结果回传。
3. 首次只跑 `F0 -> F3 -> K2` 的 seed42；不要在未固定数据/指标前随意改 loss、阈值、增强或 backbone。
4. 需要证明 KD 有贡献时，增加 `-IncludeK0Control`，比较 K2 与 K0，而不是只和随机初始化 baseline 比。
5. 汇报必须同时提交 `config.json`、`training_log.txt`、`val_per_image.csv`、soft-target `metadata.json`、最终 `aggregate_metrics.csv`。
6. 在同一数据集最终选定方案后，再补 seed43/44、患者级 paired CI，并做独立 test 或 outer-CV。

## 7. 代码入口

| 文件 | 用途 |
|---|---|
| `code/train_k2.py` | F0、F3、K2 三种训练阶段；最佳 checkpoint 仅由 val Dice 选择。 |
| `code/generate_dual_teacher_targets.py` | 两教师顺序占用 GPU，生成不量化成 PNG 的 float16 概率图。 |
| `code/evaluate_k2.py` | 统一模型评估，输出 aggregate 和 per-image CSV。 |
| `code/audit_dataset.py` | 训练前检查目录、文件名、mask 灰度取值。 |
| `run_k2_pipeline.py` | 推荐：跨平台完整复现主入口。 |

详见 [docs/K2_handoff.md](docs/K2_handoff.md)。
