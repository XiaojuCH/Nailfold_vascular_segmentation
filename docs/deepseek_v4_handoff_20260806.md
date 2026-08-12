# 甲襞毛细血管分割项目交接文档（供 DeepSeek V4 / Claude 接手）

> 最后更新：2026-08-06
>
> 工作目录：`D:\Projects_\JiaBi_new`
>
> 近期用途：2026-08-07 晚组会。原定 7 月 30 日汇报已延期；当前组会主材料是 `docs/group_meeting_prediction_review_20260730.md`。

## 0. 给接手 agent 的首要指令

你接手的是一个甲襞毛细血管二值分割研究项目。当前最重要的任务不是立刻堆叠新模块或反复跑分，而是：

1. 先读本文件的“必读顺序”和组会材料，再提出/执行后续工作。
2. 把 `dataset_all_filtered/test` 统一称为 **development-test**，不能把它写成论文最终独立测试集。
3. 训练选权重只能使用 `val`；不能按 development-test 指标逐步挑模型或阈值。
4. 所有横向比较必须固定：patient-level split、mask 阈值、预测阈值、图像大小和指标实现。不同训练协议只能比较同协议对照的 delta，不能只按绝对 Dice 排名。
5. 工作区是脏的，存在用户和前任 agent 未提交的改动/结果。不要执行 `git reset --hard`、`git checkout --`、删除 `results/`、删除 `TYT_Code/`，也不要覆盖已有权重。
6. 当前已得到许多负结果。不要把“再加 attention/strip convolution/loss/KD”当默认方案；先证明该改动能对应可视化中明确的失败模式，并预先写出停止条件。

建议先向用户汇报你读完后确认的三件事：数据和评估口径、最可信的正结果、已经停止的方向；再讨论下一轮实验。

## 1. 必读顺序（先读这些，不要一开始遍历全部 results）

| 优先级 | 文件/目录 | 为什么读 |
|---|---|---|
| P0 | `docs/group_meeting_prediction_review_20260730.md` | 8 月 7 日组会主材料：可视化结论、官方 DeepLabV3+ 复现、当前叙事。 |
| P0 | 本文件 `docs/deepseek_v4_handoff_20260806.md` | 项目记忆、边界和接手规则。 |
| P0 | `results/prediction_error_review_20260730/analysis_summary.md` | 436 张 development-test 的逐图复盘摘要。 |
| P0 | `results/prediction_error_review_20260730/index.html` | 最重要的可视化入口；看共同失败、green MSE 最大改善/退化、ensemble、Decoder KD V2。 |
| P0 | `docs/prediction_error_review_guide.md` | 解释 HTML/CSV/误差颜色和重生成命令。 |
| P1 | `docs/group_meeting_plan_20260721.md` | 7 月 10--21 日完整实验档案、三 seed 结果、CGMA 的严格停止结论。 |
| P1 | `docs/official_deeplabv3plus_audit_20260730.md` | 师妹 MMSeg DeepLabV3+ 与本项目的口径差异和已完成复现。 |
| P1 | `docs/prediction_visualization_manifest_20260730.json` | 代表模型、准确权重路径、对照配对。 |
| P1 | `evaluate_all.py`、`utils/metrics.py`、`datasets/dataset_vessel.py` | 统一评估、指标和数据读取的事实来源。 |
| P2 | `train_unified.py`、`models/transunet_official.py` | 常规 TransUNet 训练入口和模型接口。 |
| P2 | `models/green_prior_fusion.py`、`models/joint_framework.py`、`models/compact_green_morphology.py` | 已实现但大多已停止/仅作历史对照的模型方向。 |

不要将旧的组会文档当作最新结论覆盖来源。`docs/group_meeting_summary_20260710.md`、`docs/group_meeting_plan_20260717.md` 和更早文件保留历史，不是当前主结论。

## 2. 项目目录与运行环境

```text
D:\Projects_\JiaBi_new
|- dataset_all_filtered/                 # 当前固定主数据集，patient-level split
|- dataset_all_filtered_mmseg/           # 给官方 MMSeg 用的非破坏性 0/1 mask 视图
|- datasets/dataset_vessel.py            # 常规数据读取、增强、mask > 127
|- models/
|  |- transunet_official.py               # 当前主分割器
|  |- joint_framework.py                  # green-image enhancer / 早期 joint variants
|  |- green_prior_fusion.py               # F0--F3 directional prior fusion
|  `- compact_green_morphology.py         # CGMA，已完成反证性实验
|- losses/joint_loss.py                   # segmentation、KD、结构辅助损失
|- utils/metrics.py                       # 唯一的项目统一指标实现
|- train_unified.py                       # 统一训练入口
|- evaluate_all.py                        # 统一顺序评估入口，支持 manifest
|- analyze_prediction_errors.py           # 全量预测/可视化/失败病例复盘
|- configs/                               # 官方 MMSeg 可移植配置
|- dataset_tools/                         # 数据准备工具
|- scripts/                               # 顺序实验、复盘、MMSeg 运行脚本
|- results/                               # 权重、日志、统一评估和可视化输出
|- docs/                                  # 本项目记忆和组会材料
|- TYT_Code/                              # 师妹提供代码和 MMSegmentation 1.2.2 源码，用户文件
`- third_party/                           # TransUNet 与原始 ANFC 映射等第三方资产
```

### Python 环境

| 用途 | 解释器/环境 | 说明 |
|---|---|---|
| 常规 TransUNet、评估、可视化 | `D:\anaconda3\envs\pytorch\python.exe` | Torch `2.4.1+cu118`；不要在此环境安装完整 MMCV/MMSeg。 |
| 官方 MMSeg DeepLabV3+ | `D:\anaconda3\envs\mmseg_official\python.exe` | 隔离环境；Torch `2.1.2+cu118`、MMCV `2.1.0`、MMEngine `0.10.7`、MMSeg `1.2.2`。 |

官方 MMSeg 的两个 Windows 兼容修复已经写入脚本，不要删除：

- `scripts/setup_official_mmseg_env.ps1` 固定 `setuptools<81`，保证 Torch 2.1 能导入 `pkg_resources`。
- `scripts/run_official_deeplabv3plus_20260730.ps1` 设置 `PYTHONUTF8=1`，避免中文 Windows 上 MMEngine 读取 MSVC 版本时发生 GBK 解码失败。
- `configs/deeplabv3plus_mobilenetv2_official_all_filtered.py` 固定随机 seed，但 `deterministic=False`；MMSeg IoU 验证用 CUDA `histc`，PyTorch 2.1 不支持其 deterministic 模式。

## 3. 数据、split 与评估协议（不可随意改）

### 数据集

主数据为 `dataset_all_filtered`：

| split | 图像数 | 用途 |
|---|---:|---|
| `train` | 1838 | 训练和训练期增强 |
| `val` | 449 | 选 checkpoint、预先定义的模型决策和阈值选择 |
| `test` | 436 | 当前称 development-test；只用于阶段性统一诊断，不可再当最终论文 test |

- split 由患者级划分，三个 split 的患者没有重叠。恢复患者 ID 使用 `third_party/ANFC_OURS_All_dataset/backup_original_names/rename_mapping.txt`。
- 当前没有把 test 重新划入训练，也没有把整个数据集重新划分。此前已审计并记录在 `docs/group_meeting_plan_20260717.md`。
- 原始 RGB 和 mask 都为 PNG，训练/评估统一 resize 到 `256 x 256`。
- 原始 mask 不是纯 0/255：有大量抗锯齿灰度边缘。常规项目规则为 `mask > 127` 得到前景。严禁悄悄改成 `mask > 0` 后再和旧结果直接比较。

### 标准指标协议

统一实现为 `utils/metrics.py`，`evaluate_all.py` 是统一入口：

- 预测阈值：`sigmoid(logits) > 0.5`。
- 汇总方式：**先逐图计算，再对 436 张图取算术平均（macro per-image）**，不是全像素混合 micro 指标。
- 指标：Dice、IoU、Sensitivity/Recall、Precision、Specificity、Accuracy、**surface HD95**、clDice、Boundary F1（tolerance=2 pixels）。
- HD95 是预测/GT mask 表面双向距离的第 95 百分位；空 mask 有明确 fallback，不能替换成任意 foreground-distance 实现。
- clDice 使用 `skimage.morphology.skeletonize`；Boundary F1 只作补充，血管结构更优先看 clDice 和 HD95。

### 训练选权重规则

`train_unified.py` 每个 epoch 在 `val` 上算逐图 Dice，按 val Dice 保存 `best_model.pth`，并 early stop。默认不读取 test；只有显式传入 `--evaluate_test_after_training` 才会在训练结束后运行一次 test。此设计是正确的，不能为了追 development-test 数字改成按 test 选权重。

## 4. 当前最可信的实验结论

以下是可以在组会中报告的结论，注意“同协议比较”和“development-test”措辞。

### 4.1 稳定正证据：green image prior（scratch）

三 seed scratch 结果来自 `results/scratch_delta_multiseed_20260710/metrics_summary.csv`：

| 方法 | Dice mean +/- std | 相对同协议 TransUNet | HD95 mean | clDice mean | Boundary F1 mean | 结论 |
|---|---:|---:|---:|---:|---:|---|
| TransUNet scratch | 0.7518 +/- 0.0009 | baseline | 23.64 | 0.8403 | 0.6373 | 对照 |
| green MSE10, Grad0 | 0.7584 +/- 0.0011 | **+0.0066** | 23.42 | 0.8467 | 0.6468 | 当前最稳定的 green-prior 证据 |

该模型不是把所有图都提高，而是主要修复低召回困难图：Scratch baseline Recall < 0.6 的 37 张中，green MSE 改善了 30 张，平均 Dice `+0.0459`；在较正常的 353 张图中，平均 Dice 基本不变（约 `+0.00005`）。

叙事应为：green prior 补充低对比、断裂或暗细毛细血管的局部可见性。不要写成“全面优于 RGB”。

### 4.2 有价值但未成为最终单模型主线的线索

| 线索 | 已知结果 | 正确解释 |
|---|---|---|
| ImageNet21k pretraining | 三 seed pretrained baseline Dice `0.7569 +/- 0.0012`，高于 scratch；green 的平均增益缩小至 `+0.0009` | 预训练是强初始化，压缩了 green prior 的边际增益；不能仅挑有利 pretrain/scratch 协议。 |
| K2 双教师输出蒸馏 | seed42 相对 pretrained F0 Dice `+0.0037`、clDice `+0.0069`、Boundary F1 `+0.0105`；K2 绝对 Dice `0.7609` | 最好的单模型 KD 单次信号，但未完成强阳性门槛/多 seed主线验证，不能称最终胜出。 |
| F0 + F3 概率 ensemble | 相对 pretrained F0：Dice `+0.0064`、HD95 `-0.74`、clDice `+0.0073`、Boundary F1 `+0.0136` | RGB 语义与 green morphology 有明确互补；这是双模型方案，尚未可靠压缩为单模型。 |
| C3 old anisotropic/no-BN/no-aug | scratch 3 seed Dice `+0.0058`；pretrained 3 seed `+0.0023` | 旧 C3 的 strip conv 是 `1x7 -> 7x1` 串联，更近似分解二维卷积，不是独立方向建模；只能当历史对照，不能过度宣称形态创新。 |

### 4.3 重要负结论：已停止或低优先级的方向

| 方向 | 结果/问题 | 接手后的处理 |
|---|---|---|
| decoder feature consistency V1 | 同一网络的 RGB/enhanced 与 green view 裸 decoder MSE，不是独立强 teacher；预训练下 Dice `-0.0065` | 不要原样重跑。 |
| Decoder KD V2 | development-test 相对 pretrained F0 Dice `-0.0029`、HD95 `+1.72`、Boundary F1 `-0.0056`；可视化中 FP 增多 | 现有 V2 失败，禁止把它作为主线。若未来重做 KD，需要独立 frozen teacher、可靠性/局部权重或 logits/结构蒸馏，并建立全新对照。 |
| final-layer CNN + TransUNet dual fusion | 不是完整 TransFuse；提升不稳定/较小 | 暂停。 |
| F3 directional multiscale | scratch Dice 仅 `+0.0014`；常以 Recall 增加换 Precision 下降；gate 有塌缩迹象 | 不要继续加 strip branch 或更多注入尺度。 |
| direct clDice/Boundary/cbDice loss sweep | 容易增加 Recall 和 FP，未形成稳定 Dice 净收益 | 不继续随意调权重。 |
| CGMA 2x2 与强增强 S0--S4 | 局部 contrast prior / boundary-centerline 辅助在 val 未满足预设继续条件；test 微小涨点不可信 | 已按规则停止。保留为严谨负结果。 |
| 官方 MMSeg DeepLabV3+ | 同口径 development-test Dice `0.7382`，低于 scratch TransUNet `0.7522` | 不继续把“换 DeepLabV3+”当主线；但不能据此判师妹结果错误。 |

完整结果与停止逻辑见 `docs/group_meeting_plan_20260721.md`。任何试图复活以上方向的计划，必须先回答：它与旧实现有何本质差异、对准哪个可视化失败模式、对照/停止规则是什么。

## 5. 当前 development-test 可视化复盘（最值得继续的证据）

目录：`results/prediction_error_review_20260730`。

### 已有资产

- 9 组代表模型 x 436 张预测，共 3924 张预测 PNG。
- `index.html`：可视化总览和全部病例索引。
- `all_cases/`：每张图的 RGB、GT 和多模型 TP/FP/FN 横向图；TP=绿、FP=红、FN=蓝。
- `rankings/`：baseline 最好/最差、最高 FN/FP/HD95、方法最大改善/退化、模型分歧样本。
- `per_image_metrics.csv`：逐图完整指标、TP/TN/FP/FN、前景面积、连通域、green contrast 诊断特征。
- `patient_metrics.csv`：按患者聚合的指标。统计检验和聚合不可把同一患者多张图当独立样本。
- `probability_cache/`：float16 概率图，可免推理重生成画图和报告。

### 由图像得到的事实

1. 错误并不只是不足检：低对比/大范围连续血管时有大面积 FN；背景纹理和反光接近血管时也会有大面积 FP。
2. 典型大 FN：`ANFC_001302.png`、`ANFC_001303.png`、`ANFC_001343.png`；典型纹理/反光 FP：`ANFC_000389.png`。
3. 困难病例按患者聚集：`8_84237`、`8_55896`、`9_60031`、`8_92229`。其中 `8_92229` 有 43 张，含许多极端漏检图。
4. F3、S2 等结构/方向方法多数通过提高 Recall 获得小的结构指标信号，同时牺牲 Precision，说明“更敏感地把背景当血管”。
5. Decoder KD V2 的红色 FP 明显增多；F0+F3 ensemble 的局部互补最清楚，但 uniform soft KD 没有完整保留其局部收益。

组会现场建议打开：baseline 最差、green MSE 最大改善、green MSE 最大退化、Decoder KD V2 退化、F0+F3 ensemble 改善五类样本。不要只展示 Ours 获胜图；必须同时展示 wins 和 losses。

## 6. 官方 DeepLabV3+ 与师妹代码审计

### 已完成的公平官方复现

师妹提供 `TYT_Code/mmsegmentation-main`，为 OpenMMLab MMSegmentation `1.2.2`。本项目已用其官方结构完成 scratch 复现：

- 架构：`MobileNetV2 + DepthwiseSeparableASPPHead`，2 类 softmax。
- 训练：CrossEntropy + `2 x Dice`、AdamW lr `1e-3`、10000 iterations、seed42、水平/垂直翻转、正负45度旋转、`PhotoMetricDistortion`。
- 数据：固定当前 patient-level split；`dataset_all_filtered_mmseg` 用硬链接图像，mask 依据 `>127` 转严格 0/1，不改原始数据。
- 权重：只以 val MMSeg `mDice` 选择；最佳为 `results/official_deeplabv3plus_20260730/work_dirs/seed42_scratch_10k/best_mDice_iter_10000.pth`，val mDice `87.79`（包含背景类，不能与前景 Dice 直接比较）。
- 最终统一 development-test 结果：Dice `0.7382`、IoU `0.5956`、Recall `0.7196`、Precision `0.7809`、HD95 `23.34`、clDice `0.8221`、Boundary F1 `0.6214`。

相对 scratch TransUNet：Dice `-0.0140`、Recall `-0.0641`，但 Precision `+0.0381`、HD95 `-0.77`。即更保守、少一些误检，但漏掉更多血管。

### 为什么仍不能判定师妹“算错”

师妹称 HD95 可到 15.x，但暂未提供足以复核的材料。她提供的 `TYT_Code/evaluate_deeplabv3plus_to_csv.py` 存在以下口径不确定性：

- 数据根目录在她电脑的绝对路径，未知是否同一个 patient split。
- 评估中 mask 是 `mask > 0`，与本项目标准 `mask > 127` 不同；mask 灰度边缘会显著改变边界/HD95。
- 它 import 了没有一同提供的 `metrics.py`，HD95 具体定义未知。
- 未提供实际 best checkpoint、train/val/test 文件名清单、训练 mask 的 unique values、后处理/阈值策略。

优先向师妹索取四项材料：`train/val/test` 文件名清单、实际训练 mask（或 unique values）、best checkpoint、原始 `metrics.py`。拿到 checkpoint 后，先用 `evaluate_mmseg_deeplabv3plus.py` 在当前 436 张 development-test 统一复评，再讨论是否复现她的完整训练过程。

## 7. 关键代码接口与运行命令

### 常规训练和统一评估

```powershell
# 统一评估现有 manifest；此操作仅加载权重并预测，不会重新训练。
D:\anaconda3\envs\pytorch\python.exe evaluate_all.py `
  --manifest docs\unified_eval_manifest_all_filtered.json `
  --dataset all_filtered --split test --threshold 0.5 --img_size 256 --batch_size 4

# 预测可视化复盘；默认会复用已有预测缓存。
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_prediction_error_review_20260730.ps1 -ReusePredictions
```

`evaluate_all.py` 评估 manifest 时会先构建与权重匹配的模型，再加载完整 state dict；日志里出现“未加载预训练权重”只表示构造模型时没有加载 ImageNet npz，**不代表训练得到的 `best_model.pth` 没有加载**。

### 官方 MMSeg DeepLabV3+

```powershell
# 首次创建隔离环境；不要在 pytorch 环境做这一步。
.\scripts\setup_official_mmseg_env.ps1

# 仅检查 config 和数据视图。
.\scripts\run_official_deeplabv3plus_20260730.ps1 -Stage smoke

# 已训练过；只有有明确理由才重跑。若从中断 checkpoint 恢复：
.\scripts\run_official_deeplabv3plus_20260730.ps1 -Stage train -Resume

# 用指定 checkpoint 做统一评估。
.\scripts\run_official_deeplabv3plus_20260730.ps1 -Stage evaluate -Checkpoint <path-to-checkpoint>
```

### 常见接口约定

- `VesselDataset`：读 RGB 后转 RGB，缩放 256，正常 mask 二值化为 `>127`；光度增强只作用 image，几何增强同步作用于辅助图/soft targets。
- `train_unified.py --mode`：`baseline`、`ours`、`prior_fusion`、`soft_kd`、`cgma`。后四种是历史实验实现，不默认等于推荐方向。
- `ours` 主线的历史最佳为 `teacher_mode=green_only`、`enhancer=basic`、`joint_model=v1`、fixed `lambda_mse=10`、`lambda_grad=0`。
- `soft_kd`：使用离线 `.npy` soft target，训练不在线加载两位 teacher，目标数据在 `results/dual_teacher_kd_20260717/` 相关子目录。
- 所有新实验都应保存 `config.json`、`training_log.txt`、`best_model.pth` 和 `val_per_image.csv`；批量实验额外保存 run summary/metrics summary/manifest。

## 8. 重要结果与权重索引

不要把每个历史权重都搬到新目录。常用权重路径已固定在 `docs/prediction_visualization_manifest_20260730.json`，以下仅列最常用的事实索引：

| 名称 | 代表权重/结果 | 用途 |
|---|---|---|
| Scratch TransUNet seed42 | `results/experiments/all_filtered/baseline_retrain_20260619/0619_0232/best_model.pth` | scratch 主对照 |
| Green MSE seed42 | `results/experiments/all_filtered/ours_green_only_mse_only_20260620/0620_0616/best_model.pth` | 绿色先验代表权重 |
| Pretrained F0 seed42 | `results/experiments/all_filtered/f0_transunet_corrected_pretrained_seed42_20260715/0715_1907/best_model.pth` | 预训练 RGB 专家 |
| F3 directional seed42 | `results/experiments/all_filtered/f3_directional_green_multiscale_scratch_seed42_20260715/0715_1556/best_model.pth` | 绿色 morphology 专家 / ensemble 成员 |
| K2 soft KD seed42 | `results/experiments/all_filtered/K2_uniform_lambda1p0_20260717/0717_0206/best_model.pth` | 最好单模型 KD 单次结果 |
| 官方 DeepLabV3+ | `results/official_deeplabv3plus_20260730/work_dirs/seed42_scratch_10k/best_mDice_iter_10000.pth` | 官方框架强 baseline |
| 可视化复盘 | `results/prediction_error_review_20260730/` | 组会展示、失败模式分析 |

## 9. 论文与后续研究边界

### 当前可以说什么

- Green-channel image prior 在 scratch、多 seed 下有稳定增益，且可视化显示其主要补偿低对比困难图的漏检。
- RGB 预训练专家与 green morphology 专家在概率 ensemble 中有互补性。
- 许多复杂模块的失败模式已被可视化解释：通常是 Recall 增加伴随 FP/Precision 损失，或 decoder KD 直接造成过分割。
- 官方 MMSeg DeepLabV3+ 在统一口径未超过 TransUNet；框架名称本身不是性能差异的充分解释。

### 当前不能说什么

- 不能称 `dataset_all_filtered/test` 是论文最终独立 test，也不能把反复查看后的数字包装成最终泛化性能。
- 不能说 K2 或 C3 已稳定优于主线；它们缺少满足预设条件的充分多 seed/最终验证。
- 不能说师妹 HD95 15.x 错，也不能说她模型一定强；目前缺 checkpoint/split/metric 口径。
- 不能只凭 `+0.003` 到 `+0.007` 的单 split 提升宣称已足以投 SCI 2--3 区。

### 对论文可信度真正重要的下一阶段工作

1. **数据和标注审计**：按患者检查困难子群（特别是 `8_92229`、`8_55896`、`8_84237`），记录成像条件、曝光、反光、对焦、标注一致性；必要时做子群性能报告或标注复核。
2. **强 baseline**：优先准备 nnU-Net v2 2D 或 MedNeXt 的同 split 公平对照。它们是投稿必要基线，不等于创新点。官方 DeepLabV3+ 已完成，不要重复跑同一配置。
3. **独立验证**：最终模型确定后，进行患者级 outer 5-fold CV，或保留从未用于研发的患者级 final holdout。所有统计按患者聚合。
4. **若继续探索模型**：只探索有明确失败模式依据的紧凑机制，例如根据两专家不一致性进行局部可靠性融合/蒸馏，而不是再追加普通 attention、更多分支或裸 feature MSE。应先制定：基线、val-only 决策、固定 seed、继续门槛、停止门槛、计算预算。
5. **最终报告**：多 seed mean +/- std、患者级 paired delta/bootstrap CI、可视化 wins/losses、参数量/FLOPs/推理速度、困难子群与局限性。

## 10. 接手后的推荐工作流

### 8 月 7 日组会前

1. 不必冒险开新训练。确认 `docs/group_meeting_prediction_review_20260730.md` 的数字与 CSV 一致。
2. 准备展示 `results/prediction_error_review_20260730/index.html` 中至少 5 组正/负对照案例。
3. 对导师的关键问题使用以下简洁回答：

> 我们从全部 436 张 development-test 可视化中确认，green prior 的提升主要集中在低对比、低召回困难图；当前瓶颈同时包含漏检和纹理/反光误检。官方 DeepLabV3+ 在相同 split、标签与统一 surface-HD95 口径下未超过 TransUNet，因此师妹 15.x HD95 需要先以 checkpoint、split 和原始 metrics 对齐复核。下一步先做困难患者/标注审计、强 baseline 和独立患者级验证，不再盲目堆叠模块。

### 组会后，如用户同意继续研究

1. 先拿师妹 checkpoint / split / metrics；没有这些材料时不要猜测其结果原因。
2. 再与用户确定优先级：强 baseline（nnU-Net/MedNeXt）还是可靠性融合新线。两条线不建议同时无计划大规模开跑。
3. 对每轮新实验写一个短 manifest/计划，至少包含：唯一 experiment 名、数据版本、split、init、seed、augmentation、loss、val 选权重规则、development-test 是否允许评估、成功/停止标准、输出目录。
4. 新权重加入 `evaluate_all.py` manifest 前，先做随机 forward、真实一 epoch smoke、state-dict reload、val 统一评估；确认后再跑完整训练。

## 11. 已知问题与避免重复踩坑

| 问题 | 正确处理 |
|---|---|
| PowerShell `-Seeds 43,44` 被解析为一个字符串 | 传 `-Seeds "43,44"`；相关脚本的解析已修复以支持逗号/空格/分号。 |
| manifest JSON 有 UTF-8 BOM | `evaluate_all.py` 应用 `utf-8-sig` 读 JSON；不要改回普通 `utf-8`。 |
| 评估日志显示“未加载预训练权重” | 先确认完整训练 state dict 是否随后加载；构造时没加载 ImageNet npz 不等于评估未加载 `best_model.pth`。 |
| DeepLab/MMSeg mask 直接用原始灰度 | 必须用 `dataset_all_filtered_mmseg` 的 0/1 标签视图，不能直接把 0--255 灰度当 class index。 |
| MMSeg 在中文 Windows 启动报 GBK/MSVC 解码错 | 使用已修改的官方脚本，保持 `PYTHONUTF8=1`。 |
| MMSeg val 报 CUDA `histc` deterministic 错 | 配置保持 `randomness.deterministic=False`，同时固定 seed。 |
| 只看 development-test 新高 | 查看同协议 val 是否支持、逐患者 delta、wins/losses 和 Precision/HD95/clDice/Boundary F1；没有则不宣称收益。 |
| 用同一患者多张图直接显著性检验 | 先聚合到患者级再做 paired bootstrap/Wilcoxon。 |

## 12. 可直接粘贴给 DeepSeek V4 的启动提示词

```text
你现在接手 D:\Projects_\JiaBi_new 的甲襞毛细血管二值分割研究项目。请全程使用中文，先不要改代码或启动训练。

请按以下顺序读取：
1) docs/deepseek_v4_handoff_20260806.md
2) docs/group_meeting_prediction_review_20260730.md
3) results/prediction_error_review_20260730/analysis_summary.md
4) docs/group_meeting_plan_20260721.md
5) docs/official_deeplabv3plus_audit_20260730.md
6) docs/prediction_visualization_manifest_20260730.json

然后用不超过 15 条要点回答：
- 当前数据划分、mask/预测阈值、统一指标和 development-test 边界；
- 最可信的正结果与哪些实验已明确停止；
- 可视化显示的主要失败模式和困难患者；
- 官方 DeepLabV3+ 复现及师妹 HD95 15.x 为什么尚不能直接比较；
- 8 月 7 日组会建议怎样汇报；
- 组会后最值得推进的两条路线及各自需要的验证条件。

严格规则：不要删除/重置当前脏工作区；不要按 development-test 选 checkpoint；不要把不同训练协议的绝对 Dice 直接排名；不要未经计划就继续堆 attention、strip convolution、裸 decoder MSE 或结构损失。提出任何新实验前，先写出同协议 baseline、val-only 选模、seed、停止条件和输出目录。
```

## 13. 交接完成检查清单

- [ ] 已阅读 P0 文件并能复述为什么 test 只能叫 development-test。
- [ ] 已知项目标准 mask 规则是 `>127`、预测阈值是 `0.5`、HD95 是 surface HD95。
- [ ] 已知 green MSE scratch 三 seed `+0.0066` 是最稳定的正证据。
- [ ] 已知 K2/C3/ensemble 是线索而不是论文最终胜出结论。
- [ ] 已知 decoder KD、复杂 F3、CGMA 和官方 DeepLabV3+ 当前不应被重复作为默认主线。
- [ ] 已打开至少一次 `results/prediction_error_review_20260730/index.html`，同时看过改善与退化样本。
- [ ] 已知师妹结果需要四项材料才能公平复核。
- [ ] 已确认工作区脏，后续编辑不得破坏既有结果和用户文件。
