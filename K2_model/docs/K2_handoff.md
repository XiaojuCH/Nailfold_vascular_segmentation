# K2 方法说明

## 方法配置

- F0 教师：ImageNet21k 初始化后，在目标数据集训练完成的 RGB TransUNet；完整训练权重初始化学生。
- F3 教师：scratch `directional_multiscale` green-prior fusion 模型。
- 学生：普通 TransUNet；无 enhancer、无额外 decoder head、推理只需要一个 `best_model.pth`。
- soft target：`0.5 * sigmoid(F0 logits) + 0.5 * sigmoid(F3 logits)`。
- KD：逐像素 `BCEWithLogits(student logits, soft target)`，uniform、`lambda_kd=1.0`。
- 主分割损失：`BCEWithLogits + soft Dice`。

## 与 Green MSE、K0 的区别

| 名称 | 是否教师网络 | 约束位置 | 训练/推理结构 |
|---|---|---|---|
| Green MSE | 否；固定 green-only 图像先验 | 输入图像级 MSE | Enhancer + TransUNet 均参与推理 |
| F0 | 无 KD | 仅 GT 分割监督 | TransUNet |
| K0 | 软标签会读取，但 `lambda_kd=0` | 仅 GT 分割监督 | 从 F0 继续训练的 TransUNet |
| K2 | 是；冻结 F0 与 F3 | 输出概率级 soft KD | 从 F0 继续训练的单个 TransUNet |

K0 是必须理解的控制：它和 K2 使用同一 F0 初始化、学习率、训练轮数和数据增强；唯一差别是 K0 的 KD 系数为 0。因此 `K2 - K0` 比 `K2 - scratch baseline` 更能说明软标签本身的贡献。

## 复现风险

1. 每个新数据集都要在该数据集上重新训练 F0、F3，再生成 soft target；本包不提供可直接迁移的血管教师权重。
2. 训练期仅用 val 选择 checkpoint；test 只在方案固定后评估。需要报告稳定性时补 seed43/44 和患者级统计。
3. `mask > 127`、阈值 `0.5`、surface-HD95 和 per-image macro 是当前统一协议的一部分。任何改变都必须重新对所有对照统一复评。
