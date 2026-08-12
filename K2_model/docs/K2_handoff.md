# K2 交接说明

## 当前实验版本

- 名称：`K2_uniform_lambda1p0_seed42`。
- F0 教师：ImageNet21k 预训练 TransUNet，完整训练权重初始化学生。
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

## 当前甲襞结果（固定 436 张 development-test）

| 实验 | Dice | IoU | Recall | Precision | HD95 | clDice | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch TransUNet | 0.7522 | 0.6140 | 0.7838 | 0.7428 | 24.11 | 0.8403 | 0.6405 |
| Green MSE scratch | 0.7571 | 0.6193 | 0.7960 | 0.7398 | 23.28 | 0.8451 | 0.6477 |
| F0 pretrained | 0.7571 | 0.6199 | 0.8099 | 0.7293 | 21.72 | 0.8468 | 0.6346 |
| K0 fine-tune control | 0.7561 | 0.6193 | 0.7781 | 0.7541 | 21.90 | 0.8490 | 0.6451 |
| K2 soft KD | 0.7609 | 0.6248 | 0.8041 | 0.7388 | 21.35 | 0.8537 | 0.6451 |
| F0+F3 ensemble | 0.7636 | 0.6280 | 0.8053 | 0.7432 | 20.98 | 0.8542 | 0.6483 |

K2 相对 F0：Dice `+0.0037`、HD95 `-0.37`、clDice `+0.0069`、Boundary F1 `+0.0105`。相对 K0：Dice `+0.0047`、HD95 `-0.55`、clDice `+0.0047`。这表明该 seed 的增益不只是“多训练一些 epoch”。

## 复现风险

1. 现有 test 已用于开发选择，只能作为 development-test；新的结论必须使用患者级 outer-CV 或未接触 holdout。
2. K2 对病例并非完全稳定。它会在部分病例抑制 FP 或补连续结构，也会在部分图上增加 FP/漏检；当前单 seed 不应被包装为最终结论。
3. 新数据集与甲襞域不同。若 F0/F3 直接迁移，不再是“同数据集双教师 KD”的实验；需要单列为 domain-transfer 研究。
4. `mask > 127`、阈值 `0.5`、surface-HD95 和 per-image macro 是当前统一协议的一部分。任何改变都必须重新对所有对照进行统一复评。
