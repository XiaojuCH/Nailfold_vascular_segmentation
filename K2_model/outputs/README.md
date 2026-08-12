# 输出目录

训练和评估脚本默认将新实验写入本目录。不要覆盖不同数据集或不同 seed 的结果；建议使用如 `outputs/dataset_name_seed42` 的独立目录。

当前的 `smoke_dataset_audit.json` 与 `smoke_reference_eval/` 是本包在 2026-08-08 对原始 `dataset_all_filtered` 的交付核验记录：

- val：Dice `0.7977`、HD95 `18.30`；
- development-test：Dice `0.7609`、HD95 `21.35`。

它们不是新实验，也不应用于选择模型。
