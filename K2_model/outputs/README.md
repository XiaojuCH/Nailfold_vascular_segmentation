# 输出目录

训练和评估脚本默认将新实验写入本目录。不要覆盖不同数据集或不同 seed 的结果；建议使用如 `outputs/dataset_name_seed42` 的独立目录。

每次完成实验后，保留该目录下的 `config.json`、`best_model.pth`、训练日志、soft target 元数据和评估 CSV。不要将旧数据集的权重、soft target 或指标结果混入新的实验目录。
