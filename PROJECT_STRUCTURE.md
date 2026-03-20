# 项目目录结构

```
JiaBi_new/
├── README.md                      # 项目说明
├── LICENSE                        # 许可证
├── .gitignore                     # Git忽略文件
│
├── models/                        # 模型定义
│   ├── __init__.py
│   ├── enhancer.py               # 轻量级增强网络
│   ├── joint_model.py            # 联合模型
│   ├── joint_framework.py        # 完整框架
│   ├── unet_baseline.py          # U-Net基线
│   ├── unet_plus_plus.py         # U-Net++基线
│   ├── transunet.py              # 简化版TransUNet
│   └── transunet_official.py     # 官方TransUNet适配器
│
├── datasets/                      # 数据加载
│   └── dataset_vessel.py         # 甲襞血管数据集
│
├── losses/                        # 损失函数
│   └── joint_loss.py             # 联合蒸馏损失
│
├── utils/                         # 工具函数
│   ├── metrics.py                # 评估指标
│   ├── seed.py                   # 随机种子
│   └── model_stats.py            # 模型统计
│
├── trainers/                      # 训练器
│   └── trainer.py                # 通用训练器
│
├── scripts/                       # 脚本
│   ├── generate_teacher.py       # 生成Teacher Prior
│   ├── visualize_comparison.py   # 模型对比可视化
│   ├── visualize_mechanism.py    # 机制可视化
│   └── visualize_all_mechanisms.py
│
├── tests/                         # 测试文件
│   └── test_official_transunet.py
│
├── train_baselines.py            # 基线模型训练
├── train_ours.py                 # 提出方法训练
│
├── dataset_raw_split/            # 数据集
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/                        # 预训练权重
│   └── vit_checkpoint/
│       └── imagenet21k/
│           └── R50+ViT-B_16.npz
│
├── results/                      # 实验结果
│   ├── experiments/              # 训练结果
│   │   ├── baselines/
│   │   ├── ours_transunet/
│   │   └── ours_grad30/
│   ├── visualizations/           # 可视化结果
│   │   ├── vis_results_overlay/
│   │   ├── vis_results_mechanism/
│   │   └── vis_results_all_mechanisms/
│   └── checkpoints/              # 模型检查点
│
├── third_party/                  # 第三方代码
│   ├── TransUNet/                # 官方TransUNet
│   ├── project_TransUNet/        # 下载的完整项目
│   └── QH_Dataset/               # 其他数据集
│
└── docs/                         # 文档
    └── OFFICIAL_TRANSUNET_GUIDE.md
```

## 目录说明

### 核心代码
- `models/` - 所有模型实现
- `datasets/` - 数据加载器
- `losses/` - 损失函数
- `utils/` - 工具函数

### 训练脚本
- `train_baselines.py` - 训练基线模型（U-Net, U-Net++, TransUNet）
- `train_ours.py` - 训练提出的联合蒸馏框架

### 实验结果
- `results/experiments/` - 训练日志、权重、曲线
- `results/visualizations/` - 可视化图片
- `results/checkpoints/` - 模型检查点

### 第三方
- `third_party/TransUNet/` - 官方TransUNet实现
- `third_party/project_TransUNet/` - 预训练权重来源

### 数据
- `dataset_raw_split/` - 训练/验证/测试数据
- `model/vit_checkpoint/` - 预训练权重
