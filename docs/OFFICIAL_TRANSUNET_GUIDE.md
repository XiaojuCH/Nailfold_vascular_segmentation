# 官方 TransUNet 集成指南

## 1. 已完成
✅ 官方代码已下载到 `TransUNet/` 目录

## 2. 下载预训练权重

官方使用 Google ViT 预训练权重（在 ImageNet-21k 上训练）

### 方法1：Google Drive（推荐）
访问：https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd

下载 `R50+ViT-B_16.npz` 文件，放到：
```
model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

### 方法2：命令行下载（需要 gdown）
```bash
pip install gdown
mkdir -p model/vit_checkpoint/imagenet21k
# 下载 R50+ViT-B_16 权重（约 300MB）
gdown 1pJx3KnfDnGGY_MM_6-8nZMCWN0Qhx_Zt -O model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## 3. 安装依赖
```bash
cd TransUNet
pip install -r requirements.txt
```

## 4. 集成到你的项目

### 方式1：使用适配器（推荐）

已创建 `models/transunet_official.py` 适配器

**在 train_baselines.py 中使用：**
```python
from models.transunet_official import TransUNetOfficial

# 替换原来的 TransUNet
elif args.model == "transunet":
    model = TransUNetOfficial(
        n_channels=3,
        n_classes=1,
        img_size=256,
        pretrained_path='model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    ).to(DEVICE)
```

**在 train_ours.py 中使用：**
```python
from models.transunet_official import TransUNetOfficial

# 替换 segmentor
segmentor = TransUNetOfficial(
    n_channels=3,
    n_classes=1,
    img_size=256,
    pretrained_path='model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
)
```

### 方式2：直接使用官方代码
```python
import sys
sys.path.insert(0, 'TransUNet')
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
```

## 5. 快速开始

```bash
# 1. 创建目录
mkdir -p model/vit_checkpoint/imagenet21k

# 2. 下载预训练权重（需要手动从 Google Drive 下载）
# 链接: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd
# 下载 R50+ViT-B_16.npz 放到 model/vit_checkpoint/imagenet21k/

# 3. 测试模型
python example_use_official_transunet.py

# 4. 训练
python train_baselines.py --model transunet
```

## 6. 注意事项

- **预训练权重必须下载**：官方模型依赖预训练权重，否则性能会很差
- **权重文件约 300MB**：需要一定下载时间
- **显存需求**：batch_size 建议设为 12 或更小
- **输入尺寸**：官方默认 224x224，你的是 256x256（已适配）

## 7. 模型配置

官方提供多个配置：
- `R50-ViT-B_16`：ResNet50 + ViT-Base（推荐，平衡性能和速度）
- `ViT-B_16`：纯 ViT-Base（更慢但可能更准）
- `ViT-L_16`：ViT-Large（最强但最慢）
