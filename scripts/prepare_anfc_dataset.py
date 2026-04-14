"""
ANFC_THU 数据集预处理脚本

输入：third_party/QH_Dataset/ANFC_THU_data/ANFC_THU_segmentation/
      - {id}.jpg  原始图像
      - {id}.json LabelImg polygon 标注 (label="blur")

输出：dataset_anfc_split/
      train/ val/ test/
      ├── images/         原始图像
      ├── masks/          二值掩码 (polygon → filled mask)
      └── teacher_priors/ 绿通道 CLAHE 增强图

Split: 按患者ID分组后随机划分 70/15/15
"""

import os
import sys
import json
import random
import shutil
import numpy as np
import cv2
from tqdm import tqdm

# 确保从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========================== 配置 ==========================
SRC_DIR = "third_party/QH_Dataset/ANFC_THU_data/ANFC_THU_segmentation"
OUT_DIR = "dataset_anfc_split"
IMG_SIZE = 256
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
RANDOM_SEED = 42
# =========================================================


def json_to_mask(json_path, img_h, img_w):
    """将 LabelImg polygon JSON 转换为二值 mask"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def generate_teacher(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """绿通道 CLAHE 增强 → 3通道灰度图"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    b, g, r = cv2.split(img_bgr)
    enhanced_g = clahe.apply(g)
    return cv2.merge([enhanced_g, enhanced_g, enhanced_g])


def get_patient_id(filename):
    """从文件名提取患者ID，例如 '8_115825_1.jpg' → '8_115825'"""
    name = os.path.splitext(filename)[0]
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else name


def save_split(file_list, split_name, src_dir, out_dir):
    img_dir = os.path.join(out_dir, split_name, "images")
    mask_dir = os.path.join(out_dir, split_name, "masks")
    teacher_dir = os.path.join(out_dir, split_name, "teacher_priors")
    for d in [img_dir, mask_dir, teacher_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n[{split_name}] 处理 {len(file_list)} 张图像...")
    failed = 0
    for jpg_name in tqdm(file_list):
        stem = os.path.splitext(jpg_name)[0]
        json_name = stem + ".json"

        jpg_path = os.path.join(src_dir, jpg_name)
        json_path = os.path.join(src_dir, json_name)

        if not os.path.exists(json_path):
            print(f"  警告: 找不到标注文件 {json_name}，跳过")
            failed += 1
            continue

        img_bgr = cv2.imread(jpg_path)
        if img_bgr is None:
            print(f"  警告: 无法读取图像 {jpg_name}，跳过")
            failed += 1
            continue

        img_h, img_w = img_bgr.shape[:2]

        # 生成 mask（在原始分辨率下 fillPoly，再 resize）
        mask = json_to_mask(json_path, img_h, img_w)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # 生成 teacher prior（在原始分辨率下增强，再 resize）
        teacher_bgr = generate_teacher(img_bgr)
        teacher_resized = cv2.resize(teacher_bgr, (IMG_SIZE, IMG_SIZE))

        # resize 原图
        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

        # 保存（统一用 png 避免 jpg 压缩损失 mask 精度）
        out_name = stem + ".png"
        cv2.imwrite(os.path.join(img_dir, out_name), img_resized)
        cv2.imwrite(os.path.join(mask_dir, out_name), mask_resized)
        cv2.imwrite(os.path.join(teacher_dir, out_name), teacher_resized)

    print(f"  完成，失败 {failed} 张")


def main():
    random.seed(RANDOM_SEED)

    # 收集所有 jpg 文件
    all_jpgs = sorted([
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith(".jpg")
    ])
    print(f"共找到 {len(all_jpgs)} 张图像")

    # 按患者ID分组
    patient_map = {}
    for jpg in all_jpgs:
        pid = get_patient_id(jpg)
        patient_map.setdefault(pid, []).append(jpg)

    patient_ids = sorted(patient_map.keys())
    random.shuffle(patient_ids)
    print(f"共 {len(patient_ids)} 位患者: {patient_ids}")

    n = len(patient_ids)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    train_pids = patient_ids[:n_train]
    val_pids = patient_ids[n_train:n_train + n_val]
    test_pids = patient_ids[n_train + n_val:]

    train_files = [f for pid in train_pids for f in patient_map[pid]]
    val_files = [f for pid in val_pids for f in patient_map[pid]]
    test_files = [f for pid in test_pids for f in patient_map[pid]]

    print(f"\nSplit (按患者):")
    print(f"  Train: {len(train_pids)} 患者, {len(train_files)} 图像")
    print(f"  Val:   {len(val_pids)} 患者, {len(val_files)} 图像")
    print(f"  Test:  {len(test_pids)} 患者, {len(test_files)} 图像")

    save_split(train_files, "train", SRC_DIR, OUT_DIR)
    save_split(val_files, "val", SRC_DIR, OUT_DIR)
    save_split(test_files, "test", SRC_DIR, OUT_DIR)

    print(f"\n[完成] 数据集已保存至 {OUT_DIR}/")
    print("运行训练命令示例:")
    print(f"  python train_unified.py --mode baseline --data_dir {OUT_DIR} --exp_name anfc_baseline_transunet")
    print(f"  python train_unified.py --mode ours --data_dir {OUT_DIR} --exp_name anfc_ours_transunet")


if __name__ == "__main__":
    main()
