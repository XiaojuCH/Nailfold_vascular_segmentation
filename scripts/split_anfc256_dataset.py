"""按患者ID分割ANFC_THU_data_256数据集"""
import os
import shutil
import random
from collections import defaultdict

SRC_DIR = "third_party/ANFC_THU_data_256"
OUT_DIR = "dataset_anfc256_split"
MAPPING_FILE = f"{SRC_DIR}/backup_original_names/rename_mapping.txt"

random.seed(42)

# 读取映射，按患者ID分组
patient_files = defaultdict(list)
with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if '->' not in line or '新文件名' in line:
            continue
        new_name, old_name = [x.strip() for x in line.split('->')]
        # ANFC数据已经过滤，直接提取患者ID
        patient_id = '_'.join(old_name.split('_')[:2])  # 8_115825 或 9_xxxxx
        patient_files[patient_id].append(new_name)

# 分割患者
patients = list(patient_files.keys())
random.shuffle(patients)
n_train = int(len(patients) * 0.70)
n_val = int(len(patients) * 0.15)

splits = {
    'train': patients[:n_train],
    'val': patients[n_train:n_train+n_val],
    'test': patients[n_train+n_val:]
}

print(f"总患者: {len(patients)}, Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# 复制文件
for split_name, patient_list in splits.items():
    os.makedirs(f"{OUT_DIR}/{split_name}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/{split_name}/masks", exist_ok=True)

    count = 0
    for patient_id in patient_list:
        for filename in patient_files[patient_id]:
            shutil.copy2(f"{SRC_DIR}/images/{filename}", f"{OUT_DIR}/{split_name}/images/{filename}")
            shutil.copy2(f"{SRC_DIR}/masks/{filename}", f"{OUT_DIR}/{split_name}/masks/{filename}")
            count += 1
    print(f"{split_name}: {count} 张")

print(f"完成！保存至: {OUT_DIR}")
