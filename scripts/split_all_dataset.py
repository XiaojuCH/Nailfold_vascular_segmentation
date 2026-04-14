"""按患者ID分割ALL数据集（ANFC+JiaBi混合，120个患者）"""
import os
import shutil
import random
from collections import defaultdict

SRC_DIR = "third_party/ANFC_OURS_All_dataset"
OUT_DIR = "dataset_all_split"
MAPPING_FILE = f"{SRC_DIR}/backup_original_names/rename_mapping.txt"

random.seed(42)

# 读取映射，按患者ID分组
patient_files = defaultdict(list)
with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if '->' not in line or '新文件名' in line:
            continue
        new_name, old_name = [x.strip() for x in line.split('->')]
        # 提取患者ID（支持8_、9_和frame_格式）
        if old_name.startswith('frame_'):
            # frame_000001_box1.png -> frame_000001
            patient_id = '_'.join(old_name.split('_')[:2])
        else:
            # 8_115825_1_box1.png -> 8_115825
            patient_id = '_'.join(old_name.split('_')[:2])
        patient_files[patient_id].append(new_name)

# 分割患者：先按70/15/15随机分，再把图片多的患者从train换到val/test补足数量
patients = list(patient_files.keys())
random.shuffle(patients)
n_train = int(len(patients) * 0.70)
n_val   = int(len(patients) * 0.15)

train_set = set(patients[:n_train])
val_set   = set(patients[n_train:n_train+n_val])
test_set  = set(patients[n_train+n_val:])

def count_imgs(pset):
    return sum(len(patient_files[p]) for p in pset)

TARGET = int(len([f for files in patient_files.values() for f in files]) * 0.15)

# 从train中把图片最多的患者移到val/test，直到达到目标数量
for target_set in [val_set, test_set]:
    while count_imgs(target_set) < TARGET:
        # 找train中图片最多的患者
        best = max(train_set, key=lambda p: len(patient_files[p]))
        train_set.remove(best)
        target_set.add(best)

splits = {
    'train': list(train_set),
    'val':   list(val_set),
    'test':  list(test_set),
}

print(f"总患者: {len(patients)}, Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

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
