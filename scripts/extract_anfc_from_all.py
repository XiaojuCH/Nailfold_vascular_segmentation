"""从All数据集中提取纯ANFC数据（8_和9_开头的68个患者）"""
import os
import shutil

SRC_DIR = "third_party/ANFC_OURS_All_dataset"
OUT_DIR = "third_party/ANFC_THU_data_256"
MAPPING_FILE = f"{SRC_DIR}/backup_original_names/rename_mapping.txt"

# 创建输出目录
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUT_DIR}/masks", exist_ok=True)
os.makedirs(f"{OUT_DIR}/backup_original_names", exist_ok=True)

# 读取映射文件，只保留ANFC数据
anfc_count = 0
with open(MAPPING_FILE, 'r', encoding='utf-8') as f_in:
    with open(f"{OUT_DIR}/backup_original_names/rename_mapping.txt", 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if '->' not in line:
                f_out.write(line)
                continue

            parts = line.strip().split(' -> ')
            if len(parts) != 2:
                continue

            new_name, old_name = [x.strip() for x in parts]

            # 只保留8_和9_开头的ANFC数据
            if not (old_name.startswith('8_') or old_name.startswith('9_')):
                continue

            # 复制文件
            shutil.copy2(f"{SRC_DIR}/images/{new_name}", f"{OUT_DIR}/images/{new_name}")
            shutil.copy2(f"{SRC_DIR}/masks/{new_name}", f"{OUT_DIR}/masks/{new_name}")
            f_out.write(line)
            anfc_count += 1

print(f"提取完成！ANFC数据: {anfc_count} 张")
