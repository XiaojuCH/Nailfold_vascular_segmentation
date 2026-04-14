import os
import shutil
from pathlib import Path

def rename_dataset(dataset_root):
    """标准化QH数据集的文件命名"""
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    # 获取所有图片文件并排序
    image_files = sorted(images_dir.glob("*.png"))

    print(f"找到 {len(image_files)} 个图片文件")

    # 创建备份目录
    backup_dir = dataset_root / "backup_original_names"
    backup_dir.mkdir(exist_ok=True)

    # 保存原始命名映射
    mapping_file = backup_dir / "rename_mapping.txt"

    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("新文件名 -> 原文件名\n")
        f.write("=" * 80 + "\n")

        for idx, img_path in enumerate(image_files, start=1):
            old_name = img_path.name
            new_name = f"ANFC_{idx:06d}.png"

            # 重命名图片
            new_img_path = images_dir / new_name
            img_path.rename(new_img_path)

            # 重命名对应的mask
            old_mask_path = masks_dir / old_name
            if old_mask_path.exists():
                new_mask_path = masks_dir / new_name
                old_mask_path.rename(new_mask_path)

            f.write(f"{new_name} -> {old_name}\n")

            if idx % 500 == 0:
                print(f"已处理 {idx}/{len(image_files)} 个文件")

    print(f"\n重命名完成！")
    print(f"映射文件保存在: {mapping_file}")

if __name__ == "__main__":
    dataset_path = r"D:\Projects_\JiaBi_new\third_party\ANFC_THU_data_256"
    rename_dataset(dataset_path)
