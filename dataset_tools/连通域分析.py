import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def select_valid_vessel_images(img_input_dir, mask_input_dir,
                               img_output_dir, mask_output_dir,
                               invalid_img_dir, invalid_mask_dir,
                               min_area=50,
                               max_thickness_radius=25,  # 【全新核心指标】：允许的最大半径（即最厚不能超过50像素宽）
                               min_aspect_ratio=2.5,
                               max_solidity=0.6):
    img_in = Path(img_input_dir)
    mask_in = Path(mask_input_dir)

    img_out = Path(img_output_dir)
    mask_out = Path(mask_output_dir)
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    invalid_img_out = Path(invalid_img_dir)
    invalid_mask_out = Path(invalid_mask_dir)
    invalid_img_out.mkdir(parents=True, exist_ok=True)
    invalid_mask_out.mkdir(parents=True, exist_ok=True)

    mask_files = list(mask_in.glob("*.png"))

    kept_count = 0
    discarded_count = 0

    print(f"开始筛选数据集，共发现 {len(mask_files)} 个样本对...")
    print("规则：恢复宽容度，仅通过检测『色块局部最大厚度』来剔除一块一块的死角斑块。")

    for mask_path in tqdm(mask_files, desc="Selecting Images"):
        img_path = img_in / mask_path.name
        if not img_path.exists():
            continue

        mask = cv2.imread(str(mask_path), 0)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

        has_valid_vessel = False
        has_fatal_blob = False

        # 遍历连通域
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # 面积太小的噪点，直接忽略，不参与任何判定
            if area < min_area:
                continue

            # 提取当前连通域的独立 Mask
            comp_mask = np.uint8(labels == i) * 255

            # =======================================================
            # 【全新杀招：厚度探测】
            # 用距离变换算出该形状内部，距离边缘最远的点的距离（即最大内切圆半径）
            dist_transform = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            max_thickness = np.max(dist_transform)

            # 如果这个色块的最厚处半径超过了阈值，说明它是个"块"，而不是"管"！
            # 无论旁边有没有好血管，直接一票否决整张图！
            if max_thickness > max_thickness_radius:
                has_fatal_blob = True
                break  # 发现大坨坨，直接死刑，退出循环
            # =======================================================

            # 提取轮廓用于形态学计算 (恢复第二版的判断逻辑)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            # 1. 计算长宽比
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if min(w, h) == 0:
                aspect_ratio = 0
            else:
                aspect_ratio = max(w, h) / min(w, h)

            # 2. 计算凸实度
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                solidity = 1.0
            else:
                solidity = float(area) / hull_area

            # 如果长得像血管，标记为找到合格血管
            if aspect_ratio >= min_aspect_ratio or solidity <= max_solidity:
                has_valid_vessel = True

        # --- 最终判定分流 ---
        # 必须是：找到了血管，且【没有】肥大的斑块，才算合格
        if has_valid_vessel and not has_fatal_blob:
            shutil.copy2(str(mask_path), str(mask_out / mask_path.name))
            shutil.copy2(str(img_path), str(img_out / img_path.name))
            kept_count += 1
        else:
            # 包含肥大斑块，或者全都是小噪点没有血管，全部剔除
            shutil.copy2(str(mask_path), str(invalid_mask_out / mask_path.name))
            shutil.copy2(str(img_path), str(invalid_img_out / img_path.name))
            discarded_count += 1

    print("\n" + "=" * 30)
    print("数据集筛选完成！")
    print(f"✅ 合格保留: {kept_count}")
    print(f"❌ 剔除脏数据: {discarded_count}")
    print("=" * 30)


if __name__ == "__main__":
    BASE_DIR = Path(r"C:\Workfolder\NailFold\nailData\orgin_data")
    SAVE_DIR = Path(r"C:\Workfolder\NailFold\nailData\selected_data")

    IMG_INPUT = BASE_DIR / "images"
    MASK_INPUT = BASE_DIR / "masks"

    IMG_OUTPUT = SAVE_DIR / "images"
    MASK_OUTPUT = SAVE_DIR / "masks"
    INVALID_IMG_OUTPUT = SAVE_DIR / 'invalid_data' / "images"
    INVALID_MASK_OUTPUT = SAVE_DIR / 'invalid_data' / "mask"

    select_valid_vessel_images(
        img_input_dir=IMG_INPUT,
        mask_input_dir=MASK_INPUT,
        img_output_dir=IMG_OUTPUT,
        mask_output_dir=MASK_OUTPUT,
        invalid_img_dir=INVALID_IMG_OUTPUT,
        invalid_mask_dir=INVALID_MASK_OUTPUT,
        min_area=50,
        max_thickness_radius=25,  # 【关键参数】：管壁最厚处半径>25像素(即直径>50像素)，就判定为斑块。如果还有块状物漏网，就把25调小(比如20)；如果粗血管被误杀，调大(比如35)
        min_aspect_ratio=2.5,  # 恢复第二版的宽容度
        max_solidity=0.6  # 恢复第二版的宽容度
    )