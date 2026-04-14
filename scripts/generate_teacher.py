import os
import cv2
from tqdm import tqdm

def generate_teacher_priors(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    提取图像绿通道并应用 CLAHE 增强，生成供 Teacher 分支使用的物理先验图。
    """
    os.makedirs(output_dir, exist_ok=True)
    filenames = sorted(os.listdir(input_dir))
    
    # 实例化 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    print(f"开始生成 Teacher 先验图，共 {len(filenames)} 张...")
    for filename in tqdm(filenames):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"警告: 无法读取图像 {filename}")
            continue
            
        # OpenCV 默认读取为 BGR，分离通道 (B=0, G=1, R=2)
        b, g, r = cv2.split(img)
        
        # 对绿通道应用 CLAHE
        enhanced_g = clahe.apply(g)
        
        # 为了与网络输入维度保持一致（如果是单通道也可以，这里转回3通道灰度图方便处理）
        enhanced_img = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
        
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, enhanced_img)

if __name__ == "__main__":
    # 请根据你的实际路径修改
    TRAIN_IMG_DIR = "dataset_all_filtered/train/images"
    TRAIN_TEACHER_DIR = "dataset_all_filtered/train/teacher_priors"
    
    generate_teacher_priors(TRAIN_IMG_DIR, TRAIN_TEACHER_DIR)
    
    # Val 和 Test 也同样处理
    generate_teacher_priors("dataset_all_filtered/val/images", "dataset_all_filtered/val/teacher_priors")
    generate_teacher_priors("dataset_all_filtered/test/images", "dataset_all_filtered/test/teacher_priors")