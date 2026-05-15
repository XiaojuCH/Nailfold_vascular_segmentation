import os
import cv2
from tqdm import tqdm

def generate_teacher_priors(input_dir, output_dir, mode="green+clahe", clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    生成 Teacher 先验图。
    mode: "green+clahe"（默认）| "clahe_only"（全图CLAHE）| "green_only"（仅绿通道）
    """
    os.makedirs(output_dir, exist_ok=True)
    filenames = sorted(os.listdir(input_dir))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    print(f"[{mode}] 生成 Teacher 先验图，共 {len(filenames)} 张...")
    for filename in tqdm(filenames):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is None:
            continue

        b, g, r = cv2.split(img)

        if mode == "green+clahe":
            ch = clahe.apply(g)
        elif mode == "clahe_only":
            # 对灰度图做 CLAHE
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch = clahe.apply(gray)
        elif mode == "green_only":
            ch = g
        else:
            raise ValueError(f"未知 mode: {mode}")

        enhanced_img = cv2.merge([ch, ch, ch])
        cv2.imwrite(os.path.join(output_dir, filename), enhanced_img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="green+clahe", choices=["green+clahe", "clahe_only", "green_only"])
    parser.add_argument("--dataset", default="dataset_all_filtered")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        generate_teacher_priors(
            input_dir=f"{args.dataset}/{split}/images",
            output_dir=f"{args.dataset}/{split}/teacher_priors_{args.mode.replace('+', '_')}",
            mode=args.mode,
        )
