import os
import cv2
import numpy as np
from tqdm import tqdm


def _normalize_to_uint8(x):
    x = x.astype(np.float32)
    min_v = float(x.min())
    max_v = float(x.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - min_v) / (max_v - min_v)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _frangi_vesselness(green):
    try:
        from skimage.filters import frangi
    except Exception as exc:
        raise RuntimeError("green_frangi teacher requires scikit-image with skimage.filters.frangi") from exc

    image = green.astype(np.float32) / 255.0
    try:
        vesselness = frangi(
            image,
            sigmas=(1, 2, 3),
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=True,
        )
    except TypeError:
        vesselness = frangi(image, scale_range=(1, 3), scale_step=1, black_ridges=True)
    return _normalize_to_uint8(vesselness)


def generate_teacher_priors(input_dir, output_dir, mode="green+clahe", clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Generate teacher prior images.

    Modes:
    - green+clahe: CLAHE on green channel.
    - clahe_only: CLAHE on grayscale image.
    - green_only: raw green channel.
    - green_blackhat: morphological black-hat on green channel for dark thin vessel structures.
    - green_clahe_blackhat: CLAHE green followed by black-hat.
    - green_frangi: Frangi vesselness on green channel, assuming dark ridges.
    """
    os.makedirs(output_dir, exist_ok=True)
    filenames = sorted(os.listdir(input_dir))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    print(f"[{mode}] Generate teacher priors: {len(filenames)} files")
    for filename in tqdm(filenames):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is None:
            continue

        b, g, r = cv2.split(img)

        if mode == "green+clahe":
            ch = clahe.apply(g)
        elif mode == "clahe_only":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch = clahe.apply(gray)
        elif mode == "green_only":
            ch = g
        elif mode == "green_blackhat":
            ch = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
            ch = clahe.apply(_normalize_to_uint8(ch))
        elif mode == "green_clahe_blackhat":
            enhanced_g = clahe.apply(g)
            ch = cv2.morphologyEx(enhanced_g, cv2.MORPH_BLACKHAT, kernel)
            ch = clahe.apply(_normalize_to_uint8(ch))
        elif mode == "green_frangi":
            ch = _frangi_vesselness(g)
            ch = clahe.apply(ch)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        teacher_img = cv2.merge([ch, ch, ch])
        cv2.imwrite(os.path.join(output_dir, filename), teacher_img)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="green+clahe",
        choices=[
            "green+clahe",
            "clahe_only",
            "green_only",
            "green_blackhat",
            "green_clahe_blackhat",
            "green_frangi",
        ],
    )
    parser.add_argument("--dataset", default="dataset_all_filtered")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        generate_teacher_priors(
            input_dir=f"{args.dataset}/{split}/images",
            output_dir=f"{args.dataset}/{split}/teacher_priors_{args.mode.replace('+', '_')}",
            mode=args.mode,
        )