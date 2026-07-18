import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class VesselDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        teacher_dir=None,
        soft_target_dir=None,
        disagreement_dir=None,
        img_size=256,
        augment=False,
        intensity_aug=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.teacher_dir = teacher_dir
        self.soft_target_dir = soft_target_dir
        self.disagreement_dir = disagreement_dir
        self.img_size = img_size
        self.augment = augment
        self.intensity_aug = intensity_aug

        self.filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self._validate_auxiliary_files(self.soft_target_dir, "soft target")
        self._validate_auxiliary_files(self.disagreement_dir, "disagreement")

    def _validate_auxiliary_files(self, directory, label):
        if directory is None:
            return
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Missing {label} directory: {directory}")
        missing = [
            filename
            for filename in self.filenames
            if not os.path.isfile(os.path.join(directory, os.path.splitext(filename)[0] + ".npy"))
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} {label} files in {directory}. First files: {preview}"
            )

    def _load_probability_map(self, directory, filename, label):
        path = os.path.join(directory, os.path.splitext(filename)[0] + ".npy")
        array = np.load(path, allow_pickle=False)
        if array.shape == (1, self.img_size, self.img_size):
            array = array[0]
        expected_shape = (self.img_size, self.img_size)
        if array.shape != expected_shape:
            raise ValueError(f"{label} shape mismatch for {filename}: {array.shape} != {expected_shape}")
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains NaN or Inf: {path}")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < -1e-4 or maximum > 1.0001:
            raise ValueError(f"{label} outside [0, 1] for {filename}: [{minimum}, {maximum}]")
        return np.clip(array, 0.0, 1.0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        teacher = None
        if self.teacher_dir is not None:
            teacher_path = os.path.join(self.teacher_dir, filename)
            teacher = cv2.imread(teacher_path)
            teacher = cv2.cvtColor(teacher, cv2.COLOR_BGR2RGB)
            teacher = cv2.resize(teacher, (self.img_size, self.img_size))

        soft_target = None
        if self.soft_target_dir is not None:
            soft_target = self._load_probability_map(self.soft_target_dir, filename, "soft target")

        disagreement = None
        if self.disagreement_dir is not None:
            disagreement = self._load_probability_map(self.disagreement_dir, filename, "disagreement")

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # ================= 定制化医学数据增强 =================
        if self.augment:
            # 1. 水平翻转 (符合解剖学对称性)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
                if teacher is not None: teacher = np.flip(teacher, axis=1)
                if soft_target is not None: soft_target = np.flip(soft_target, axis=1)
                if disagreement is not None: disagreement = np.flip(disagreement, axis=1)

            # 2. 小角度旋转 (-15度 到 15度，模拟手指微斜)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
                # 使用反射边界(BORDER_REFLECT)避免黑边影响网络判断
                image = cv2.warpAffine(image, M, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                if teacher is not None:
                    teacher = cv2.warpAffine(teacher, M, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                if soft_target is not None:
                    soft_target = cv2.warpAffine(
                        soft_target,
                        M,
                        (self.img_size, self.img_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                if disagreement is not None:
                    disagreement = cv2.warpAffine(
                        disagreement,
                        M,
                        (self.img_size, self.img_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )

            # 3. 亮度与对比度抖动 (仅对输入原图进行，绝不能改变 mask 和 teacher 物理先验)
            if self.intensity_aug and np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # 对比度控制 [0.8, 1.2]
                beta = np.random.uniform(-15, 15)    # 亮度控制 [-15, 15]
                # Keep the photometric transform linear. convertScaleAbs applies
                # an absolute value, which can invert low-intensity relationships.
                image = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        # ====================================================

        # Normalize to [0, 1] & HWC → CHW
        image = (image.astype(np.float32) / 255.0).transpose(2, 0, 1)
        mask = (mask > 127).astype(np.float32) 

        sample = {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0) 
        }

        if teacher is not None:
            teacher = (teacher.astype(np.float32) / 255.0).transpose(2, 0, 1)
            sample["teacher"] = torch.tensor(teacher, dtype=torch.float32)

        if soft_target is not None:
            soft_target = np.ascontiguousarray(soft_target, dtype=np.float32)
            sample["soft_target"] = torch.from_numpy(soft_target).unsqueeze(0)

        if disagreement is not None:
            disagreement = np.ascontiguousarray(disagreement, dtype=np.float32)
            sample["disagreement"] = torch.from_numpy(disagreement).unsqueeze(0)

        return sample
