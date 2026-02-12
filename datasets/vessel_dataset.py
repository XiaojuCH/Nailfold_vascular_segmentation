import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VesselDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Data augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        mask = (mask > 127).astype(np.int64)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.long)
        }
