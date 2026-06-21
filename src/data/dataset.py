"""PyTorch Dataset wrapping the synthetic ID manifest."""

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ImageNet normalization (we're using transfer learning from ImageNet weights)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(min_height=int(image_size * 1.15), min_width=int(image_size * 1.15),
                       border_mode=0, fill=255),
        A.RandomCrop(height=image_size, width=image_size),
        A.Rotate(limit=8, border_mode=0, fill=255, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ImageCompression(quality_range=(60, 95), p=0.4),
        # Notably absent: HorizontalFlip — would invert text/security marks
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                       border_mode=0, fill=255),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


class IDDataset(Dataset):
    def __init__(self, manifest_csv, transforms=None, project_root=PROJECT_ROOT):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transforms = transforms
        self.project_root = Path(project_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.project_root / row["path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transforms:
            img = self.transforms(image=img)["image"]
        label = torch.tensor(row["label"], dtype=torch.long)
        return img, label