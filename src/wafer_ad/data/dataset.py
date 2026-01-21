import logging
import os
from PIL import Image
import torch

from typing import Tuple
from torchvision import transforms

from torch.utils.data import random_split

from torchvision.transforms import InterpolationMode

from wafer_ad.utils.path import resolve_path


class WaferDataset:
    def __init__(
        self, 
        data_folder: str,
        is_for_train: bool = True,
        idx_dataset: int = 1,
    ) -> None:
        self.idx_dataset = idx_dataset
        self.is_for_train = is_for_train
        
        self.dataset_folder = resolve_path(data_folder)

        
        self.x, self.y, self.mask = self.load_dataset_folder()
        
        folder = os.path.join(self.dataset_folder, 'train', 'normal')
        with Image.open(os.path.join(folder, os.listdir(folder)[0])) as img:
            w, h = img.size
            self.img_size: Tuple[int, int] = (h, w)
         
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.img_size, InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
    def load_dataset_folder(self) -> Tuple[list, list, list]:
        phase = 'train' if self.is_for_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_folder, phase)
        gt_dir = os.path.join(self.dataset_folder, 'ground_truth', "mask")

        for img_type in sorted(os.listdir(img_dir)):  # "normal" or "anomalous" subfolders

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                logging.warning(f'Skipped non-directory file: {img_type_dir}')
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            if img_type == 'normal':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_fpath_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)])
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "Number of samples and labels do not match."
        assert len(x) == len(mask), "Number of samples and masks do not match."

        return list(x), list(y), list(mask)
    
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, *self.img_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)


def split_train_val(
    dataset: torch.utils.data.Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    assert 0.0 < val_ratio < 1.0, "val_ratio must be between 0 and 1."

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )
    return train_set, val_set
