import os
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_data_loaders(
    i: int,
    img_size: Tuple[int, int],
    batch_size: int = 16,
) -> Tuple[DataLoader,DataLoader]:

    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_set = ImageFolder(
        os.path.join("data", "texture_ad", "wafer", str(i), "train", "normal"),
        transform=transform_train,
    )
    test_set = ImageFolder(
        os.path.join("data", "texture_ad", "wafer", str(i), "test"),
        transform=transform_train,
        target_transform=lambda y: 1 - y,
    )

    train_loader = DataLoader(
        train_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, test_loader
