from typing import Tuple
from torch.utils.data import DataLoader

from wafer_ad.data.dataset import WaferDataset


def get_data_loaders(
    idx_dataset: int = 1,
    batch_size: int = 16,
) -> Tuple[DataLoader,DataLoader]:

    train_set = WaferDataset(idx_dataset=idx_dataset, is_for_train=True)
    test_set = WaferDataset(idx_dataset=idx_dataset, is_for_train=False)

    train_loader = DataLoader(
        train_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return train_loader, test_loader
