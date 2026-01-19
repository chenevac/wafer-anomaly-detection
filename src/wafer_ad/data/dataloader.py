from typing import Tuple
from torch.utils.data import DataLoader

from wafer_ad.data.dataset import WaferDataset, split_train_val


def get_data_loaders(
    idx_dataset: int = 1,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader,DataLoader,DataLoader]:
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        idx_dataset: Index of the dataset to use.
        batch_size: Batch size for the DataLoaders.
        validation_ratio: Ratio of the training set to use for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple containing DataLoaders for training, validation, and testing.
    """

    train_and_val_set = WaferDataset(idx_dataset=idx_dataset, is_for_train=True)
    train_set, val_set = split_train_val(train_and_val_set, val_ratio=validation_ratio, seed=seed)
    
    
    test_set = WaferDataset(idx_dataset=idx_dataset, is_for_train=False)

    train_loader = DataLoader(
        train_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set, 
        pin_memory=True, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return train_loader, val_loader, test_loader
