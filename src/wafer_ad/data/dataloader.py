from typing import Tuple
from torch.utils.data import DataLoader

from wafer_ad.data.dataset import WaferDataset, split_train_val


def get_data_loaders(
    data_folder: str,
    batch_size: int = 16,
    validation_ratio: float = 0.2,
    seed: int = 42,
    
) -> Tuple[DataLoader,DataLoader,DataLoader]:
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for the DataLoaders.
        validation_ratio: Ratio of the training set to use for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple containing DataLoaders for training, validation, and testing.
    """

    train_and_val_set = WaferDataset(is_for_train=True, data_folder=data_folder)
    train_set, val_set = split_train_val(train_and_val_set, val_ratio=validation_ratio, seed=seed)
    
    
    test_set = WaferDataset(is_for_train=False, data_folder=data_folder)

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
