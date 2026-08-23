"""
MSNet — PyTorch Tabular Dataset & DataLoaders
=============================================
Provides PyTorch Dataset and DataLoader wrappers for clinical & volumetric data.
Includes weighted random sampling to handle class imbalance.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """
    PyTorch Dataset for preprocessed clinical & volumetric tabular data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature matrix [N, num_features]
            y: Target label vector [N]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_tabular_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    splits: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0,
    use_balanced_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build Train, Validation, and Test DataLoaders from preprocessed arrays.

    Args:
        X: Preprocessed feature matrix [N, num_features]
        y: Target label vector [N]
        splits: Dictionary containing 'train', 'val', 'test' index arrays
        batch_size: Batch size for training
        num_workers: DataLoader worker count
        use_balanced_sampler: Whether to use WeightedRandomSampler for class balance

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    train_ds = TabularDataset(X[train_idx], y[train_idx])
    val_ds = TabularDataset(X[val_idx], y[val_idx])
    test_ds = TabularDataset(X[test_idx], y[test_idx])

    # Calculate class distribution and class weights
    y_train = y[train_idx]
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)

    # Balanced class weights (Inverse Frequency)
    num_classes = len(classes)
    class_weights = total_samples / (num_classes * counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    logger.info(
        f"Train Class Counts: {dict(zip(classes.tolist(), counts.tolist()))} | Class Weights: {class_weights.round(3).tolist()}"
    )

    if use_balanced_sampler:
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_weights_tensor
