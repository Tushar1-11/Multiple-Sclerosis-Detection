"""
MSNet — PyTorch Dataset with MONAI Transforms
================================================
Handles:
  - Loading multi-sequence MRI NIfTI files
  - Loading pre-processed clinical metadata
  - Modality dropout (random during training)
  - MONAI-based augmentation and transform pipelines
  - Support for missing modalities (zero-fill strategy)
  - Binary and multi-class label handling

Dataset structure (CSV):
  subject_id | T1 | T2 | FLAIR | GD_T1 | label | age | gender | edss | ...

Missing modality handling:
  When a modality path is None or missing:
    - The volume tensor is zero-filled
    - A modality mask (binary) records absent modalities
    - The model uses this mask to gate attention weights
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import nibabel as nib

# MONAI transforms — extensive medical augmentation library
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAffined,
    Rand3DElasticd,
    ToTensord,
    CropForegroundd,
    ResizeWithPadOrCropd,
    EnsureTyped,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Modality Configuration
# ----------------------------------------------------------
MODALITIES = ["T1", "T2", "FLAIR", "GD_T1"]


# ============================================================
# Transform Builders
# ============================================================

def build_train_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Build MONAI training augmentation pipeline.

    Augmentations are applied stochastically and are designed
    to simulate realistic MRI acquisition variability:
      - Random spatial augmentations (flip, rotate, zoom)
      - Random intensity augmentations (noise, bias field)
      - Elastic deformation (mimics tissue variability)

    Args:
        spatial_size: Target [D, H, W] voxel size
        voxel_spacing: Target isotropic voxel spacing in mm

    Returns:
        Composed MONAI transform pipeline
    """
    keys = MODALITIES

    return Compose([
        # ---- Load & Preprocess ----
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=keys,
            pixdim=voxel_spacing,
            mode="bilinear",
            allow_missing_keys=True,
        ),
        ResizeWithPadOrCropd(
            keys=keys,
            spatial_size=spatial_size,
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(
            keys=keys,
            nonzero=True,  # Normalize only within brain mask (non-zero)
            channel_wise=True,
            allow_missing_keys=True,
        ),

        # ---- Spatial Augmentations ----
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
        RandRotate90d(keys=keys, prob=0.3, max_k=3, allow_missing_keys=True),
        RandZoomd(
            keys=keys,
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.2,
            allow_missing_keys=True,
        ),
        RandAffined(
            keys=keys,
            prob=0.2,
            rotate_range=[0.1, 0.1, 0.1],
            shear_range=[0.1, 0.1],
            translate_range=[5, 5, 5],
            scale_range=[0.1, 0.1, 0.1],
            mode="bilinear",
            allow_missing_keys=True,
        ),

        # ---- Intensity Augmentations ----
        RandGaussianNoised(
            keys=keys,
            prob=0.2,
            mean=0.0,
            std=0.05,
            allow_missing_keys=True,
        ),
        RandBiasFieldd(
            keys=keys,
            prob=0.1,
            allow_missing_keys=True,
        ),
        Rand3DElasticd(
            keys=keys,
            prob=0.1,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            allow_missing_keys=True,
        ),

        # ---- Type Conversion ----
        EnsureTyped(keys=keys, dtype=torch.float32, allow_missing_keys=True),
        ToTensord(keys=keys, allow_missing_keys=True),
    ])


def build_val_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Build deterministic MONAI validation/test transform pipeline.
    No augmentation — only normalization and padding.
    """
    keys = MODALITIES

    return Compose([
        LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        Spacingd(
            keys=keys,
            pixdim=voxel_spacing,
            mode="bilinear",
            allow_missing_keys=True,
        ),
        ResizeWithPadOrCropd(
            keys=keys,
            spatial_size=spatial_size,
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(
            keys=keys,
            nonzero=True,
            channel_wise=True,
            allow_missing_keys=True,
        ),
        EnsureTyped(keys=keys, dtype=torch.float32, allow_missing_keys=True),
        ToTensord(keys=keys, allow_missing_keys=True),
    ])


# ============================================================
# Modality Dropout
# ============================================================

class ModalityDropout:
    """
    Randomly drops (zeros out) MRI modalities during training.

    Modality dropout forces the model to learn robust representations
    that do not rely entirely on any single sequence, which is crucial
    for real-world deployment where some scans may be unavailable.

    Reference:
      Ma et al. (2021) — "Modality-Aware Mutual Learning for
      Multi-Modal Medical Image Segmentation"

    Args:
        dropout_prob: Probability of dropping any single modality.
            A value of 0.2 means each modality has a 20% chance
            of being zeroed out independently.
        min_modalities: Minimum number of modalities to keep non-zero.
    """

    def __init__(self, dropout_prob: float = 0.2, min_modalities: int = 1,
                 spatial_size: tuple = (128, 128, 128)):
        self.dropout_prob = dropout_prob
        self.min_modalities = min_modalities
        self.spatial_size = spatial_size

    def __call__(
        self,
        volumes: Dict[str, Optional[torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply modality dropout.

        Args:
            volumes: Dict of {modality_name: tensor or None}

        Returns:
            (augmented_volumes, modality_mask)
        """
        n_modalities = len(MODALITIES)
        modality_mask = torch.ones(n_modalities, dtype=torch.float32)

        # Determine which modalities to drop
        drop_flags = [
            (vol is not None) and (torch.rand(1).item() < self.dropout_prob)
            for vol in volumes.values()
        ]

        # Ensure at least min_modalities remain
        available = [not d and (v is not None) for d, v in zip(drop_flags, volumes.values())]
        if sum(available) < self.min_modalities:
            # Don't drop anything if too few are available
            drop_flags = [False] * n_modalities

        result = {}
        for i, (modality, tensor) in enumerate(volumes.items()):
            if tensor is None or drop_flags[i]:
                # Zero-fill missing/dropped modality
                result[modality] = torch.zeros(1, *self.spatial_size, dtype=torch.float32)
                modality_mask[i] = 0.0  # Mark as absent
            else:
                result[modality] = tensor

        return result, modality_mask


# ============================================================
# MS Dataset
# ============================================================

class MSDataset(Dataset):
    """
    PyTorch Dataset for multimodal MS classification.

    Loads multi-sequence MRI volumes and clinical metadata for each
    subject. Handles missing modalities via zero-fill and modality
    masking. Applies MONAI augmentation transforms.

    Args:
        df: Metadata DataFrame with columns for paths and clinical features
        transform: MONAI Compose transform pipeline
        clinical_features: List of clinical column names
        spatial_size: MRI volume spatial dimensions
        modality_dropout_prob: Dropout probability (training only)
        apply_modality_dropout: Whether to apply modality dropout
        task: 'binary' or 'multiclass'
        preload: If True, pre-load all volumes into memory (fast but RAM-heavy)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        clinical_features: Optional[List[str]] = None,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        modality_dropout_prob: float = 0.0,
        apply_modality_dropout: bool = False,
        task: str = "multiclass",
        preload: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.clinical_features = clinical_features or [
            "age", "gender", "edss", "disease_duration",
            "igg_index", "oligoclonal_bands", "t_cell_count"
        ]
        self.spatial_size = spatial_size
        self.task = task
        self.preload = preload

        # Modality dropout module (only active during training)
        self.dropout = (
            ModalityDropout(
                dropout_prob=modality_dropout_prob,
                spatial_size=spatial_size,
            )
            if apply_modality_dropout
            else None
        )

        # Pre-load data if requested
        self._cache = {}
        if self.preload:
            logger.info("[Dataset] Pre-loading all volumes into memory...")
            for idx in range(len(self.df)):
                self._cache[idx] = self._load_subject(idx)
            logger.info(f"[Dataset] Loaded {len(self._cache)} subjects")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a sample dict with keys:
          - 'volumes': Tensor [n_mod, 1, D, H, W] — MRI sequences
          - 'modality_mask': Tensor [n_mod] — 1=present, 0=missing
          - 'clinical': Tensor [n_features] — normalized clinical data
          - 'label': Tensor (scalar) — class label
          - 'subject_id': str — subject identifier
        """
        row = self.df.iloc[idx]

        # ---- Load MRI Volumes ----
        if self.preload and idx in self._cache:
            volumes, modality_mask = self._cache[idx]
        else:
            volumes, modality_mask = self._load_subject(idx)

        # Stack modalities: [n_modalities, 1, D, H, W]
        volume_stack = torch.stack([
            volumes[mod] for mod in MODALITIES
        ], dim=0)  # [4, 1, D, H, W]

        # ---- Clinical Features ----
        clinical = self._load_clinical(row)

        # ---- Label ----
        label = int(row["label"])
        if self.task == "binary":
            label = int(label > 0)  # HC=0, any MS=1

        return {
            "volumes": volume_stack,           # [4, 1, D, H, W]
            "modality_mask": modality_mask,    # [4]
            "clinical": clinical,              # [n_features]
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": str(row.get("subject_id", idx)),
        }

    def _load_subject(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Load and transform MRI volumes for one subject."""
        row = self.df.iloc[idx]

        # Build MONAI data dict for existing modalities
        data_dict = {}
        for mod in MODALITIES:
            path = row.get(mod)
            if pd.notna(path) and path and Path(str(path)).exists():
                data_dict[mod] = str(path)

        # Run MONAI transform pipeline
        try:
            transformed = self.transform(data_dict)
        except Exception as e:
            logger.error(f"[Dataset] Transform failed for idx={idx}: {e}")
            transformed = {}

        # Build volumes dict, zero-fill missing modalities
        dummy_shape = (1, *self.spatial_size)
        volumes = {}
        for mod in MODALITIES:
            if mod in transformed and transformed[mod] is not None:
                vol = transformed[mod]
                # Ensure shape [1, D, H, W]
                if vol.dim() == 3:
                    vol = vol.unsqueeze(0)
                volumes[mod] = vol.float()
            else:
                volumes[mod] = torch.zeros(dummy_shape, dtype=torch.float32)

        # Apply modality dropout (training only)
        if self.dropout is not None:
            volumes, modality_mask = self.dropout(volumes)
        else:
            # Build mask from actual availability
            modality_mask = torch.tensor(
                [1.0 if row.get(mod) and pd.notna(row.get(mod)) else 0.0
                 for mod in MODALITIES],
                dtype=torch.float32,
            )

        return volumes, modality_mask

    def _load_clinical(self, row: pd.Series) -> torch.Tensor:
        """Extract and tensor-ify clinical feature vector."""
        features = []
        for feat in self.clinical_features:
            val = row.get(feat, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)
        return torch.tensor(features, dtype=torch.float32)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.
        Used to handle class imbalance in MS datasets.

        Returns:
            Tensor of shape [n_classes]
        """
        labels = self.df["label"].values
        if self.task == "binary":
            labels = (labels > 0).astype(int)

        classes, counts = np.unique(labels, return_counts=True)
        n_total = len(labels)
        weights = n_total / (len(classes) * counts)
        weights_tensor = torch.zeros(len(classes), dtype=torch.float32)
        for cls, w in zip(classes, weights):
            weights_tensor[cls] = w
        return weights_tensor


# ============================================================
# DataLoader Factory
# ============================================================

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Uses WeightedRandomSampler for balanced class sampling during
    training when classes are imbalanced (typical in MS datasets).

    Args:
        train_df: Training split DataFrame
        val_df: Validation split DataFrame
        test_df: Test split DataFrame
        config: Loaded YAML config dict

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    spatial_size = tuple(data_cfg["spatial_size"])
    voxel_spacing = tuple(data_cfg["voxel_spacing"])
    batch_size = data_cfg["batch_size"]
    n_workers = data_cfg["num_workers"]
    clinical_features = data_cfg["clinical_features"]
    modality_dropout_prob = data_cfg["modality_dropout"]["dropout_prob"]
    apply_md = data_cfg["modality_dropout"]["enabled"]

    # ---- Build Datasets ----
    train_dataset = MSDataset(
        df=train_df,
        transform=build_train_transforms(spatial_size, voxel_spacing),
        clinical_features=clinical_features,
        spatial_size=spatial_size,
        modality_dropout_prob=modality_dropout_prob,
        apply_modality_dropout=apply_md,
        task=data_cfg["task"],
    )

    val_dataset = MSDataset(
        df=val_df,
        transform=build_val_transforms(spatial_size, voxel_spacing),
        clinical_features=clinical_features,
        spatial_size=spatial_size,
        apply_modality_dropout=False,
        task=data_cfg["task"],
    )

    test_dataset = MSDataset(
        df=test_df,
        transform=build_val_transforms(spatial_size, voxel_spacing),
        clinical_features=clinical_features,
        spatial_size=spatial_size,
        apply_modality_dropout=False,
        task=data_cfg["task"],
    )

    # ---- Weighted Sampler for class balance ----
    class_weights = train_dataset.get_class_weights()
    sample_weights = class_weights[train_df["label"].values]
    if data_cfg["task"] == "binary":
        labels_bin = (train_df["label"].values > 0).astype(int)
        sample_weights = class_weights[labels_bin]

    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(train_dataset),
        replacement=True,
    )

    # prefetch_factor only valid when using multiprocessing workers
    prefetch = data_cfg.get("prefetch_factor", 2) if n_workers > 0 else None

    # ---- Build Loaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        prefetch_factor=prefetch,
        persistent_workers=(n_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        prefetch_factor=prefetch,
        persistent_workers=(n_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Single-instance for inference
        shuffle=False,
        num_workers=n_workers,
    )

    logger.info(
        f"[DataLoaders] Train={len(train_dataset)} | "
        f"Val={len(val_dataset)} | Test={len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
