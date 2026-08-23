"""
MSNet — Clinical Metadata Processor
======================================
Handles:
  - Loading clinical CSV data
  - Validating and imputing missing values
  - Feature encoding (gender, binary biomarkers)
  - Continuous feature normalization (robust scaler)
  - EDSS discretization and validation (0.0–10.0 scale)
  - Train/val/test stratified splitting

Clinical Features:
  - age              : Patient age (years)
  - gender           : Biological sex (0=female, 1=male)
  - edss             : Expanded Disability Status Scale (0.0–10.0)
  - disease_duration : Duration since first symptom (years)
  - igg_index        : CSF IgG index (normal < 0.7)
  - oligoclonal_bands: Presence of OCBs in CSF (0=absent, 1=present)
  - t_cell_count     : CD4+ T-cell count (cells/µL)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Valid EDSS range (Kurtzke Expanded Disability Status Scale)
EDSS_MIN, EDSS_MAX = 0.0, 10.0

# Expected clinical feature column names
CLINICAL_COLUMNS = [
    "age",
    "gender",
    "edss",
    "disease_duration",
    "igg_index",
    "oligoclonal_bands",
    "t_cell_count",
]

# Features to apply RobustScaler (continuous)
CONTINUOUS_FEATURES = [
    "age",
    "edss",
    "disease_duration",
    "igg_index",
    "t_cell_count",
]

# Binary/categorical features (no scaling)
BINARY_FEATURES = [
    "gender",
    "oligoclonal_bands",
]


# ============================================================
# Validator
# ============================================================

class ClinicalValidator:
    """
    Validates clinical feature ranges and flags anomalies.

    Ensures biologically plausible values for downstream
    model training. Invalid values are set to NaN for imputation.
    """

    # Biologically plausible ranges
    RANGES = {
        "age": (0, 120),
        "edss": (0.0, 10.0),
        "disease_duration": (0, 80),
        "igg_index": (0.0, 5.0),
        "t_cell_count": (0, 3000),
        "gender": (0, 1),
        "oligoclonal_bands": (0, 1),
    }

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate all clinical features, clip or nullify outliers.

        Args:
            df: Raw clinical DataFrame

        Returns:
            (cleaned_df, validation_report)
        """
        report = {}
        df = df.copy()

        for col, (lo, hi) in self.RANGES.items():
            if col not in df.columns:
                logger.warning(f"[Validator] Column '{col}' not found in DataFrame")
                continue

            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                logger.warning(
                    f"[Validator] '{col}': {n_out} out-of-range values "
                    f"(valid: [{lo}, {hi}]) -> set to NaN"
                )
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
            report[col] = {"out_of_range": int(n_out), "n_missing": int(df[col].isna().sum())}

        return df, report


# ============================================================
# Imputer
# ============================================================

class ClinicalImputer:
    """
    Handles missing value imputation for clinical features.

    Strategies:
     - Continuous features: Median imputation (robust to skew)
     - Binary features: Mode imputation
    """

    def __init__(self):
        self._medians: Dict[str, float] = {}
        self._modes: Dict[str, float] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "ClinicalImputer":
        """Fit imputation statistics on training set."""
        for col in CONTINUOUS_FEATURES:
            if col in df.columns:
                self._medians[col] = df[col].median()

        for col in BINARY_FEATURES:
            if col in df.columns:
                self._modes[col] = df[col].mode().iloc[0]

        self._is_fitted = True
        logger.info(f"[Imputer] Fit complete. Medians: {self._medians}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to a DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("Imputer must be fit before transform.")
        df = df.copy()

        for col, median in self._medians.items():
            if col in df.columns:
                n_filled = df[col].isna().sum()
                df[col] = df[col].fillna(median)
                if n_filled > 0:
                    logger.debug(f"[Imputer] '{col}': filled {n_filled} NaN -> median={median:.3f}")

        for col, mode_val in self._modes.items():
            if col in df.columns:
                n_filled = df[col].isna().sum()
                df[col] = df[col].fillna(mode_val)
                if n_filled > 0:
                    logger.debug(f"[Imputer] '{col}': filled {n_filled} NaN -> mode={mode_val}")

        return df


# ============================================================
# Scaler
# ============================================================

class ClinicalScaler:
    """
    Applies RobustScaler to continuous clinical features.

    RobustScaler centers data using median and scales using IQR,
    making it insensitive to clinical outliers (e.g., extreme EDSS).

    Note: Binary features (gender, OCB) are NOT scaled.
    """

    def __init__(self):
        self._scaler = RobustScaler()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "ClinicalScaler":
        """Fit scaler on training data continuous columns."""
        cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        self._scaler.fit(df[cols].values)
        self._fitted_cols = cols
        self._is_fitted = True
        logger.info(f"[Scaler] Fit on columns: {cols}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale continuous columns, return normalized DataFrame."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fit before transform.")
        df = df.copy()
        cols = self._fitted_cols
        df[cols] = self._scaler.transform(df[cols].values)
        return df


# ============================================================
# Full Clinical Processor
# ============================================================

class ClinicalProcessor:
    """
    Full clinical metadata processing pipeline.

    Steps:
      1. Load and validate CSV
      2. Impute missing values (fit on train, apply to val/test)
      3. Normalize continuous features
      4. Return numpy arrays ready for torch.Tensor conversion

    Args:
        features: List of feature column names to use
        label_col: Name of label column in CSV
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        label_col: str = "label",
    ):
        self.features = features or CLINICAL_COLUMNS
        self.label_col = label_col
        self.validator = ClinicalValidator()
        self.imputer = ClinicalImputer()
        self.scaler = ClinicalScaler()
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit on training data and return processed feature array.

        Args:
            df: Training DataFrame (must contain feature + label cols)

        Returns:
            np.ndarray of shape [N, len(features)]
        """
        df, report = self.validator.validate(df)
        logger.info(f"[ClinicalProcessor] Validation report: {report}")

        df = self.imputer.fit(df).transform(df)
        df = self.scaler.fit(df).transform(df)

        self._is_fitted = True
        return df[self.features].values.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted processor to val/test data.

        Args:
            df: Validation or test DataFrame

        Returns:
            np.ndarray of shape [N, len(features)]
        """
        if not self._is_fitted:
            raise RuntimeError("Processor must be fit_transform'd first.")
        df, _ = self.validator.validate(df)
        df = self.imputer.transform(df)
        df = self.scaler.transform(df)
        return df[self.features].values.astype(np.float32)

    def get_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract integer class labels from DataFrame."""
        return df[self.label_col].values.astype(np.int64)


# ============================================================
# Stratified Data Splitter
# ============================================================

def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/val/test split.

    Stratification ensures class proportions are maintained across
    all splits, which is critical for imbalanced MS datasets.

    Args:
        df: Full metadata DataFrame
        label_col: Column containing class labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (remainder → test)
        seed: Random seed for reproducibility

    Returns:
        (train_df, val_df, test_df)
    """
    assert train_ratio + val_ratio < 1.0, \
        "train_ratio + val_ratio must be < 1.0"

    test_ratio = 1.0 - train_ratio - val_ratio
    labels = df[label_col].values

    # First: split off test set
    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_val_idx, test_idx = next(splitter1.split(df, labels))

    # Second: split train_val into train and val
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )
    train_idx, val_idx = next(
        splitter2.split(df_train_val, df_train_val[label_col].values)
    )

    train_df = df_train_val.iloc[train_idx].reset_index(drop=True)
    val_df = df_train_val.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"[Split] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}"
    )
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df[label_col].value_counts().sort_index().to_dict()
        logger.info(f"  {split_name} class distribution: {dist}")

    return train_df, val_df, test_df
