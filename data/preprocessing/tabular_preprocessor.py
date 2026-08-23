"""
MSNet - Tabular Preprocessor for MS Clinical & Volumetric Data
==============================================================
Handles preprocessing for `train_data.csv`:
  - Optional domain-driven feature engineering (25+ new features)
  - Loading tab or comma-separated CSV files
  - Missing value imputation and indicator creation
  - Feature scaling (StandardScaler)
  - One-hot / categorical encoding
  - Target extraction (GROUP-CODE or PHENOTYPE-CODE)
  - Stratified train/val/test splitting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from data.preprocessing.feature_engineer import MSFeatureEngineer

logger = logging.getLogger(__name__)

# Feature column definitions
NUMERICAL_COLS = [
    "Age",
    "T2 LV (cubic mm)",
    "Disease duration",
    "EDSS",
    "ncorticalGREY [ml]",
    "nWHITE [ml]",
    "nThalamus [ml]",
    "nHippocampus [ml]",
    "nOtherDGM [ml]",
    "nBrainstem [ml]",
    "nGM_Cerebellum [ml]",
    "nWM_Cerebellum [ml]",
]

CATEGORICAL_COLS = [
    "SEX-CODE",
    "INSTITUTION-CODE",
    "Treatment-code",
]

TARGET_COLS = {
    "phenotype": "PHENOTYPE-CODE",  # 0=HC, 1=RRMS, 2=Progressive MS
    "group": "GROUP-CODE",          # 0=HC, 1=MS Patient
}


class TabularPreprocessor:
    """
    Preprocessor for MS clinical and volumetric dataset.
    """

    def __init__(
        self,
        target_name: str = "phenotype",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        exclude_cols: Optional[List[str]] = None,
        use_feature_engineering: bool = True,
    ):
        """
        Args:
            target_name: 'phenotype' (3-class) or 'group' (binary)
            test_size: Ratio of test set
            val_size: Ratio of validation set
            random_state: Seed for reproducibility
            exclude_cols: List of numerical column names to exclude from features.
                          Useful for removing columns that create biased missing-value
                          indicators (e.g. 'EDSS', 'Disease duration').
            use_feature_engineering: If True, applies MSFeatureEngineer before
                                     scaling to add 25+ domain-driven features.
        """
        self.target_name = target_name
        self.target_col = TARGET_COLS.get(target_name.lower(), "PHENOTYPE-CODE")
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.exclude_cols: List[str] = exclude_cols or []
        self.use_feature_engineering = use_feature_engineering
        self.feature_engineer = MSFeatureEngineer() if use_feature_engineering else None

        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.fitted = False
        self.feature_names: List[str] = []
        self.num_features: int = 0

    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load CSV data with auto delimiter detection."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")

        # Try reading tab-separated first, fallback to comma-separated
        try:
            df = pd.read_csv(data_path, sep="\t")
            if df.shape[1] <= 1:
                df = pd.read_csv(data_path, sep=",")
        except Exception:
            df = pd.read_csv(data_path, sep=",")

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded dataset from {data_path} with shape {df.shape}")
        return df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Fit preprocessor on dataframe and return features and target.

        Returns:
            X: Normalized feature matrix [N, num_features]
            y: Target label vector [N]
            split_indices: Dictionary of train, val, test indices
        """
        df_clean = df.copy()

        # Apply domain-driven feature engineering (adds feat_* columns)
        if self.use_feature_engineering and self.feature_engineer is not None:
            df_clean = self.feature_engineer.transform(df_clean)
            logger.info(f"Feature engineering applied: {len(self.feature_engineer.get_feature_names())} new features added")

        # Extract target
        if self.target_col not in df_clean.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in dataset.")

        y = df_clean[self.target_col].values.astype(np.int64)

        # Handle missing values — process base + engineered numerical cols
        num_processed = []
        num_feature_names = []
        # Combine base numericals with engineered feat_* columns
        eng_feat_cols = (
            [c for c in df_clean.columns if c.startswith("feat_")]
            if self.use_feature_engineering else []
        )
        active_num_cols = (
            [c for c in NUMERICAL_COLS if c not in self.exclude_cols]
            + eng_feat_cols
        )
        for col in active_num_cols:
            if col in df_clean.columns:
                series = df_clean[col].copy()
                has_na = series.isna().any()
                # Impute missing values with 0 (e.g. for healthy controls missing EDSS/disease duration)
                imputed = series.fillna(0.0).values.reshape(-1, 1)
                num_processed.append(imputed)
                num_feature_names.append(col)

                # Create binary indicator if column had NaNs
                if has_na:
                    indicator = series.isna().astype(float).values.reshape(-1, 1)
                    num_processed.append(indicator)
                    num_feature_names.append(f"{col}_missing")

        X_num = np.hstack(num_processed)
        X_num_scaled = self.scaler.fit_transform(X_num)

        # Process categoricals
        cat_processed = []
        cat_feature_names = []
        for col in CATEGORICAL_COLS:
            if col in df_clean.columns:
                series = df_clean[col].fillna(-1).astype(str).values.reshape(-1, 1)
                cat_processed.append(series)

        if cat_processed:
            X_cat_raw = np.hstack(cat_processed)
            X_cat_ohe = self.ohe.fit_transform(X_cat_raw)
            cat_feature_names = list(self.ohe.get_feature_names_out(CATEGORICAL_COLS))
            X_all = np.hstack([X_num_scaled, X_cat_ohe])
        else:
            X_all = X_num_scaled

        self.feature_names = num_feature_names + cat_feature_names
        self.num_features = X_all.shape[1]
        self.fitted = True

        logger.info(f"Preprocessed {self.num_features} features for {len(y)} samples.")

        # Create stratified train / val / test split
        indices = np.arange(len(y))
        train_val_idx, test_idx, y_train_val, _ = train_test_split(
            indices, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        relative_val_size = self.val_size / (1.0 - self.test_size)
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx, y_train_val, test_size=relative_val_size, random_state=self.random_state, stratify=y_train_val
        )

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        logger.info(
            f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )

        return X_all.astype(np.float32), y, splits

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform new dataframe using fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df_clean = df.copy()

        # Apply feature engineering (must mirror fit_transform)
        use_feat_eng = getattr(self, "use_feature_engineering", True)
        if use_feat_eng and getattr(self, "feature_engineer", None) is not None:
            df_clean = self.feature_engineer.transform(df_clean)

        # Extract target if present
        y = None
        if self.target_col in df_clean.columns:
            y = df_clean[self.target_col].values.astype(np.int64)

        # Process numericals — base + engineered
        num_processed = []
        eng_feat_cols = (
            [c for c in df_clean.columns if c.startswith("feat_")]
            if use_feat_eng else []
        )
        exclude_cols = getattr(self, "exclude_cols", set())
        active_num_cols = (
            [c for c in NUMERICAL_COLS if c not in exclude_cols]
            + eng_feat_cols
        )
        for col in active_num_cols:
            if col in df_clean.columns:
                series = df_clean[col].copy()
                imputed = series.fillna(0.0).values.reshape(-1, 1)
                num_processed.append(imputed)

                # Check if missing indicator was included during fit
                if f"{col}_missing" in self.feature_names:
                    indicator = series.isna().astype(float).values.reshape(-1, 1)
                    num_processed.append(indicator)

        X_num = np.hstack(num_processed)
        X_num_scaled = self.scaler.transform(X_num)

        # Process categoricals
        cat_processed = []
        for col in CATEGORICAL_COLS:
            if col in df_clean.columns:
                series = df_clean[col].fillna(-1).astype(str).values.reshape(-1, 1)
                cat_processed.append(series)

        if cat_processed:
            X_cat_raw = np.hstack(cat_processed)
            X_cat_ohe = self.ohe.transform(X_cat_raw)
            X_all = np.hstack([X_num_scaled, X_cat_ohe])
        else:
            X_all = X_num_scaled

        return X_all.astype(np.float32), y
