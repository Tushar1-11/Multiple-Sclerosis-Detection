"""
MSNet -- Domain-Driven Feature Engineering for MS Clinical Data
================================================================
Generates clinically meaningful features from raw MS biomarkers.

All features are grounded in MS neurology literature:

  BRAIN VOLUME AGGREGATES
    - Total brain volume, total GM, total WM, total deep GM
  
  ATROPHY RATIOS (key MS biomarkers)
    - Grey/White matter ratio
    - Thalamus % of total brain   (thalamic atrophy = MS hallmark)
    - Hippocampus % of total brain (memory-related atrophy)
    - Deep GM fraction             (subcortical atrophy index)
    - Cortical thinning index      (cortical GM / total GM)
    - Cerebellar GM/WM ratio
  
  LESION BURDEN FEATURES
    - log1p(T2 LV)                 (lesion volume is heavily right-skewed)
    - T2 LV per total brain volume (normalized lesion burden)
    - T2 LV per white matter       (WM-relative lesion load)
    - sqrt(T2 LV)                  (variance-stabilizing)

  DISEASE PROGRESSION FEATURES (only where Disease duration is known)
    - EDSS per year                (disability accrual rate)
    - T2 LV per year               (lesion accumulation rate)
    - log1p(Disease duration)

  INTERACTION FEATURES
    - Age x Thalamus ratio         (age-modulated thalamic atrophy)
    - Age x log1p(T2 LV)           (age-lesion interaction)
    - Lesion burden x EDSS         (disability-lesion coupling)

  CEREBELLAR FEATURES
    - Total cerebellar volume
    - Cerebellar fraction of brain
    - Cerebellar GM dominance

Usage:
  from data.preprocessing.feature_engineer import MSFeatureEngineer
  eng = MSFeatureEngineer()
  df_enriched = eng.transform(df)  # adds new columns in-place copy
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MSFeatureEngineer:
    """
    Domain-driven feature engineering for MS clinical + volumetric data.
    Adds ~25 clinically meaningful features to the raw dataframe.
    """

    # Original brain volume columns
    BRAIN_COLS = [
        "ncorticalGREY [ml]",
        "nWHITE [ml]",
        "nThalamus [ml]",
        "nHippocampus [ml]",
        "nOtherDGM [ml]",
        "nBrainstem [ml]",
        "nGM_Cerebellum [ml]",
        "nWM_Cerebellum [ml]",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to a copy of df.

        Args:
            df: Raw dataframe with MS clinical + volumetric columns

        Returns:
            df_eng: Enriched dataframe with added feature columns
        """
        d = df.copy()
        eps = 1e-6  # small constant to avoid division by zero

        # ── 1. BRAIN VOLUME AGGREGATES ────────────────────────────────

        cg  = d.get("ncorticalGREY [ml]",   pd.Series(0, index=d.index))
        wm  = d.get("nWHITE [ml]",          pd.Series(0, index=d.index))
        tha = d.get("nThalamus [ml]",        pd.Series(0, index=d.index))
        hip = d.get("nHippocampus [ml]",     pd.Series(0, index=d.index))
        dgm = d.get("nOtherDGM [ml]",        pd.Series(0, index=d.index))
        bst = d.get("nBrainstem [ml]",       pd.Series(0, index=d.index))
        cgm = d.get("nGM_Cerebellum [ml]",  pd.Series(0, index=d.index))
        cwm = d.get("nWM_Cerebellum [ml]",  pd.Series(0, index=d.index))

        # Deep grey matter (thalamus + hippocampus + other DGM)
        d["feat_DeepGM_total"]       = tha + hip + dgm

        # Total grey matter (cortical + deep + cerebellar)
        d["feat_TotalGM"]            = cg + tha + hip + dgm + cgm

        # Total white matter (WM + cerebellar WM)
        d["feat_TotalWM"]            = wm + cwm

        # Total cerebellar volume
        d["feat_Cerebellum_total"]   = cgm + cwm

        # Total brain (all compartments)
        total_brain = cg + wm + tha + hip + dgm + bst + cgm + cwm
        d["feat_TotalBrain"]         = total_brain

        logger.debug("Added: volume aggregates (DeepGM, TotalGM, TotalWM, Cerebellum, TotalBrain)")

        # ── 2. ATROPHY RATIOS ─────────────────────────────────────────

        # Grey/White matter ratio (global atrophy index)
        d["feat_GM_WM_ratio"]        = d["feat_TotalGM"] / (d["feat_TotalWM"] + eps)

        # Thalamus % of total brain (thalamic atrophy is strongest MS predictor)
        d["feat_Thalamus_frac"]      = tha / (total_brain + eps) * 100

        # Hippocampus % of total brain
        d["feat_Hippocampus_frac"]   = hip / (total_brain + eps) * 100

        # Deep GM fraction (subcortical atrophy index)
        d["feat_DeepGM_frac"]        = d["feat_DeepGM_total"] / (total_brain + eps) * 100

        # Cortical GM fraction (cortical thinning index)
        d["feat_CorticalGM_frac"]    = cg / (total_brain + eps) * 100

        # Cortical GM / Total GM (how much of GM is cortical vs subcortical)
        d["feat_CorticalGM_of_GM"]   = cg / (d["feat_TotalGM"] + eps)

        # Cerebellar GM/WM ratio (cerebellar integrity)
        d["feat_Cerebellum_GMratio"] = cgm / (cwm + eps)

        # Brainstem fraction
        d["feat_Brainstem_frac"]     = bst / (total_brain + eps) * 100

        # Cerebellar fraction of total brain
        d["feat_Cerebellum_frac"]    = d["feat_Cerebellum_total"] / (total_brain + eps) * 100

        logger.debug("Added: atrophy ratios")

        # ── 3. LESION BURDEN FEATURES ─────────────────────────────────

        t2lv = d.get("T2 LV (cubic mm)", pd.Series(0, index=d.index)).fillna(0).clip(0)

        # Log-transform (T2LV is heavily right-skewed)
        d["feat_T2LV_log1p"]         = np.log1p(t2lv)

        # Square root (variance-stabilizing)
        d["feat_T2LV_sqrt"]          = np.sqrt(t2lv + eps)

        # Normalized lesion burden (lesion / total brain volume)
        d["feat_T2LV_per_brain"]     = t2lv / (total_brain * 1000 + eps)  # ml -> mm³ conversion

        # Lesion per white matter (WM-specific lesion load)
        d["feat_T2LV_per_WM"]        = t2lv / (d["feat_TotalWM"] * 1000 + eps)

        logger.debug("Added: lesion burden features")

        # ── 4. DISEASE PROGRESSION FEATURES ──────────────────────────

        dur = d.get("Disease duration", pd.Series(np.nan, index=d.index)).clip(0)
        edss_col = d.get("EDSS", pd.Series(np.nan, index=d.index))

        # Log disease duration (right-skewed)
        d["feat_DiseaseDur_log1p"]   = np.log1p(dur.fillna(0))

        # EDSS per year of disease (disability accrual rate)
        # Only meaningful for MS patients (duration > 0)
        d["feat_EDSS_per_year"]      = (edss_col.fillna(0) /
                                         dur.replace(0, np.nan)).fillna(0).clip(0, 20)

        # T2LV per year of disease (lesion accumulation rate)
        d["feat_T2LV_per_year"]      = (t2lv /
                                         (dur.replace(0, np.nan) + eps)).fillna(0)

        # Log T2LV per year
        d["feat_T2LV_per_year_log"]  = np.log1p(d["feat_T2LV_per_year"])

        logger.debug("Added: disease progression features")

        # ── 5. INTERACTION FEATURES ───────────────────────────────────

        age = d.get("Age", pd.Series(0, index=d.index)).fillna(0)

        # Age x Thalamus fraction (age-modulated thalamic atrophy)
        d["feat_Age_x_Thalamus"]     = age * d["feat_Thalamus_frac"]

        # Age x log1p(T2LV) (older patients accumulate more lesions)
        d["feat_Age_x_T2LV_log"]     = age * d["feat_T2LV_log1p"]

        # Age x Deep GM fraction
        d["feat_Age_x_DeepGM"]       = age * d["feat_DeepGM_frac"]

        # Lesion burden x EDSS (disability-lesion coupling)
        d["feat_T2LV_x_EDSS"]        = d["feat_T2LV_log1p"] * edss_col.fillna(0)

        # GM/WM ratio x Age (age-dependent atrophy)
        d["feat_GMratio_x_Age"]      = d["feat_GM_WM_ratio"] * age

        logger.debug("Added: interaction features")

        # ── 6. SUMMARY ────────────────────────────────────────────────
        new_feats = [c for c in d.columns if c.startswith("feat_")]
        logger.info(f"Feature engineering: added {len(new_feats)} new features "
                    f"(total cols: {len(d.columns)})")

        return d

    def get_feature_names(self) -> List[str]:
        """Return list of all engineered feature names."""
        return [
            # Volume aggregates
            "feat_DeepGM_total", "feat_TotalGM", "feat_TotalWM",
            "feat_Cerebellum_total", "feat_TotalBrain",
            # Atrophy ratios
            "feat_GM_WM_ratio", "feat_Thalamus_frac", "feat_Hippocampus_frac",
            "feat_DeepGM_frac", "feat_CorticalGM_frac", "feat_CorticalGM_of_GM",
            "feat_Cerebellum_GMratio", "feat_Brainstem_frac", "feat_Cerebellum_frac",
            # Lesion burden
            "feat_T2LV_log1p", "feat_T2LV_sqrt",
            "feat_T2LV_per_brain", "feat_T2LV_per_WM",
            # Disease progression
            "feat_DiseaseDur_log1p", "feat_EDSS_per_year",
            "feat_T2LV_per_year", "feat_T2LV_per_year_log",
            # Interactions
            "feat_Age_x_Thalamus", "feat_Age_x_T2LV_log", "feat_Age_x_DeepGM",
            "feat_T2LV_x_EDSS", "feat_GMratio_x_Age",
        ]
