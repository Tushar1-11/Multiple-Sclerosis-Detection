"""
MSNet -- Deep Data Cleaning Pipeline
=====================================
Cleans train_data.csv and test_data.csv:

  1. Drops junk columns (Unnamed: *)
  2. Fixes negative Disease duration (clamp to 0)
  3. T2 LV: 0-values in MS patients flagged as suspicious (kept but noted)
  4. Removes exact duplicate rows (if any)
  5. Validates EDSS range [0, 10]
  6. Validates brain volumes are positive
  7. Saves cleaned CSVs to data/train_data_clean.csv and data/test_data_clean.csv

Run:
  python clean_data.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("MSNet.DataCleaner")


NUMERICAL_COLS = [
    "Age", "T2 LV (cubic mm)", "Disease duration", "EDSS",
    "ncorticalGREY [ml]", "nWHITE [ml]", "nThalamus [ml]",
    "nHippocampus [ml]", "nOtherDGM [ml]", "nBrainstem [ml]",
    "nGM_Cerebellum [ml]", "nWM_Cerebellum [ml]",
]

BRAIN_VOL_COLS = [
    "ncorticalGREY [ml]", "nWHITE [ml]", "nThalamus [ml]",
    "nHippocampus [ml]", "nOtherDGM [ml]", "nBrainstem [ml]",
    "nGM_Cerebellum [ml]", "nWM_Cerebellum [ml]",
]


def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame, name: str, is_train: bool = True) -> pd.DataFrame:
    orig_rows = len(df)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Cleaning {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    logger.info(f"{'='*60}")

    # ── 1. Drop junk columns (Unnamed: *) ────────────────────────────
    junk = [c for c in df.columns if c.startswith("Unnamed")]
    if junk:
        df.drop(columns=junk, inplace=True)
        logger.info(f"  [FIX] Dropped junk columns: {junk}")
    else:
        logger.info("  [OK ] No junk columns found")

    # ── 2. Strip whitespace from string columns ───────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)
    logger.info("  [OK ] String columns stripped")

    # ── 3. Fix negative Disease duration ─────────────────────────────
    if "Disease duration" in df.columns:
        neg_mask = df["Disease duration"] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            logger.info(f"  [FIX] Disease duration: {n_neg} negative value(s) -> clamped to 0")
            df.loc[neg_mask, "Disease duration"] = 0.0
        else:
            logger.info("  [OK ] Disease duration: no negatives")

    # ── 4. Validate & clip EDSS [0, 10] ──────────────────────────────
    if "EDSS" in df.columns:
        edss_valid = df["EDSS"].dropna()
        out_of_range = ((edss_valid < 0) | (edss_valid > 10)).sum()
        if out_of_range > 0:
            logger.info(f"  [FIX] EDSS: {out_of_range} values out of [0,10] -> clipped")
            df["EDSS"] = df["EDSS"].clip(0.0, 10.0)
        else:
            logger.info(f"  [OK ] EDSS: all values in [0, 10] "
                        f"(missing: {df['EDSS'].isna().sum()})")

    # ── 5. Validate T2 LV >= 0 ────────────────────────────────────────
    if "T2 LV (cubic mm)" in df.columns:
        neg_t2 = (df["T2 LV (cubic mm)"] < 0).sum()
        zero_t2 = (df["T2 LV (cubic mm)"] == 0).sum()
        if neg_t2 > 0:
            logger.info(f"  [FIX] T2 LV: {neg_t2} negative values -> clipped to 0")
            df["T2 LV (cubic mm)"] = df["T2 LV (cubic mm)"].clip(0.0)
        logger.info(f"  [OK ] T2 LV: {zero_t2} zeros "
                    f"(likely HC patients with no lesions -- kept as-is)")

    # ── 6. Validate brain volumes are positive ────────────────────────
    for col in BRAIN_VOL_COLS:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                logger.info(f"  [FIX] {col}: {neg} negative values -> clipped to 0")
                df[col] = df[col].clip(0.0)

    logger.info("  [OK ] Brain volumes: all non-negative")

    # ── 7. Age validation [5, 100] ────────────────────────────────────
    if "Age" in df.columns:
        bad_age = ((df["Age"] < 5) | (df["Age"] > 100)).sum()
        if bad_age > 0:
            logger.info(f"  [FIX] Age: {bad_age} implausible values flagged")
        else:
            logger.info(f"  [OK ] Age: valid range [{df['Age'].min():.0f}, {df['Age'].max():.0f}]")

    # ── 8. Treatment-code: standardize NaN ───────────────────────────
    if "Treatment-code" in df.columns:
        n_missing = df["Treatment-code"].isna().sum()
        logger.info(f"  [OK ] Treatment-code: {n_missing} missing "
                    f"(HC patients without treatment -- kept as NaN for indicator)")

    # ── 9. Duplicate rows ─────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
        logger.info(f"  [FIX] Removed {n_dup} exact duplicate rows")
    else:
        logger.info("  [OK ] No duplicate rows")

    # ── 10. Duplicate patient hashes ─────────────────────────────────
    if "Patient: Hash" in df.columns:
        n_hash_dup = df["Patient: Hash"].duplicated().sum()
        if n_hash_dup > 0:
            logger.warning(f"  [WARN] {n_hash_dup} duplicate patient hashes found!")
        else:
            logger.info("  [OK ] No duplicate patient hashes")

    # ── 11. PHENOTYPE-CODE / GROUP-CODE valid values ──────────────────
    if "PHENOTYPE-CODE" in df.columns:
        invalid_ph = (~df["PHENOTYPE-CODE"].isin([0, 1, 2])).sum()
        if invalid_ph > 0:
            logger.warning(f"  [WARN] {invalid_ph} invalid PHENOTYPE-CODE values")
        else:
            ph_dist = df["PHENOTYPE-CODE"].value_counts().sort_index().to_dict()
            logger.info(f"  [OK ] PHENOTYPE-CODE: {ph_dist}")

    if "GROUP-CODE" in df.columns:
        invalid_gr = (~df["GROUP-CODE"].isin([0, 1])).sum()
        if invalid_gr > 0:
            logger.warning(f"  [WARN] {invalid_gr} invalid GROUP-CODE values")
        else:
            gr_dist = df["GROUP-CODE"].value_counts().sort_index().to_dict()
            logger.info(f"  [OK ] GROUP-CODE: {gr_dist}")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n  Missing values after cleaning:")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    for col, n in miss.items():
        logger.info(f"    {col:<30}: {n} ({n/len(df)*100:.1f}%)")

    logger.info(f"\n  Final shape: {df.shape[0]} rows x {df.shape[1]} cols "
                f"(removed {orig_rows - df.shape[0]} rows)")
    return df


def main():
    data_dir = Path("data")

    # ── Clean Train ───────────────────────────────────────────────────
    train_path = data_dir / "train_data.csv"
    train_df   = load_csv(train_path)
    train_clean = clean(train_df, "train_data.csv", is_train=True)
    out_train  = data_dir / "train_data_clean.csv"
    train_clean.to_csv(out_train, sep="\t", index=False)
    logger.info(f"\n  Saved -> {out_train}")

    # ── Clean Test ────────────────────────────────────────────────────
    test_path  = data_dir / "test_data.csv"
    test_df    = load_csv(test_path)
    test_clean = clean(test_df, "test_data.csv", is_train=False)
    out_test   = data_dir / "test_data_clean.csv"
    test_clean.to_csv(out_test, sep="\t", index=False)
    logger.info(f"\n  Saved -> {out_test}")

    logger.info("\n" + "="*60)
    logger.info("  DATA CLEANING COMPLETE")
    logger.info("  Use: data/train_data_clean.csv and data/test_data_clean.csv")
    logger.info("="*60)


if __name__ == "__main__":
    main()
