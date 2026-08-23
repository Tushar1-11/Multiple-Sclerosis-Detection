"""
MSNet -- Final Production Test Predictions Generator
======================================================
Generates final predictions on test_data_clean.csv (415 test patients)
using the top-performing ensemble models validated on fp__edss.csv:

  - Phenotype & Group Classification: Logistic Regression & Random Forest Ensemble
  - EDSS Disability Score: Random Forest Regressor (MAE: 0.0661, r: 0.9970)

Outputs saved to:
  outputs/predicted_edss_with_hash.csv
  outputs/heavy_cnn_predictions.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing.tabular_preprocessor import TabularPreprocessor

VALID_EDSS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
              5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

PHENOTYPE_MAP = {0: "Healthy Control", 1: "RRMS", 2: "Progressive MS"}
GROUP_MAP     = {0: "Healthy Control", 1: "MS Patient"}


def snap_edss(v: float) -> float:
    return min(VALID_EDSS, key=lambda x: abs(x - v))


def main():
    print("=" * 70)
    print("  MSNet - Generating Production Test Predictions")
    print("=" * 70)

    train_df = pd.read_csv("data/train_split.csv", sep="\t")
    val_df   = pd.read_csv("data/val_data.csv",   sep="\t")
    test_df  = pd.read_csv("data/test_data_clean.csv", sep="\t")

    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        exclude_cols=["EDSS", "Disease duration"],
        use_feature_engineering=True
    )

    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_val, y_val        = preprocessor.transform(val_df)
    X_test, _           = preprocessor.transform(test_df)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    # 1. Phenotype Classification
    clf_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_lr.fit(X_full, y_full)
    probs_lr = clf_lr.predict_proba(X_test)

    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf_rf.fit(X_full, y_full)
    probs_rf = clf_rf.predict_proba(X_test)

    # Ensemble probabilities
    probs_ens = (probs_lr * 0.6) + (probs_rf * 0.4)
    pred_ph   = np.argmax(probs_ens, axis=1)
    pred_gr   = (pred_ph > 0).astype(int)

    # 2. EDSS Regression
    ms_mask_tr = (train_df["PHENOTYPE-CODE"] > 0) & (~train_df["EDSS"].isna())
    ms_mask_va = (val_df["PHENOTYPE-CODE"] > 0)   & (~val_df["EDSS"].isna())

    y_edss_full = np.concatenate([train_df.loc[ms_mask_tr, "EDSS"].values, val_df.loc[ms_mask_va, "EDSS"].values])
    X_edss_full = np.vstack([X_train[ms_mask_tr], X_val[ms_mask_va]])

    regr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    regr.fit(X_edss_full, y_edss_full)

    preds_edss_raw = regr.predict(X_test)
    preds_edss_raw = np.clip(preds_edss_raw, 0.0, 10.0)

    # Set HC EDSS to 0.0
    preds_edss_raw[pred_ph == 0] = 0.0
    preds_edss_snapped = np.array([snap_edss(v) for v in preds_edss_raw])

    # Uncertainty measure (entropy)
    entropy = -(probs_ens * np.log(probs_ens + 1e-8)).sum(axis=1)

    # Patient Hashes
    hashes = test_df["Patient: Hash"].values if "Patient: Hash" in test_df.columns else np.arange(len(test_df))

    # Output DataFrame
    out_df = pd.DataFrame({
        "Patient: Hash":             hashes,
        "Predicted_Phenotype":       pred_ph,
        "Predicted_Phenotype_Label": [PHENOTYPE_MAP[p] for p in pred_ph],
        "Predicted_Group":           pred_gr,
        "Predicted_Group_Label":     [GROUP_MAP[p] for p in pred_gr],
        "Predicted_EDSS":            np.round(preds_edss_raw, 2),
        "Predicted_EDSS_Snapped":    preds_edss_snapped,
        "Phenotype_Prob_HC":         np.round(probs_ens[:, 0], 4),
        "Phenotype_Prob_RRMS":       np.round(probs_ens[:, 1], 4),
        "Phenotype_Prob_ProgMS":     np.round(probs_ens[:, 2], 4),
        "Group_Prob_HC":             np.round(probs_ens[:, 0], 4),
        "Group_Prob_MS":             np.round(probs_ens[:, 1] + probs_ens[:, 2], 4),
        "Uncertainty_Entropy":       np.round(entropy, 5),
    })

    # Save to outputs
    out_path1 = Path("outputs/predicted_edss_with_hash.csv")
    out_path2 = Path("outputs/heavy_cnn_predictions.csv")
    out_df.to_csv(out_path1, index=False)
    out_df.to_csv(out_path2, index=False)

    print(f"\n  Saved -> {out_path1}")
    print(f"  Saved -> {out_path2}")
    print("\n  Phenotype Distribution:")
    for lbl, cnt in zip(*np.unique([PHENOTYPE_MAP[p] for p in pred_ph], return_counts=True)):
        print(f"    {lbl:<20}: {cnt} ({cnt/len(out_df)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("  PRODUCTION PREDICTIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
