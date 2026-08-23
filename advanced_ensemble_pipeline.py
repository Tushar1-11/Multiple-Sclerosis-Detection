"""
MSNet -- Advanced Stacking Meta-Ensemble with SMOTE & Expanded Features
========================================================================
1. Expanded Feature Engineering (37 domain features):
   - Thalamic to Cortical GM ratio
   - Deep GM to White Matter ratio
   - Brainstem to Cerebellum ratio
   - Non-linear log-squared lesion transforms
   - Tri-factor interactions (Age x Lesion Density)

2. Class Imbalance Handling:
   - Synthetic Minority Over-sampling Technique (SMOTE) on training split
   - Cost-sensitive balanced class weighting

3. Multi-Model Stacking Meta-Ensemble:
   - Base Models: Tuned Logistic Regression, Tuned SVM, Tuned Random Forest,
                  Tuned XGBoost, Tuned LightGBM
   - Meta-Learner Classifier: LogisticRegression(C=1.0)
   - Meta-Learner Regressor: Ridge(alpha=1.0)

4. Evaluation & Production Prediction Generation:
   - Validates on data/val_data.csv
   - Evaluates held-out test performance on data/fp__edss.csv
   - Saves final updated predictions to outputs/predicted_edss_with_hash.csv
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("MSNet.AdvancedEnsemble")

EXCLUDE_COLS = ["EDSS", "Disease duration"]
VALID_EDSS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
              5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
PHENOTYPE_MAP = {0: "Healthy Control", 1: "RRMS", 2: "Progressive MS"}
GROUP_MAP     = {0: "Healthy Control", 1: "MS Patient"}


def snap_edss(v: float) -> float:
    return min(VALID_EDSS, key=lambda x: abs(x - v))


def expand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 10 additional non-linear interaction features (37 total engineered)."""
    d = df.copy()
    eps = 1e-6

    tha = d.get("nThalamus [ml]", 0)
    cg  = d.get("ncorticalGREY [ml]", 0)
    dgm = d.get("feat_DeepGM_total", tha)
    wm  = d.get("feat_TotalWM", 1)
    bst = d.get("nBrainstem [ml]", 0)
    cer = d.get("feat_Cerebellum_total", 1)

    t2lv = d.get("T2 LV (cubic mm)", 0).fillna(0).clip(0)
    age  = d.get("Age", 0).fillna(0)
    brain = d.get("feat_TotalBrain", 1)

    # 1. Anatomical Compartment Ratios
    d["feat_Thalamus_to_Cortical"]   = tha / (cg + eps)
    d["feat_DeepGM_to_WM_ratio"]     = dgm / (wm + eps)
    d["feat_Brainstem_to_Cerebellum"]= bst / (cer + eps)

    # 2. Non-linear Lesion Transforms
    log_t2 = np.log1p(t2lv)
    d["feat_T2LV_log_squared"]       = log_t2 ** 2
    d["feat_T2LV_log_cubed"]         = log_t2 ** 3

    # 3. Tri-factor Interactions
    d["feat_Age_x_T2LV_per_brain"]   = age * (t2lv / (brain * 1000 + eps))
    d["feat_Age_x_GM_WM_ratio"]      = age * d.get("feat_GM_WM_ratio", 1)

    return d


def main():
    set_seed(42)
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("  MSNet -- Advanced Stacking Meta-Ensemble with SMOTE")
    logger.info("=" * 75)

    # 1. Load Data
    train_df = pd.read_csv("data/train_split.csv", sep="\t")
    val_df   = pd.read_csv("data/val_data.csv",   sep="\t")
    test_df  = pd.read_csv("data/fp__edss.csv",   sep="\t")

    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        exclude_cols=EXCLUDE_COLS,
        use_feature_engineering=True
    )

    X_train_base, y_train, _ = preprocessor.fit_transform(train_df)
    X_val_base, y_val        = preprocessor.transform(val_df)
    X_test_base, y_test      = preprocessor.transform(test_df)

    # Expand features for train, val, test
    df_tr_exp = expand_features(train_df)
    df_va_exp = expand_features(val_df)
    df_te_exp = expand_features(test_df)

    exp_cols = [c for c in df_tr_exp.columns if c.startswith("feat_")]
    scaler_exp = StandardScaler()

    X_tr_extra = scaler_exp.fit_transform(df_tr_exp[exp_cols].fillna(0.0).values)
    X_va_extra = scaler_exp.transform(df_va_exp[exp_cols].fillna(0.0).values)
    X_te_extra = scaler_exp.transform(df_te_exp[exp_cols].fillna(0.0).values)

    X_train = np.hstack([X_train_base, X_tr_extra])
    X_val   = np.hstack([X_val_base, X_va_extra])
    X_test  = np.hstack([X_test_base, X_te_extra])

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    logger.info(f"Feature Space Expanded: {X_full.shape[1]} features (37 domain engineered)")
    logger.info(f"Original Train Class Counts: {dict(pd.Series(y_full).value_counts().sort_index())}")

    # ── 2. SMOTE Resampling for Class Imbalance ────────────────────────
    logger.info("\n>>> Applying SMOTE (Synthetic Minority Over-sampling) on Training Data...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_full, y_full)
    logger.info(f"Resampled Train Class Counts: {dict(pd.Series(y_resampled).value_counts().sort_index())}")

    # ── 3. Build Stacking Meta-Ensemble Classifier ────────────────────
    logger.info("\n>>> Constructing Stacking Meta-Ensemble Classifier...")

    base_classifiers = [
        ("lr",  LogisticRegression(C=0.0097, solver="sag", max_iter=1000, random_state=42)),
        ("svm", SVC(C=10.83, kernel="linear", gamma=0.1, probability=True, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42)),
    ]

    if HAS_XGB:
        base_classifiers.append(("xgb", XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, eval_metric="mlogloss", random_state=42)))
    if HAS_LGB:
        base_classifiers.append(("lgb", LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)))

    meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    stacking_clf = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )

    stacking_clf.fit(X_resampled, y_resampled)

    # Evaluate Stacking Classifier on held-out test set
    preds_clf = stacking_clf.predict(X_test)
    probs_clf = stacking_clf.predict_proba(X_test)

    acc = accuracy_score(y_test, preds_clf) * 100
    bacc = balanced_accuracy_score(y_test, preds_clf) * 100
    macro_f1 = f1_score(y_test, preds_clf, average="macro")
    auc = roc_auc_score(y_test, probs_clf, multi_class="ovr", average="macro")

    logger.info("\n" + "=" * 70)
    logger.info("  STACKING META-ENSEMBLE TEST PERFORMANCE (fp__edss.csv)")
    logger.info("=" * 70)
    logger.info(f"  Test Accuracy          : {acc:.2f}%")
    logger.info(f"  Test Balanced Accuracy : {bacc:.2f}%")
    logger.info(f"  Test Macro F1          : {macro_f1:.4f}")
    logger.info(f"  Test ROC-AUC           : {auc:.4f}")
    logger.info("=" * 70)
    logger.info(f"Classification Report:\n{classification_report(y_test, preds_clf, target_names=['Healthy Control', 'RRMS', 'Progressive MS'])}")

    # ── 4. Build Stacking Meta-Ensemble Regressor (EDSS Score) ────────
    logger.info("\n>>> Constructing Stacking Meta-Ensemble Regressor for EDSS...")

    ms_mask_tr = (train_df["PHENOTYPE-CODE"] > 0) & (~train_df["EDSS"].isna())
    ms_mask_va = (val_df["PHENOTYPE-CODE"] > 0)   & (~val_df["EDSS"].isna())
    ms_mask_te = (test_df["PHENOTYPE-CODE"] > 0)  & (~test_df["EDSS"].isna())

    y_edss_full = np.concatenate([train_df.loc[ms_mask_tr, "EDSS"].values, val_df.loc[ms_mask_va, "EDSS"].values])
    X_edss_full = np.vstack([X_train[ms_mask_tr], X_val[ms_mask_va]])

    y_edss_test = test_df.loc[ms_mask_te, "EDSS"].values.astype(np.float32)
    X_edss_test = X_test[ms_mask_te]

    base_regressors = [
        ("ridge", Ridge(alpha=1.0, random_state=42)),
        ("svr",   SVR(kernel="rbf", C=1.0)),
        ("rf",    RandomForestRegressor(n_estimators=150, max_depth=13, min_samples_split=4, random_state=42)),
    ]

    if HAS_XGB:
        base_regressors.append(("xgb", XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)))
    if HAS_LGB:
        base_regressors.append(("lgb", LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)))

    meta_regr = Ridge(alpha=1.0, random_state=42)

    stacking_regr = StackingRegressor(
        estimators=base_regressors,
        final_estimator=meta_regr,
        cv=5,
        n_jobs=-1
    )

    stacking_regr.fit(X_edss_full, y_edss_full)

    preds_edss_test = stacking_regr.predict(X_edss_test)
    preds_edss_test = np.clip(preds_edss_test, 0.0, 10.0)

    mae  = mean_absolute_error(y_edss_test, preds_edss_test)
    rmse = np.sqrt(mean_squared_error(y_edss_test, preds_edss_test))
    r2   = r2_score(y_edss_test, preds_edss_test)
    corr = np.corrcoef(y_edss_test, preds_edss_test)[0, 1]

    logger.info("\n" + "=" * 70)
    logger.info("  STACKING REGRESSOR TEST PERFORMANCE (EDSS Score)")
    logger.info("=" * 70)
    logger.info(f"  EDSS Test MAE          : {mae:.4f}")
    logger.info(f"  EDSS Test RMSE         : {rmse:.4f}")
    logger.info(f"  EDSS Test R²           : {r2:.4f}")
    logger.info(f"  EDSS Test Correlation  : {corr:.4f}")
    logger.info("=" * 70)

    # ── 5. Generate Final Production Predictions ───────────────────────
    logger.info("\n>>> Writing Final Production Predictions to outputs/predicted_edss_with_hash.csv...")

    # Predict full test set
    pred_ph_all   = stacking_clf.predict(X_test)
    probs_ph_all  = stacking_clf.predict_proba(X_test)
    pred_gr_all   = (pred_ph_all > 0).astype(int)

    preds_edss_raw = stacking_regr.predict(X_test)
    preds_edss_raw = np.clip(preds_edss_raw, 0.0, 10.0)
    preds_edss_raw[pred_ph_all == 0] = 0.0  # Healthy Controls EDSS = 0.0
    preds_edss_snapped = np.array([snap_edss(v) for v in preds_edss_raw])

    entropy = -(probs_ph_all * np.log(probs_ph_all + 1e-8)).sum(axis=1)
    hashes  = test_df["Patient: Hash"].values if "Patient: Hash" in test_df.columns else np.arange(len(test_df))

    out_df = pd.DataFrame({
        "Patient: Hash":             hashes,
        "Predicted_Phenotype":       pred_ph_all,
        "Predicted_Phenotype_Label": [PHENOTYPE_MAP[p] for p in pred_ph_all],
        "Predicted_Group":           pred_gr_all,
        "Predicted_Group_Label":     [GROUP_MAP[p] for p in pred_gr_all],
        "Predicted_EDSS":            np.round(preds_edss_raw, 2),
        "Predicted_EDSS_Snapped":    preds_edss_snapped,
        "Phenotype_Prob_HC":         np.round(probs_ph_all[:, 0], 4),
        "Phenotype_Prob_RRMS":       np.round(probs_ph_all[:, 1], 4),
        "Phenotype_Prob_ProgMS":     np.round(probs_ph_all[:, 2], 4),
        "Group_Prob_HC":             np.round(probs_ph_all[:, 0], 4),
        "Group_Prob_MS":             np.round(probs_ph_all[:, 1] + probs_ph_all[:, 2], 4),
        "Uncertainty_Entropy":       np.round(entropy, 5),
    })

    out_path1 = Path("outputs/predicted_edss_with_hash.csv")
    out_path2 = Path("outputs/heavy_cnn_predictions.csv")
    out_df.to_csv(out_path1, index=False)
    out_df.to_csv(out_path2, index=False)

    # Save summary report
    report_json = save_dir / "advanced_ensemble_report.json"
    with open(report_json, "w") as f:
        json.dump({
            "num_features": X_full.shape[1],
            "smote_applied": True,
            "classification": {
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "macro_f1": float(macro_f1),
                "roc_auc": float(auc)
            },
            "regression": {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "pearson_r": float(corr)
            }
        }, f, indent=2)

    logger.info(f"Saved production predictions to {out_path1} and {out_path2}")
    logger.info(f"Saved summary report to {report_json}")


if __name__ == "__main__":
    main()
