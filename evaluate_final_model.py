"""
MSNet -- Final Held-Out Test Evaluation & Learning Curve Diagnostics
======================================================================
1. Computes empirical learning curves (10% to 100% of training data)
   to diagnose overfitting / underfitting.
2. Performs the SINGLE FINAL test evaluation on the held-out test set
   (data/fp__edss.csv - 415 patients).
3. Evaluates full metrics:
   - Classification: Accuracy, Precision, Recall, F1 (Macro/Weighted),
                     ROC-AUC (OvR), Confusion Matrix
   - Regression: MAE, MSE, RMSE, R², Pearson r
4. Saves full evaluation report to outputs/final_evaluation_report.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("MSNet.FinalEval")

EXCLUDE_COLS = ["EDSS", "Disease duration"]
PHENOTYPE_LABELS = ["Healthy Control", "RRMS", "Progressive MS"]


def main():
    set_seed(42)
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("  MSNet -- Final Held-Out Test Evaluation & Learning Curve Suite")
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

    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_val, y_val        = preprocessor.transform(val_df)
    X_test, y_test      = preprocessor.transform(test_df)

    # Production full training set (train + val)
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    logger.info(f"Full Training Cohort : {X_full.shape[0]} samples | {X_full.shape[1]} engineered features")
    logger.info(f"Held-Out Test Cohort : {X_test.shape[0]} patients (data/fp__edss.csv)")
    logger.info("=" * 75)

    # ── 2. Learning Curve Analysis (Overfitting / Underfitting Check) ─
    logger.info("\n>>> Computing Learning Curves (Train vs Validation performance by sample size)...")
    clf_model = SVC(C=10.83, kernel="linear", gamma=0.1, probability=True, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(  # type: ignore
        clf_model, X_full, y_full,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5, scoring="f1_macro", n_jobs=-1, random_state=42
    )

    lc_data = []
    logger.info(f"{'Train Size':<12} | {'Train F1':<12} | {'Val F1':<12} | {'Overfit Gap':<12}")
    logger.info("-" * 55)
    for sz, tr, va in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        gap = tr - va
        logger.info(f"{sz:<12} | {tr:.4f}       | {va:.4f}     | {gap:.4f}")
        lc_data.append({
            "train_size": int(sz),
            "train_f1_mean": float(tr),
            "val_f1_mean": float(va),
            "overfit_gap": float(gap)
        })

    # ── 3. SINGLE HELD-OUT TEST EVALUATION ──────────────────────────
    logger.info("\n>>> Performing SINGLE FINAL evaluation on held-out test set...")
    
    # Fit final tuned model on full training cohort
    clf_model.fit(X_full, y_full)
    
    test_preds = clf_model.predict(X_test)
    test_probs = clf_model.predict_proba(X_test)

    # Classification Metrics
    acc  = accuracy_score(y_test, test_preds)
    bacc = balanced_accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    weighted_f1 = f1_score(y_test, test_preds, average="weighted")
    auc = roc_auc_score(y_test, test_probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_test, test_preds)

    prec, rec, f1_cls, supp = precision_recall_fscore_support(y_test, test_preds)

    per_class_metrics = {}
    for i, lbl in enumerate(PHENOTYPE_LABELS):
        per_class_metrics[lbl] = {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1_score":  float(f1_cls[i]),
            "support":   int(supp[i])
        }

    logger.info("\n" + "=" * 70)
    logger.info("  FINAL HELD-OUT TEST CLASSIFICATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Accuracy           : {acc * 100:.2f}%")
    logger.info(f"  Balanced Accuracy  : {bacc * 100:.2f}%")
    logger.info(f"  Macro F1           : {macro_f1:.4f}")
    logger.info(f"  Weighted F1        : {weighted_f1:.4f}")
    logger.info(f"  ROC-AUC            : {auc:.4f}")
    logger.info("=" * 70)
    logger.info(f"Confusion Matrix:\n{cm}")

    # ── 4. EDSS REGRESSION EVALUATION ─────────────────────────────────
    logger.info("\n>>> Evaluating EDSS Regression on held-out MS test patients...")
    ms_mask_tr = (train_df["PHENOTYPE-CODE"] > 0) & (~train_df["EDSS"].isna())
    ms_mask_va = (val_df["PHENOTYPE-CODE"] > 0)   & (~val_df["EDSS"].isna())
    ms_mask_te = (test_df["PHENOTYPE-CODE"] > 0)  & (~test_df["EDSS"].isna())

    y_edss_full = np.concatenate([train_df.loc[ms_mask_tr, "EDSS"].values, val_df.loc[ms_mask_va, "EDSS"].values])
    X_edss_full = np.vstack([X_train[ms_mask_tr], X_val[ms_mask_va]])

    y_edss_test = test_df.loc[ms_mask_te, "EDSS"].values.astype(np.float32)
    X_edss_test = X_test[ms_mask_te]

    regr_model = RandomForestRegressor(n_estimators=150, max_depth=13, min_samples_split=4, random_state=42)
    regr_model.fit(X_edss_full, y_edss_full)

    edss_preds = regr_model.predict(X_edss_test)
    edss_preds = np.clip(edss_preds, 0.0, 10.0)

    mae  = float(mean_absolute_error(y_edss_test, edss_preds))
    mse  = float(mean_squared_error(y_edss_test, edss_preds))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_edss_test, edss_preds))
    corr = float(np.corrcoef(y_edss_test, edss_preds)[0, 1])

    logger.info("\n" + "=" * 70)
    logger.info("  FINAL HELD-OUT TEST EDSS REGRESSION RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Mean Absolute Error (MAE)  : {mae:.4f}")
    logger.info(f"  Mean Squared Error (MSE)   : {mse:.4f}")
    logger.info(f"  Root Mean Sq Error (RMSE)  : {rmse:.4f}")
    logger.info(f"  R² Score                   : {r2:.4f}")
    logger.info(f"  Pearson Correlation (r)    : {corr:.4f}")
    logger.info("=" * 70)

    # ── 5. Save Report ────────────────────────────────────────────────
    report_data = {
        "learning_curves": lc_data,
        "classification": {
            "accuracy":          float(acc),
            "balanced_accuracy": float(bacc),
            "macro_f1":          float(macro_f1),
            "weighted_f1":       float(weighted_f1),
            "roc_auc":           float(auc),
            "confusion_matrix":  cm.tolist(),
            "per_class":         per_class_metrics
        },
        "regression": {
            "mae":       mae,
            "mse":       mse,
            "rmse":      rmse,
            "r2":        r2,
            "pearson_r": corr
        }
    }

    report_json = save_dir / "final_evaluation_report.json"
    with open(report_json, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"\nFinal evaluation report saved -> {report_json}")


if __name__ == "__main__":
    main()
