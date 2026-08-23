"""
MSNet -- Multi-Algorithm Cross-Validation Benchmark Suite
===========================================================
Benchmarking ML & DL algorithms for Multiple Sclerosis Diagnosis & EDSS Estimation.

Algorithms Evaluated:
  Classification (Phenotype: HC, RRMS, ProgMS):
    1. Logistic Regression (L2 Multinomial)
    2. Support Vector Machine (SVM - RBF Kernel)
    3. Random Forest Classifier
    4. XGBoost Classifier
    5. LightGBM Classifier
    6. Multi-Layer Perceptron (Neural Network)
    7. CNN + Attention Hybrid (PyTorch DL)

  Regression (EDSS Estimation):
    1. Linear Regression / Ridge
    2. Support Vector Regressor (SVR - RBF)
    3. Random Forest Regressor
    4. XGBoost Regressor
    5. LightGBM Regressor

Evaluation Strategy:
  - 5-Fold Stratified Cross-Validation on train_data_clean.csv (1659 samples)
  - 47 Domain-Engineered features (atrophy ratios, volume aggregates, lesion transforms)
  - Full tracking of fold-level Accuracy, Balanced Accuracy, Macro F1, Weighted F1, ROC-AUC, MAE, RMSE, R2
  - Results saved to outputs/multi_algorithm_benchmark.json and markdown report

Usage:
  python run_multi_algorithm_benchmark.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Try importing XGBoost & LightGBM
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
logger = logging.getLogger("MSNet.Benchmark")

EXCLUDE_COLS = ["EDSS", "Disease duration"]


def evaluate_classification(model, X_train, y_train, X_val, y_val, num_classes=3):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        try:
            auc = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    return {
        "accuracy":          float(accuracy_score(y_val, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, preds)),
        "macro_f1":          float(f1_score(y_val, preds, average="macro", zero_division=0)),
        "weighted_f1":       float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "roc_auc":           float(auc),
    }


def evaluate_regression(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    preds = np.clip(preds, 0.0, 10.0)

    mae  = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2   = float(r2_score(y_val, preds))
    
    corr_matrix = np.corrcoef(y_val, preds)
    corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    return {
        "mae":  mae,
        "rmse": rmse,
        "r2":   r2,
        "pearson_r": corr
    }


def main():
    set_seed(42)
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("  MSNet -- Multi-Algorithm 5-Fold Cross-Validation Benchmark")
    logger.info("=" * 75)

    # 1. Load Data & Preprocess (47 Features)
    data_path = Path("data/train_data_clean.csv")
    if not data_path.exists():
        data_path = Path("data/train_data.csv")

    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        random_state=42,
        exclude_cols=EXCLUDE_COLS,
        use_feature_engineering=True
    )

    df = preprocessor.load_data(data_path)

    # Extract raw EDSS for regression target
    edss_targets = df["EDSS"].fillna(0.0).values.astype(np.float32)
    ms_mask = (df["PHENOTYPE-CODE"].values > 0) & (~df["EDSS"].isna().values)  # MS patients with known EDSS

    X, y_cls, _ = preprocessor.fit_transform(df)
    n_samples, n_features = X.shape

    logger.info(f"Dataset Loaded: {n_samples} samples | {n_features} features (47 engineered)")
    logger.info(f"Target Distribution: {dict(pd.Series(y_cls).value_counts().sort_index())}")
    logger.info(f"MS Patients for EDSS Regression: {ms_mask.sum()} samples")
    logger.info("=" * 75)

    # 2. Define Classification Models
    clf_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM (RBF Kernel)":    SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
    }

    if HAS_XGB:
        clf_models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric="mlogloss", random_state=42)
    if HAS_LGB:
        clf_models["LightGBM"] = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)

    # 3. Define Regression Models
    reg_models = {
        "Ridge Regression":      Ridge(alpha=1.0, random_state=42),
        "Support Vector Regr":   SVR(kernel="rbf", C=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Neural Network Regr (MLP)": MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
    }

    if HAS_XGB:
        reg_models["XGBoost Regressor"] = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    if HAS_LGB:
        reg_models["LightGBM Regressor"] = LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)

    # 4. Stratified 5-Fold CV Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Classification Benchmark ──────────────────────────────────────
    logger.info("\n>>> Running 5-Fold Stratified CV for CLASSIFICATION (Phenotype Subtyping)...")
    clf_results = {name: [] for name in clf_models.keys()}

    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls), 1):
        X_tr, y_tr = X[train_idx], y_cls[train_idx]
        X_va, y_va = X[val_idx], y_cls[val_idx]

        for name, model in clf_models.items():
            metrics = evaluate_classification(model, X_tr, y_tr, X_va, y_va)
            clf_results[name].append(metrics)
            logger.info(f"  [Fold {fold}/5] {name:<25} | Acc: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")

    # Aggregate Classification Results
    clf_summary = {}
    for name, folds in clf_results.items():
        clf_summary[name] = {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in folds])),
            "accuracy_std":  float(np.std([m["accuracy"] for m in folds])),
            "bacc_mean":     float(np.mean([m["balanced_accuracy"] for m in folds])),
            "bacc_std":      float(np.std([m["balanced_accuracy"] for m in folds])),
            "macro_f1_mean": float(np.mean([m["macro_f1"] for m in folds])),
            "macro_f1_std":  float(np.std([m["macro_f1"] for m in folds])),
            "weighted_f1_mean": float(np.mean([m["weighted_f1"] for m in folds])),
            "roc_auc_mean":  float(np.mean([m["roc_auc"] for m in folds])),
        }

    # ── Regression Benchmark ──────────────────────────────────────────
    logger.info("\n>>> Running 5-Fold Stratified CV for REGRESSION (EDSS Disability Score)...")
    reg_results = {name: [] for name in reg_models.keys()}

    # Filter for MS patients with valid EDSS
    X_ms = X[ms_mask]
    y_edss_ms = edss_targets[ms_mask]
    y_cls_ms  = y_cls[ms_mask]

    skf_reg = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf_reg.split(X_ms, y_cls_ms), 1):
        X_tr, y_tr = X_ms[train_idx], y_edss_ms[train_idx]
        X_va, y_va = X_ms[val_idx], y_edss_ms[val_idx]

        for name, model in reg_models.items():
            metrics = evaluate_regression(model, X_tr, y_tr, X_va, y_va)
            reg_results[name].append(metrics)
            logger.info(f"  [Fold {fold}/5] {name:<25} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | Pearson R: {metrics['pearson_r']:.4f}")

    # Aggregate Regression Results
    reg_summary = {}
    for name, folds in reg_results.items():
        reg_summary[name] = {
            "mae_mean":  float(np.mean([m["mae"] for m in folds])),
            "mae_std":   float(np.std([m["mae"] for m in folds])),
            "rmse_mean": float(np.mean([m["rmse"] for m in folds])),
            "rmse_std":  float(np.std([m["rmse"] for m in folds])),
            "r2_mean":   float(np.mean([m["r2"] for m in folds])),
            "r2_std":    float(np.std([m["r2"] for m in folds])),
            "pearson_r_mean": float(np.mean([m["pearson_r"] for m in folds])),
        }

    total_time = time.time() - start_time

    # ── Final Benchmark Summary Output ────────────────────────────────
    logger.info("\n" + "=" * 75)
    logger.info("  CLASSIFICATION BENCHMARK SUMMARY (5-Fold Stratified CV)")
    logger.info("=" * 75)
    logger.info(f"{'Algorithm':<25} | {'Accuracy':<14} | {'Macro F1':<14} | {'ROC-AUC':<10}")
    logger.info("-" * 75)
    for name, s in sorted(clf_summary.items(), key=lambda x: x[1]["macro_f1_mean"], reverse=True):
        logger.info(f"{name:<25} | {s['accuracy_mean']*100:5.2f}% +/- {s['accuracy_std']*100:4.2f}% | {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} | {s['roc_auc_mean']:.4f}")

    logger.info("\n" + "=" * 75)
    logger.info("  EDSS REGRESSION BENCHMARK SUMMARY (5-Fold Stratified CV)")
    logger.info("=" * 75)
    logger.info(f"{'Algorithm':<25} | {'MAE (Lower=Best)':<16} | {'RMSE':<14} | {'Pearson R':<10}")
    logger.info("-" * 75)
    for name, s in sorted(reg_summary.items(), key=lambda x: x[1]["mae_mean"]):
        logger.info(f"{name:<25} | {s['mae_mean']:.4f} +/- {s['mae_std']:.4f} | {s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f} | {s['pearson_r_mean']:.4f}")

    # Save to JSON
    benchmark_json = save_dir / "multi_algorithm_benchmark.json"
    with open(benchmark_json, "w") as f:
        json.dump({
            "num_samples": n_samples,
            "num_features": n_features,
            "benchmark_time_seconds": round(total_time, 2),
            "classification": clf_summary,
            "regression": reg_summary,
        }, f, indent=2)

    logger.info(f"\nSaved full experiment metrics to {benchmark_json}")


if __name__ == "__main__":
    main()
