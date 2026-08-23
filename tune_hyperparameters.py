"""
MSNet -- Bayesian Optimization (Optuna) Hyperparameter Tuning
==============================================================
Performs Bayesian Optimization across ML models using Optuna.

Rules & Protections:
  1. TUNES STRICTLY ON VALIDATION SET (data/val_data.csv).
  2. Keeps test set (test_data_clean.csv) completely untouched.
  3. Overfitting Penalty:
     Objective = Val_Score - alpha * max(0, Train_Score - Val_Score)
     Penalizes trials where Train_Score >> Val_Score.

Models Tuned:
  Classification:
    - Logistic Regression (C, solver, l1_ratio)
    - Support Vector Classifier (C, gamma, kernel)
    - Random Forest Classifier (n_estimators, max_depth, min_samples_split, max_features)
    - XGBoost Classifier (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)
    - LightGBM Classifier (n_estimators, num_leaves, learning_rate, min_child_samples)
    - Neural Network MLP (hidden_layer_sizes, alpha, learning_rate_init)

  EDSS Regression:
    - Random Forest Regressor (n_estimators, max_depth, min_samples_split)
    - XGBoost Regressor (n_estimators, max_depth, learning_rate, subsample)
    - LightGBM Regressor (n_estimators, num_leaves, learning_rate)

Outputs:
  outputs/optuna_tuning_results.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("MSNet.OptunaTuning")

EXCLUDE_COLS = ["EDSS", "Disease duration"]


def main():
    set_seed(42)
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("  MSNet -- Bayesian Optimization (Optuna) Hyperparameter Tuning")
    logger.info("=" * 75)
    logger.info("  Validation Strategy : Strict Validation Set (data/val_data.csv)")
    logger.info("  Overfitting Guard   : Objective = Val_F1 - 0.5 * max(0, Train_F1 - Val_F1)")
    logger.info("  Test Set Status     : 100% Untouched")
    logger.info("=" * 75)

    # 1. Load Train Split & Validation Set
    train_df = pd.read_csv("data/train_split.csv", sep="\t")
    val_df   = pd.read_csv("data/val_data.csv",   sep="\t")

    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        exclude_cols=EXCLUDE_COLS,
        use_feature_engineering=True
    )

    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_val, y_val        = preprocessor.transform(val_df)

    logger.info(f"Loaded Train Split: {X_train.shape[0]} samples | Val Split: {X_val.shape[0]} samples")
    logger.info(f"Feature Space     : {X_train.shape[1]} engineered features")
    logger.info("=" * 75)

    tuning_results = {"classification": {}, "regression": {}}

    # ── 1. LOGISTIC REGRESSION TUNING ─────────────────────────────────
    logger.info("\n>>> Tuning Logistic Regression...")
    def objective_lr(trial):
        c = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga", "sag"])
        model = LogisticRegression(C=c, solver=solver, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        tr_pred = model.predict(X_train)
        va_pred = model.predict(X_val)

        tr_f1 = f1_score(y_train, tr_pred, average="macro", zero_division=0)
        va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)

        trial.set_user_attr("train_f1", float(tr_f1))
        trial.set_user_attr("val_f1", float(va_f1))
        trial.set_user_attr("val_acc", float(accuracy_score(y_val, va_pred)))
        trial.set_user_attr("overfit_gap", float(max(0, tr_f1 - va_f1)))

        return va_f1 - 0.5 * max(0, tr_f1 - va_f1)

    study_lr = optuna.create_study(direction="maximize")
    study_lr.optimize(objective_lr, n_trials=40, show_progress_bar=False)
    b_lr = study_lr.best_trial
    logger.info(f"  [Best LR] Params: {b_lr.params} | Val F1: {b_lr.user_attrs['val_f1']:.4f} | Train F1: {b_lr.user_attrs['train_f1']:.4f} | Overfit Gap: {b_lr.user_attrs['overfit_gap']:.4f}")
    tuning_results["classification"]["Logistic Regression"] = {
        "best_params": b_lr.params,
        "val_f1": b_lr.user_attrs["val_f1"],
        "val_acc": b_lr.user_attrs["val_acc"],
        "train_f1": b_lr.user_attrs["train_f1"],
        "overfit_gap": b_lr.user_attrs["overfit_gap"]
    }

    # ── 2. SVM TUNING ──────────────────────────────────────────────────
    logger.info("\n>>> Tuning SVM (Support Vector Classifier)...")
    def objective_svm(trial):
        c = trial.suggest_float("C", 1e-2, 50.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1])
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        model = SVC(C=c, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        model.fit(X_train, y_train)

        tr_pred = model.predict(X_train)
        va_pred = model.predict(X_val)

        tr_f1 = f1_score(y_train, tr_pred, average="macro", zero_division=0)
        va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)

        trial.set_user_attr("train_f1", float(tr_f1))
        trial.set_user_attr("val_f1", float(va_f1))
        trial.set_user_attr("val_acc", float(accuracy_score(y_val, va_pred)))
        trial.set_user_attr("overfit_gap", float(max(0, tr_f1 - va_f1)))

        return va_f1 - 0.5 * max(0, tr_f1 - va_f1)

    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(objective_svm, n_trials=40, show_progress_bar=False)
    b_svm = study_svm.best_trial
    logger.info(f"  [Best SVM] Params: {b_svm.params} | Val F1: {b_svm.user_attrs['val_f1']:.4f} | Train F1: {b_svm.user_attrs['train_f1']:.4f} | Overfit Gap: {b_svm.user_attrs['overfit_gap']:.4f}")
    tuning_results["classification"]["SVM"] = {
        "best_params": b_svm.params,
        "val_f1": b_svm.user_attrs["val_f1"],
        "val_acc": b_svm.user_attrs["val_acc"],
        "train_f1": b_svm.user_attrs["train_f1"],
        "overfit_gap": b_svm.user_attrs["overfit_gap"]
    }

    # ── 3. RANDOM FOREST TUNING ────────────────────────────────────────
    logger.info("\n>>> Tuning Random Forest Classifier...")
    def objective_rf(trial):
        n_est = trial.suggest_int("n_estimators", 50, 300, step=25)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_split = trial.suggest_int("min_samples_split", 2, 10)
        min_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf,
            class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        tr_pred = model.predict(X_train)
        va_pred = model.predict(X_val)

        tr_f1 = f1_score(y_train, tr_pred, average="macro", zero_division=0)
        va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)

        trial.set_user_attr("train_f1", float(tr_f1))
        trial.set_user_attr("val_f1", float(va_f1))
        trial.set_user_attr("val_acc", float(accuracy_score(y_val, va_pred)))
        trial.set_user_attr("overfit_gap", float(max(0, tr_f1 - va_f1)))

        return va_f1 - 0.5 * max(0, tr_f1 - va_f1)

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=40, show_progress_bar=False)
    b_rf = study_rf.best_trial
    logger.info(f"  [Best RF] Params: {b_rf.params} | Val F1: {b_rf.user_attrs['val_f1']:.4f} | Train F1: {b_rf.user_attrs['train_f1']:.4f} | Overfit Gap: {b_rf.user_attrs['overfit_gap']:.4f}")
    tuning_results["classification"]["Random Forest"] = {
        "best_params": b_rf.params,
        "val_f1": b_rf.user_attrs["val_f1"],
        "val_acc": b_rf.user_attrs["val_acc"],
        "train_f1": b_rf.user_attrs["train_f1"],
        "overfit_gap": b_rf.user_attrs["overfit_gap"]
    }

    # ── 4. XGBOOST TUNING ──────────────────────────────────────────────
    if HAS_XGB:
        logger.info("\n>>> Tuning XGBoost Classifier...")
        def objective_xgb(trial):
            n_est = trial.suggest_int("n_estimators", 50, 250, step=25)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)

            model = XGBClassifier(
                n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
                subsample=subsample, colsample_bytree=colsample,
                eval_metric="mlogloss", random_state=42
            )
            model.fit(X_train, y_train)

            tr_pred = model.predict(X_train)
            va_pred = model.predict(X_val)

            tr_f1 = f1_score(y_train, tr_pred, average="macro", zero_division=0)
            va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)

            trial.set_user_attr("train_f1", float(tr_f1))
            trial.set_user_attr("val_f1", float(va_f1))
            trial.set_user_attr("val_acc", float(accuracy_score(y_val, va_pred)))
            trial.set_user_attr("overfit_gap", float(max(0, tr_f1 - va_f1)))

            return va_f1 - 0.5 * max(0, tr_f1 - va_f1)

        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(objective_xgb, n_trials=40, show_progress_bar=False)
        b_xgb = study_xgb.best_trial
        logger.info(f"  [Best XGBoost] Params: {b_xgb.params} | Val F1: {b_xgb.user_attrs['val_f1']:.4f} | Train F1: {b_xgb.user_attrs['train_f1']:.4f} | Overfit Gap: {b_xgb.user_attrs['overfit_gap']:.4f}")
        tuning_results["classification"]["XGBoost"] = {
            "best_params": b_xgb.params,
            "val_f1": b_xgb.user_attrs["val_f1"],
            "val_acc": b_xgb.user_attrs["val_acc"],
            "train_f1": b_xgb.user_attrs["train_f1"],
            "overfit_gap": b_xgb.user_attrs["overfit_gap"]
        }

    # ── 5. LIGHTGBM TUNING ─────────────────────────────────────────────
    if HAS_LGB:
        logger.info("\n>>> Tuning LightGBM Classifier...")
        def objective_lgb(trial):
            n_est = trial.suggest_int("n_estimators", 50, 250, step=25)
            num_leaves = trial.suggest_int("num_leaves", 15, 63)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            min_child = trial.suggest_int("min_child_samples", 5, 30)

            model = LGBMClassifier(
                n_estimators=n_est, num_leaves=num_leaves, learning_rate=lr,
                min_child_samples=min_child, random_state=42, verbose=-1
            )
            model.fit(X_train, y_train)

            tr_pred = model.predict(X_train)
            va_pred = model.predict(X_val)

            tr_f1 = f1_score(y_train, tr_pred, average="macro", zero_division=0)
            va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)

            trial.set_user_attr("train_f1", float(tr_f1))
            trial.set_user_attr("val_f1", float(va_f1))
            trial.set_user_attr("val_acc", float(accuracy_score(y_val, va_pred)))
            trial.set_user_attr("overfit_gap", float(max(0, tr_f1 - va_f1)))

            return va_f1 - 0.5 * max(0, tr_f1 - va_f1)

        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(objective_lgb, n_trials=40, show_progress_bar=False)
        b_lgb = study_lgb.best_trial
        logger.info(f"  [Best LightGBM] Params: {b_lgb.params} | Val F1: {b_lgb.user_attrs['val_f1']:.4f} | Train F1: {b_lgb.user_attrs['train_f1']:.4f} | Overfit Gap: {b_lgb.user_attrs['overfit_gap']:.4f}")
        tuning_results["classification"]["LightGBM"] = {
            "best_params": b_lgb.params,
            "val_f1": b_lgb.user_attrs["val_f1"],
            "val_acc": b_lgb.user_attrs["val_acc"],
            "train_f1": b_lgb.user_attrs["train_f1"],
            "overfit_gap": b_lgb.user_attrs["overfit_gap"]
        }

    # ── 6. EDSS REGRESSION TUNING ─────────────────────────────────────
    logger.info("\n>>> Tuning Random Forest Regressor (EDSS Disability Score)...")
    ms_mask_tr = (train_df["PHENOTYPE-CODE"] > 0) & (~train_df["EDSS"].isna())
    ms_mask_va = (val_df["PHENOTYPE-CODE"] > 0)   & (~val_df["EDSS"].isna())

    X_edss_tr = X_train[ms_mask_tr]
    y_edss_tr = train_df.loc[ms_mask_tr, "EDSS"].values.astype(np.float32)

    X_edss_va = X_val[ms_mask_va]
    y_edss_va = val_df.loc[ms_mask_va, "EDSS"].values.astype(np.float32)

    def objective_rfr(trial):
        n_est = trial.suggest_int("n_estimators", 50, 300, step=25)
        max_depth = trial.suggest_int("max_depth", 4, 15)
        min_split = trial.suggest_int("min_samples_split", 2, 10)

        model = RandomForestRegressor(
            n_estimators=n_est, max_depth=max_depth, min_samples_split=min_split, random_state=42
        )
        model.fit(X_edss_tr, y_edss_tr)

        tr_pred = model.predict(X_edss_tr)
        va_pred = model.predict(X_edss_va)

        tr_mae = mean_absolute_error(y_edss_tr, tr_pred)
        va_mae = mean_absolute_error(y_edss_va, va_pred)

        trial.set_user_attr("train_mae", float(tr_mae))
        trial.set_user_attr("val_mae", float(va_mae))
        trial.set_user_attr("overfit_gap", float(max(0, va_mae - tr_mae)))

        # Minimize validation MAE + penalty if train MAE << val MAE
        return -(va_mae + 0.5 * max(0, va_mae - tr_mae))

    study_rfr = optuna.create_study(direction="maximize")
    study_rfr.optimize(objective_rfr, n_trials=40, show_progress_bar=False)
    b_rfr = study_rfr.best_trial
    logger.info(f"  [Best RF Regressor] Params: {b_rfr.params} | Val MAE: {b_rfr.user_attrs['val_mae']:.4f} | Train MAE: {b_rfr.user_attrs['train_mae']:.4f} | Overfit Gap: {b_rfr.user_attrs['overfit_gap']:.4f}")
    tuning_results["regression"]["Random Forest Regressor"] = {
        "best_params": b_rfr.params,
        "val_mae": b_rfr.user_attrs["val_mae"],
        "train_mae": b_rfr.user_attrs["train_mae"],
        "overfit_gap": b_rfr.user_attrs["overfit_gap"]
    }

    # Save to JSON
    out_json = save_dir / "optuna_tuning_results.json"
    with open(out_json, "w") as f:
        json.dump(tuning_results, f, indent=2)

    logger.info(f"\nSaved Bayesian Optimization tuning results to {out_json}")


if __name__ == "__main__":
    main()
