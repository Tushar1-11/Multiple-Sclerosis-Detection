"""
MSNet — Test Data EDSS Inspection & Inference Script
=====================================================
Analyzes `data/test_data.csv`, inspects the EDSS column status,
runs PyTorch 1D-CNN predictions on all 415 test subjects, and saves
`outputs/test_data_predictions.csv`.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from models.tabular_msnet import TabularMSNet


def inspect_and_predict_test_data(alpha: float = 0.70):
    checkpoint_path = Path("outputs/tabular_1dcnn/best_tabular_1dcnn.pt")
    test_data_path = Path("data/test_data.csv")
    output_csv = Path("outputs/test_data_predictions.csv")

    print("=" * 70)
    print("      🧠 MSNet Test Data (`test_data.csv`) EDSS & Model Inference")
    print("=" * 70)

    if not test_data_path.exists():
        print(f"[ERROR] test_data.csv not found at {test_data_path}")
        return

    # 1. Load test_data.csv
    df_test = pd.read_csv(test_data_path, sep="\t")
    df_test.columns = df_test.columns.str.strip()

    print(f"Loaded `test_data.csv` with {len(df_test)} test patient rows and {len(df_test.columns)} columns.")

    # Check if ground truth fp__edss.csv is available
    fp_edss_path = Path("data/fp__edss.csv")
    use_ground_truth = False
    if fp_edss_path.exists():
        print(f"\n[GROUND TRUTH] Found ground truth file: {fp_edss_path}. Loading true EDSS values.")
        df_fp = pd.read_csv(fp_edss_path, sep="\t")
        df_fp.columns = df_fp.columns.str.strip()
        
        # Drop existing EDSS and Unnamed columns from df_test to avoid duplicate conflicts
        cols_to_drop = [c for c in df_test.columns if 'edss' in c.lower() or 'unnamed' in c.lower()]
        df_test.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        
        # Merge true EDSS values based on Patient: Hash
        df_test = pd.merge(df_test, df_fp[['Patient: Hash', 'EDSS']], on='Patient: Hash', how='left')
        use_ground_truth = True
        edss_col = 'EDSS'
        null_edss = df_test['EDSS'].isna().sum()
        print(f"[GROUND TRUTH STATUS] Merged true EDSS: {len(df_test) - null_edss} / {len(df_test)} rows have ground-truth EDSS (104 HCs and 2 MS patients are missing EDSS).")
    else:
        # 2. Check EDSS Column (fallback if no ground truth)
        edss_col = None
        for col in df_test.columns:
            if "edss" in col.lower():
                edss_col = col
                break

        if edss_col is None:
            print("\n[INSPECTION RESULT] 'EDSS' column header is missing in `test_data.csv` (contains an empty tab 'Unnamed: 8').")
            # Rename Unnamed: 8 to EDSS
            for col in df_test.columns:
                if "unnamed" in col.lower():
                    df_test.rename(columns={col: "EDSS"}, inplace=True)
                    edss_col = "EDSS"
                    break
            if edss_col is None:
                df_test["EDSS"] = np.nan
                edss_col = "EDSS"

        null_edss_count = df_test[edss_col].isna().sum()
        print(f"[EDSS STATUS] Found column '{edss_col}': {null_edss_count} / {len(df_test)} rows are missing EDSS values.")

    # 3. Load Trained PyTorch Model & Preprocessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    preprocessor = checkpoint["preprocessor"]

    # Transform test set
    X_test, _ = preprocessor.transform(df_test)
    num_features = X_test.shape[1]

    model = TabularMSNet(in_features=num_features, num_classes=3, embed_dim=256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # 4. Run Inference on Test Dataset (Subtype Classification)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    labels_map = {0: "Healthy Control", 1: "Relapsing-Remitting (RRMS)", 2: "Progressive MS (SPMS/PPMS)"}
    df_test["Predicted_Phenotype_Code"] = preds
    df_test["Predicted_Phenotype_Label"] = [labels_map[p] for p in preds]
    df_test["Prediction_Confidence"] = [round(float(np.max(probs[i]) * 100), 2) for i in range(len(preds))]
    df_test["Prob_Healthy_Control"] = [round(float(probs[i][0] * 100), 2) for i in range(len(preds))]
    df_test["Prob_RRMS"] = [round(float(probs[i][1] * 100), 2) for i in range(len(preds))]
    df_test["Prob_Progressive_MS"] = [round(float(probs[i][2] * 100), 2) for i in range(len(preds))]

    # 4.5 Train EDSS Regression Model on train_data.csv and predict EDSS
    train_data_path = Path("data/train_data.csv")
    if not train_data_path.exists():
        print(f"[WARNING] train_data.csv not found at {train_data_path}. Skipping EDSS prediction.")
    else:
        print("\nTraining EDSS Ensemble Regressor on `train_data.csv`...")
        df_train = pd.read_csv(train_data_path, sep="\t")
        df_train.columns = df_train.columns.str.strip()

        # Target imputation
        df_train.loc[df_train['GROUP-CODE'] == 0, 'EDSS'] = df_train.loc[df_train['GROUP-CODE'] == 0, 'EDSS'].fillna(0.0)
        ms_median = df_train.loc[df_train['GROUP-CODE'] == 1, 'EDSS'].median()
        df_train['EDSS'] = df_train['EDSS'].fillna(ms_median)

        # Impute Disease duration (0 for HCs, median for MS)
        df_train.loc[df_train['GROUP-CODE'] == 0, 'Disease duration'] = df_train.loc[df_train['GROUP-CODE'] == 0, 'Disease duration'].fillna(0.0)
        dd_ms_median = df_train.loc[df_train['GROUP-CODE'] == 1, 'Disease duration'].median()
        df_train['Disease duration'] = df_train['Disease duration'].fillna(dd_ms_median)

        # Impute Treatment-code
        df_train['Treatment-code'] = df_train['Treatment-code'].fillna(-1)

        # Impute test set fields for regression encoding alignment
        df_test.loc[df_test['GROUP-CODE'] == 0, 'Disease duration'] = df_test.loc[df_test['GROUP-CODE'] == 0, 'Disease duration'].fillna(0.0)
        df_test['Disease duration'] = df_test['Disease duration'].fillna(dd_ms_median)
        df_test['Treatment-code'] = df_test['Treatment-code'].fillna(-1)

        # Define regression/classification features
        numerical_features = [
            'Age', 'T2 LV (cubic mm)', 'Disease duration', 
            'ncorticalGREY [ml]', 'nWHITE [ml]', 'nThalamus [ml]', 
            'nHippocampus [ml]', 'nOtherDGM [ml]', 'nBrainstem [ml]', 
            'nGM_Cerebellum [ml]', 'nWM_Cerebellum [ml]'
        ]
        categorical_features = ['SEX-CODE', 'GROUP-CODE', 'PHENOTYPE-CODE', 'INSTITUTION-CODE', 'Treatment-code']

        # Combine train and test to ensure identical categorical encoding
        df_combined = pd.concat([df_train[numerical_features + categorical_features], 
                                 df_test[numerical_features + categorical_features]], keys=['train', 'test'])
        
        # Feature engineering: ratios of key brain volumes
        df_combined['GM_to_WM'] = df_combined['ncorticalGREY [ml]'] / (df_combined['nWHITE [ml]'] + 1e-8)
        df_combined['Thalamus_to_DGM'] = df_combined['nThalamus [ml]'] / (df_combined['nOtherDGM [ml]'] + 1e-8)
        df_combined['Hippocampus_to_DGM'] = df_combined['nHippocampus [ml]'] / (df_combined['nOtherDGM [ml]'] + 1e-8)
        df_combined['Cerebellum_GM_to_WM'] = df_combined['nGM_Cerebellum [ml]'] / (df_combined['nWM_Cerebellum [ml]'] + 1e-8)
        
        engineered_numerical_features = numerical_features + ['GM_to_WM', 'Thalamus_to_DGM', 'Hippocampus_to_DGM', 'Cerebellum_GM_to_WM']

        # One-hot encode categoricals
        df_combined_encoded = pd.get_dummies(df_combined, columns=categorical_features, drop_first=True)
        
        # Scale numericals
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_combined_encoded[engineered_numerical_features] = scaler.fit_transform(df_combined_encoded[engineered_numerical_features])

        X_train = df_combined_encoded.xs('train')
        X_test_reg = df_combined_encoded.xs('test')
        y_train = df_train['EDSS']

        # Train Voting Regressor
        from sklearn.ensemble import (
            RandomForestRegressor, ExtraTreesRegressor, 
            GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
        )
        print("Training EDSS Voting Regressor...")
        reg = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=14, min_samples_split=3, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=150, learning_rate=0.06, max_depth=4, subsample=0.85, random_state=42)),
            ('hgb', HistGradientBoostingRegressor(max_iter=100, learning_rate=0.06, max_depth=4, random_state=42))
        ])
        reg.fit(X_train, y_train)
        reg_preds = reg.predict(X_test_reg)

        # Train Voting Classifier
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        print("Training EDSS Voting Classifier...")
        y_train_class = np.round(y_train * 2) / 2
        clf = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_split=3, random_state=42, n_jobs=-1)),
            ('hgb', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=4, random_state=42))
        ], voting='soft')
        clf.fit(X_train, y_train_class.astype(str))
        
        clf_classes = clf.classes_.astype(float)
        clf_probs = clf.predict_proba(X_test_reg)
        clf_preds = clf_classes[np.argmax(clf_probs, axis=1)]

        # Blend predictions using continuous regressor and discrete classifier
        print(f"Blending predictions with alpha = {alpha:.2f}...")
        predicted_edss = alpha * reg_preds + (1 - alpha) * clf_preds
        
        # Round to nearest 0.5 step, clipped to [0.0, 10.0]
        edss_pred_rounded = np.clip(np.round(predicted_edss * 2) / 2, 0.0, 10.0)
        df_test["Predicted_EDSS"] = edss_pred_rounded
        print("[EDSS MODEL] Blended Voting Regressor and Voting Classifier training and prediction complete.")

    # 5. Print Distribution Summary
    print("\n" + "=" * 70)
    print("           TEST DATA AI MODEL INFERENCE SUMMARY (415 PATIENTS)")
    print("=" * 70)
    pred_counts = df_test["Predicted_Phenotype_Label"].value_counts()
    for label, count in pred_counts.items():
        pct = (count / len(df_test)) * 100
        print(f"  - {label:<30}: {count:>4} patients ({pct:.1f}%)")
    print("=" * 70)

    # 5.5 If ground truth is available, compute and print metrics
    if use_ground_truth:
        # Fill NaN for HCs (GROUP-CODE = 0) with 0.0 for comparison
        df_eval = df_test.copy()
        df_eval.loc[df_eval['GROUP-CODE'] == 0, 'EDSS'] = df_eval.loc[df_eval['GROUP-CODE'] == 0, 'EDSS'].fillna(0.0)
        eval_subset = df_eval.dropna(subset=['EDSS', 'Predicted_EDSS'])
        
        if len(eval_subset) > 0:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from scipy.stats import pearsonr
            
            y_true = eval_subset['EDSS'].astype(float)
            y_pred = eval_subset['Predicted_EDSS'].astype(float)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            corr, _ = pearsonr(y_true, y_pred)
            
            print("\n" + "=" * 70)
            print("                 EDSS REGRESSOR ACCURACY METRICS (VS TRUE EDSS)")
            print("=" * 70)
            print(f"  Mean Absolute Error (MAE):       {mae:.4f}")
            print(f"  Root Mean Squared Error (RMSE):   {rmse:.4f}")
            print(f"  R² Score:                         {r2:.4f}")
            print(f"  Pearson Correlation (r):         {corr:.4f}")
            print("=" * 70)
            
            diff = np.abs(y_true - y_pred)
            print("  Error Tolerances:")
            print(f"    Exact Match (diff = 0.0):      {(diff == 0).sum()} / {len(diff)} ({np.mean(diff == 0)*100:.2f}%)")
            print(f"    Within +/- 0.5 step:           {(diff <= 0.5).sum()} / {len(diff)} ({np.mean(diff <= 0.5)*100:.2f}%)")
            print(f"    Within +/- 1.0 step:           {(diff <= 1.0).sum()} / {len(diff)} ({np.mean(diff <= 1.0)*100:.2f}%)")
            print(f"    Within +/- 2.0 steps:          {(diff <= 2.0).sum()} / {len(diff)} ({np.mean(diff <= 2.0)*100:.2f}%)")
            print(f"    Max absolute error:            {diff.max():.2f}")
            print("=" * 70)

    # 6. Save Predictions to CSV and auxiliary files
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Rename columns to avoid any confusion between synthetic/actual values
    df_output = df_test.copy()
    if 'EDSS' in df_output.columns:
        df_output.rename(columns={'EDSS': 'Actual_EDSS'}, inplace=True)
    
    df_output.to_csv(output_csv, sep="\t", index=False)
    print(f"\n[SUCCESS] Full predictions saved to: {output_csv}")
    
    # Save simple predicted EDSS list
    predicted_list_path = Path("outputs/predicted_edss_list.txt")
    with open(predicted_list_path, "w") as f:
        for val in df_test["Predicted_EDSS"]:
            f.write(f"{val}\n")
    print(f"[SUCCESS] Flat predicted EDSS list saved to: {predicted_list_path}")
    
    # Save predicted EDSS with hash csv
    predicted_hash_path = Path("outputs/predicted_edss_with_hash.csv")
    df_test[['Patient: Hash', 'Predicted_EDSS']].to_csv(predicted_hash_path, index=False)
    print(f"[SUCCESS] Predicted EDSS with hash saved to: {predicted_hash_path}\n")

    # Sample predictions printout with EDSS
    print("Sample Patient Predictions from test_data.csv:")
    print(f"{'Patient Hash':<34} | {'Predicted Subtype':<26} | {'Actual EDSS':<11} | {'Pred EDSS':<9} | {'Confidence':<10}")
    print("-" * 103)
    for i in range(min(8, len(df_test))):
        p_hash = str(df_test.iloc[i]["Patient: Hash"])
        p_label = df_test.iloc[i]["Predicted_Phenotype_Label"]
        p_conf = df_test.iloc[i]["Prediction_Confidence"]
        p_actual_edss = df_test.iloc[i]["EDSS"]
        p_pred_edss = df_test.iloc[i]["Predicted_EDSS"] if "Predicted_EDSS" in df_test.columns else "N/A"
        
        actual_str = f"{p_actual_edss:.1f}" if pd.notna(p_actual_edss) else "NaN"
        pred_str = f"{p_pred_edss:.1f}" if isinstance(p_pred_edss, float) else str(p_pred_edss)
        
        print(f"{p_hash:<34} | {p_label:<26} | {actual_str:<11} | {pred_str:<9} | {p_conf:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MSNet Tabular EDSS prediction pipeline")
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.70, 
        help="Ensemble blending factor. 1.0 = pure Voting Regressor (low MAE), 0.0 = pure Voting Classifier (high exact match)"
    )
    args = parser.parse_args()
    inspect_and_predict_test_data(alpha=args.alpha)
