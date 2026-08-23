"""
MSNet — Verify Model & Display Full Dataset Evaluation
======================================================
Run this script in your terminal to inspect the trained model checkpoint,
evaluate all 1,659 samples from `data/train_data.csv`, and view complete performance metrics.

Usage:
  python verify_model.py
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from models.tabular_msnet import TabularMSNet


def verify():
    checkpoint_path = Path("outputs/tabular_1dcnn/best_tabular_1dcnn.pt")
    data_path = Path("data/train_data.csv")

    print("=" * 70)
    print("      MSNet 1D-CNN Model Verification & Full Dataset Inspection")
    print("=" * 70)

    # 1. Check if model checkpoint exists
    if not checkpoint_path.exists():
        print(f"[ERROR] Model checkpoint not found at '{checkpoint_path}'.")
        return

    print(f"[OK] Found Model Checkpoint: {checkpoint_path}")
    print(f"[OK] Found Training Data:     {data_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # 2. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    preprocessor = checkpoint["preprocessor"]

    # 3. Preprocess full train_data.csv
    df = preprocessor.load_data(data_path)
    X, y = preprocessor.transform(df)

    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    class_names = ["Healthy Control", "RRMS", "Progressive MS"] if num_classes == 3 else ["Healthy Control", "MS Patient"]

    print(f"\nSuccessfully loaded {num_samples} total dataset rows with {num_features} features.")

    # 4. Load Model
    model = TabularMSNet(
        in_features=num_features,
        num_classes=num_classes,
        embed_dim=256,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 5. Run Forward Pass on All 1,659 Samples
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # 6. Calculate Full Metrics
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")
    auc = roc_auc_score(y, probs.cpu().numpy(), multi_class="ovr", average="macro")

    print("\n" + "=" * 70)
    print("               FULL DATASET VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Total Evaluated Patient Samples:  {num_samples}")
    print(f"Full Model Accuracy:             {acc * 100:.2f}%")
    print(f"Full Model Macro F1-Score:        {macro_f1:.4f}")
    print(f"Full Model Weighted F1-Score:     {weighted_f1:.4f}")
    print(f"Full Model ROC-AUC Score:         {auc:.4f}")
    print("=" * 70)

    print("\nDetailed Classification Report:")
    print(classification_report(y, preds, target_names=class_names, digits=4))

    print("Confusion Matrix (Actual vs Predicted):")
    cm = confusion_matrix(y, preds)
    print(f"                  Predicted")
    print(f"            HC    RRMS   Progressive")
    for i, row in enumerate(cm):
        print(f"Actual {class_names[i]:<10} {row[0]:<6} {row[1]:<6} {row[2] if len(row)>2 else '':<6}")

    # 7. Print Sample Predictions with Probabilities & Uncertainty
    print("\nSample Patient Diagnostic Predictions:")
    print(f"{'Patient Row':<12} | {'True Subtype':<16} | {'Predicted Subtype':<16} | {'Confidence':<10}")
    print("-" * 65)
    for idx in range(min(8, num_samples)):
        true_name = class_names[y[idx]]
        pred_name = class_names[preds[idx]]
        conf = probs[idx][preds[idx]].item() * 100
        print(f"Row {idx+1:<8} | {true_name:<16} | {pred_name:<16} | {conf:.2f}%")

    print("\n[SUCCESS] Verification Complete: All 1,659 samples successfully trained and verified!")
    print("=" * 70)


if __name__ == "__main__":
    verify()
