"""
MSNet — Heavy CNN Test Inference Pipeline
==========================================
Loads the trained HeavyTabularMSNet checkpoint and runs inference
on the external `data/test_data.csv` to produce predictions.

Outputs:
  outputs/heavy_cnn_predictions.csv
    - Patient: Hash
    - Predicted_Phenotype (0=HC, 1=RRMS, 2=Progressive MS)
    - Predicted_Phenotype_Label
    - Predicted_Group (0=HC, 1=MS Patient)
    - Predicted_Group_Label
    - Predicted_EDSS
    - Phenotype_Prob_HC / Phenotype_Prob_RRMS / Phenotype_Prob_ProgMS
    - Uncertainty_Entropy

Usage:
  python predict_test_heavy.py
  python predict_test_heavy.py --checkpoint outputs/heavy_cnn/best_heavy_cnn.pt
  python predict_test_heavy.py --mc_samples 50 --target phenotype
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models.heavy_tabular_msnet import HeavyTabularMSNet
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.HeavyPredict")


# ============================================================
# EDSS to Nearest Valid EDSS Step
# ============================================================

VALID_EDSS = [
    0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
    7.5, 8.0, 8.5, 9.0, 9.5, 10.0
]

def snap_to_edss(value: float) -> float:
    """Snap a continuous EDSS prediction to the nearest valid EDSS step."""
    return min(VALID_EDSS, key=lambda v: abs(v - value))


# ============================================================
# Main Inference Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Heavy CNN inference on test_data.csv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/heavy_cnn/best_heavy_cnn.pt",
        help="Path to best model checkpoint"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test_data.csv",
        help="Path to external test CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/heavy_cnn_predictions.csv",
        help="Path to write predictions CSV"
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=30,
        help="Number of MC Dropout forward passes for uncertainty estimation"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="phenotype",
        choices=["phenotype", "group"],
        help="Primary classification target"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Inference batch size"
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    logger.info("=" * 70)
    logger.info("  MSNet — Heavy CNN Test Inference")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # ── Load Checkpoint ───────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run `python train_heavy_cnn.py` first.")
        sys.exit(1)

    logger.info(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore preprocessor (fitted on train data)
    preprocessor = ckpt["preprocessor"]
    model_config = ckpt.get("model_config", {})
    in_features = model_config.get("in_features", preprocessor.num_features)
    embed_dim = model_config.get("embed_dim", 512)

    logger.info(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    logger.info(f"  Input features: {in_features} | Embed dim: {embed_dim}")

    # ── Rebuild Model ────────────────────────────────────────────────
    model = HeavyTabularMSNet(
        in_features=in_features,
        num_phenotype_classes=3,
        num_group_classes=2,
        embed_dim=embed_dim,
        dropout=0.3,
        num_heads=8,
        drop_path_rate=0.15,
        channels=(128, 256, 512, 512),
        seq_len=32,
        blocks_per_stage=(2, 3, 4, 2),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total model parameters: {total_params:,}")

    # ── Load & Preprocess Test Data ───────────────────────────────────
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        sys.exit(1)

    logger.info(f"  Loading test data: {test_data_path}")

    # Load raw test CSV to extract Patient Hash
    try:
        test_df_raw = pd.read_csv(test_data_path, sep="\t")
        if test_df_raw.shape[1] <= 1:
            test_df_raw = pd.read_csv(test_data_path, sep=",")
    except Exception:
        test_df_raw = pd.read_csv(test_data_path, sep=",")

    test_df_raw.columns = test_df_raw.columns.str.strip()
    hash_col = "Patient: Hash"
    patient_hashes = test_df_raw[hash_col].values if hash_col in test_df_raw.columns else np.arange(len(test_df_raw))

    logger.info(f"  Test samples: {len(test_df_raw)}")

    # Transform using the fitted preprocessor (no re-fitting)
    X_test, y_test = preprocessor.transform(test_df_raw)
    logger.info(f"  Preprocessed features shape: {X_test.shape}")

    # ── Run MC Dropout Inference ──────────────────────────────────────
    logger.info(f"  Running MC Dropout inference ({args.mc_samples} passes)...")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    all_phenotype_probs = []   # [mc_samples, N, 3]
    all_group_probs = []       # [mc_samples, N, 2]
    all_edss_preds = []        # [mc_samples, N]

    # Enable dropout during inference for MC sampling
    model.train()

    with torch.no_grad():
        for sample_i in range(args.mc_samples):
            batch_phenotype_probs = []
            batch_group_probs = []
            batch_edss_preds = []

            for i in range(0, len(X_test), args.batch_size):
                X_batch = X_test_tensor[i:i + args.batch_size].to(device)
                outputs = model(X_batch, return_all=True)

                ph_probs = F.softmax(outputs["phenotype"], dim=-1).cpu().numpy()
                gr_probs = F.softmax(outputs["group"], dim=-1).cpu().numpy()
                edss = outputs["edss"].squeeze(-1).cpu().numpy()

                batch_phenotype_probs.append(ph_probs)
                batch_group_probs.append(gr_probs)
                batch_edss_preds.append(edss)

            all_phenotype_probs.append(np.concatenate(batch_phenotype_probs, axis=0))
            all_group_probs.append(np.concatenate(batch_group_probs, axis=0))
            all_edss_preds.append(np.concatenate(batch_edss_preds, axis=0))

            if (sample_i + 1) % 10 == 0:
                logger.info(f"    MC pass {sample_i + 1}/{args.mc_samples} complete")

    # Aggregate MC samples
    all_phenotype_probs = np.stack(all_phenotype_probs, axis=0)  # [S, N, 3]
    all_group_probs = np.stack(all_group_probs, axis=0)          # [S, N, 2]
    all_edss_preds = np.stack(all_edss_preds, axis=0)            # [S, N]

    mean_phenotype_probs = all_phenotype_probs.mean(axis=0)      # [N, 3]
    mean_group_probs = all_group_probs.mean(axis=0)              # [N, 2]
    mean_edss = all_edss_preds.mean(axis=0)                      # [N]
    std_edss = all_edss_preds.std(axis=0)                        # [N]

    # Predicted classes
    pred_phenotype = np.argmax(mean_phenotype_probs, axis=1)     # [N]
    pred_group = np.argmax(mean_group_probs, axis=1)             # [N]

    # Snap EDSS to valid steps
    pred_edss_snapped = np.array([snap_to_edss(v) for v in mean_edss])

    # Uncertainty: predictive entropy
    eps = 1e-8
    entropy = -np.sum(mean_phenotype_probs * np.log(mean_phenotype_probs + eps), axis=1)

    # Human-readable labels
    phenotype_labels_map = {0: "Healthy Control", 1: "RRMS", 2: "Progressive MS"}
    group_labels_map = {0: "Healthy Control", 1: "MS Patient"}

    pred_phenotype_labels = [phenotype_labels_map[p] for p in pred_phenotype]
    pred_group_labels = [group_labels_map[p] for p in pred_group]

    # ── Build Output DataFrame ───────────────────────────────────────
    out_df = pd.DataFrame({
        "Patient: Hash": patient_hashes,
        "Predicted_Phenotype": pred_phenotype,
        "Predicted_Phenotype_Label": pred_phenotype_labels,
        "Predicted_Group": pred_group,
        "Predicted_Group_Label": pred_group_labels,
        "Predicted_EDSS": np.round(mean_edss, 2),
        "Predicted_EDSS_Snapped": pred_edss_snapped,
        "EDSS_Uncertainty_Std": np.round(std_edss, 4),
        "Phenotype_Prob_HC": np.round(mean_phenotype_probs[:, 0], 4),
        "Phenotype_Prob_RRMS": np.round(mean_phenotype_probs[:, 1], 4),
        "Phenotype_Prob_ProgMS": np.round(mean_phenotype_probs[:, 2], 4),
        "Group_Prob_HC": np.round(mean_group_probs[:, 0], 4),
        "Group_Prob_MS": np.round(mean_group_probs[:, 1], 4),
        "Uncertainty_Entropy": np.round(entropy, 5),
    })

    # If ground truth available, compute accuracy
    if y_test is not None:
        if args.target == "phenotype":
            correct = (pred_phenotype == y_test).sum()
        else:
            correct = (pred_group == y_test).sum()
        logger.info(f"\n  Test Accuracy (ground truth available): {correct/len(y_test)*100:.2f}%")

    # ── Save Output ──────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    logger.info(f"\n{'=' * 70}")
    logger.info("  INFERENCE COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Total test patients: {len(out_df)}")
    logger.info(f"  Predictions saved  -> {output_path}")

    # Summary statistics
    logger.info("\n  Phenotype Distribution:")
    for label, count in zip(
        *np.unique(pred_phenotype_labels, return_counts=True)
    ):
        logger.info(f"    {label:<20}: {count} ({count/len(out_df)*100:.1f}%)")

    logger.info("\n  Group Distribution:")
    for label, count in zip(
        *np.unique(pred_group_labels, return_counts=True)
    ):
        logger.info(f"    {label:<20}: {count} ({count/len(out_df)*100:.1f}%)")

    logger.info(f"\n  EDSS Statistics:")
    logger.info(f"    Mean EDSS:   {mean_edss.mean():.3f} ± {mean_edss.std():.3f}")
    logger.info(f"    Min / Max:   {mean_edss.min():.2f} / {mean_edss.max():.2f}")
    logger.info(f"    Median EDSS: {np.median(mean_edss):.3f}")

    logger.info(f"\n  Mean Uncertainty (entropy): {entropy.mean():.4f}")
    logger.info(f"  High uncertainty patients (entropy > 0.5): {(entropy > 0.5).sum()}")
    logger.info("\n  Done!")


if __name__ == "__main__":
    main()
