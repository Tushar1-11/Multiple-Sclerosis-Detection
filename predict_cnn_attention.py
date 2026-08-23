#

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_attention import CNNAttentionHybrid
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.CNNAttentionPredict")

VALID_EDSS = [0.0,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,
              5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]

PHENOTYPE_MAP = {0: "Healthy Control", 1: "RRMS", 2: "Progressive MS"}
GROUP_MAP     = {0: "Healthy Control", 1: "MS Patient"}


def snap_edss(v: float) -> float:
    return min(VALID_EDSS, key=lambda x: abs(x - v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default="outputs/cnn_attention/best_cnn_attention.pt")
    ap.add_argument("--test_data",  type=str, default="data/test_data_clean.csv")
    ap.add_argument("--output",     type=str,
                    default="outputs/cnn_attention_predictions.csv")
    ap.add_argument("--mc_samples", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    setup_logging(log_level="INFO")

    logger.info("=" * 70)
    logger.info("  MSNet - CNN + Attention Hybrid Inference")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # ── Load Checkpoint ────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run: python train_cnn_attention.py first.")
        sys.exit(1)

    logger.info(f"  Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    preprocessor = ckpt["preprocessor"]
    cfg          = ckpt.get("model_config", {})
    in_features  = cfg.get("in_features", preprocessor.num_features)
    token_dim    = cfg.get("token_dim",   256)
    attn_dim     = cfg.get("attn_dim",    256)
    num_layers   = cfg.get("num_layers",  6)
    num_heads    = cfg.get("num_heads",   8)

    logger.info(f"  Epoch: {ckpt.get('epoch','?')} | "
                f"Features: {in_features} | Token dim: {token_dim}")

    # ── Rebuild Model ──────────────────────────────────────────────────
    model = CNNAttentionHybrid(
        in_features=in_features,
        token_dim=token_dim,
        cnn_channels=(128, 256, 256, 256),
        cnn_seq_len=32,
        cnn_dropout=0.2,
        cnn_drop_path_rate=0.10,
        cnn_blocks_per_stage=(2, 3, 3, 2),
        num_attn_layers=num_layers,
        num_heads=num_heads,
        ffn_expansion=4,
        attn_dropout=0.1,
        attn_drop_path_rate=0.10,
        attn_dim=attn_dim,
        head_dropout=0.3,
        num_phenotype_classes=3,
        num_group_classes=2,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total:,}")

    # ── Load Test Data ─────────────────────────────────────────────────
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        sys.exit(1)

    try:
        raw_df = pd.read_csv(test_path, sep="\t")
        if raw_df.shape[1] <= 1:
            raw_df = pd.read_csv(test_path, sep=",")
    except Exception:
        raw_df = pd.read_csv(test_path, sep=",")

    raw_df.columns = raw_df.columns.str.strip()
    hash_col      = "Patient: Hash"
    hashes        = (raw_df[hash_col].values
                     if hash_col in raw_df.columns
                     else np.arange(len(raw_df)))

    logger.info(f"  Test samples: {len(raw_df)}")

    X_test, y_test = preprocessor.transform(raw_df)
    logger.info(f"  Preprocessed shape: {X_test.shape}")

    # ── MC Dropout Inference ───────────────────────────────────────────
    logger.info(f"  Running MC Dropout ({args.mc_samples} passes)...")
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    ph_probs_all  = []   # [S, N, 3]
    gr_probs_all  = []   # [S, N, 2]
    edss_preds_all = []  # [S, N]

    model.train()  # enable dropout for MC sampling
    with torch.no_grad():
        for s in range(args.mc_samples):
            ph_b, gr_b, ed_b = [], [], []
            for i in range(0, len(X_tensor), args.batch_size):
                Xb  = X_tensor[i:i+args.batch_size].to(device)
                out = model(Xb, return_all=True)
                ph_b.append(F.softmax(out["phenotype"], -1).cpu().numpy())
                gr_b.append(F.softmax(out["group"],     -1).cpu().numpy())
                ed_b.append(out["edss"].squeeze(-1).cpu().numpy())
            ph_probs_all.append(np.concatenate(ph_b))
            gr_probs_all.append(np.concatenate(gr_b))
            edss_preds_all.append(np.concatenate(ed_b))
            if (s+1) % 10 == 0:
                logger.info(f"    MC pass {s+1}/{args.mc_samples}")

    ph_probs  = np.stack(ph_probs_all).mean(0)   # [N, 3]
    gr_probs  = np.stack(gr_probs_all).mean(0)   # [N, 2]
    edss_mean = np.stack(edss_preds_all).mean(0) # [N]
    edss_std  = np.stack(edss_preds_all).std(0)  # [N]

    pred_ph   = np.argmax(ph_probs, 1)
    pred_gr   = np.argmax(gr_probs, 1)
    edss_snap = np.array([snap_edss(v) for v in edss_mean])
    entropy   = -(ph_probs * np.log(ph_probs + 1e-8)).sum(1)

    # ── Build Output DataFrame ─────────────────────────────────────────
    out_df = pd.DataFrame({
        "Patient: Hash":             hashes,
        "Predicted_Phenotype":       pred_ph,
        "Predicted_Phenotype_Label": [PHENOTYPE_MAP[p] for p in pred_ph],
        "Predicted_Group":           pred_gr,
        "Predicted_Group_Label":     [GROUP_MAP[p]     for p in pred_gr],
        "Predicted_EDSS":            np.round(edss_mean,  2),
        "Predicted_EDSS_Snapped":    edss_snap,
        "EDSS_Uncertainty_Std":      np.round(edss_std,   4),
        "Phenotype_Prob_HC":         np.round(ph_probs[:,0], 4),
        "Phenotype_Prob_RRMS":       np.round(ph_probs[:,1], 4),
        "Phenotype_Prob_ProgMS":     np.round(ph_probs[:,2], 4),
        "Group_Prob_HC":             np.round(gr_probs[:,0], 4),
        "Group_Prob_MS":             np.round(gr_probs[:,1], 4),
        "Uncertainty_Entropy":       np.round(entropy, 5),
    })

    # Ground truth evaluation against fp__edss.csv if available
    gt_file = Path("data/fp__edss.csv")
    if gt_file.exists():
        try:
            gt_df = pd.read_csv(gt_file, sep="\t")
            if gt_df.shape[1] <= 1:
                gt_df = pd.read_csv(gt_file, sep=",")
            gt_df.columns = gt_df.columns.str.strip()
            
            # Align by Patient Hash
            merged = out_df.merge(gt_df, on="Patient: Hash", how="inner", suffixes=("", "_gt"))
            if len(merged) > 0:mo 
                from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
                acc = accuracy_score(merged["PHENOTYPE-CODE"], merged["Predicted_Phenotype"]) * 100
                bacc = balanced_accuracy_score(merged["PHENOTYPE-CODE"], merged["Predicted_Phenotype"]) * 100
                f1 = f1_score(merged["PHENOTYPE-CODE"], merged["Predicted_Phenotype"], average="macro")
                
                logger.info("\n" + "=" * 70)
                logger.info("  GROUND TRUTH TEST EVALUATION (fp__edss.csv)")
                logger.info("=" * 70)
                logger.info(f"  Phenotype Accuracy          : {acc:.2f}%")
                logger.info(f"  Phenotype Balanced Accuracy : {bacc:.2f}%")
                logger.info(f"  Phenotype Macro F1          : {f1:.4f}")
                
                if "EDSS" in merged.columns:
                    valid_mask = merged["EDSS"].notna()
                    if valid_mask.sum() > 0:
                        gt_edss = merged.loc[valid_mask, "EDSS"].values
                        pred_e  = merged.loc[valid_mask, "Predicted_EDSS"].values
                        mae  = mean_absolute_error(gt_edss, pred_e)
                        rmse = np.sqrt(mean_squared_error(gt_edss, pred_e))
                        corr = np.corrcoef(gt_edss, pred_e)[0, 1]
                        logger.info(f"  EDSS Prediction MAE         : {mae:.4f}")
                        logger.info(f"  EDSS Prediction RMSE        : {rmse:.4f}")
                        logger.info(f"  EDSS Correlation (r)        : {corr:.4f}")
                logger.info("=" * 70)
        except Exception as e:
            logger.warning(f"Could not compute ground truth metrics: {e}")
    elif y_test is not None:
        acc = (pred_ph == y_test).mean() * 100
        logger.info(f"\n  Test Accuracy (ground truth): {acc:.2f}%")

    # ── Save ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("  INFERENCE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Patients predicted   : {len(out_df)}")
    logger.info(f"  Predictions saved    -> {out_path}")

    logger.info("\n  Phenotype Distribution:")
    for lbl, cnt in zip(*np.unique([PHENOTYPE_MAP[p] for p in pred_ph], return_counts=True)):
        logger.info(f"    {lbl:<20}: {cnt} ({cnt/len(out_df)*100:.1f}%)")

    logger.info("\n  Group Distribution:")
    for lbl, cnt in zip(*np.unique([GROUP_MAP[p] for p in pred_gr], return_counts=True)):
        logger.info(f"    {lbl:<20}: {cnt} ({cnt/len(out_df)*100:.1f}%)")

    logger.info(f"\n  EDSS: mean={edss_mean.mean():.3f} | "
                f"std={edss_mean.std():.3f} | "
                f"range=[{edss_mean.min():.2f}, {edss_mean.max():.2f}]")
    logger.info(f"  Mean uncertainty (entropy): {entropy.mean():.4f}")
    logger.info(f"  High uncertainty (>0.5):    {(entropy>0.5).sum()} patients")
    logger.info("\n  Done!")


if __name__ == "__main__":
    main()
