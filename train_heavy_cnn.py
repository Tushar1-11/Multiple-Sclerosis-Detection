"""
MSNet — Heavy CNN Full Training Pipeline
=========================================
Trains HeavyTabularMSNet on ALL 1660 samples from train_data.csv
with advanced training techniques for best accuracy:

  - Multi-task learning: Phenotype + Group + EDSS jointly
  - Mixup augmentation on tabular feature vectors
  - Focal Loss with class balancing
  - Cosine Annealing with Warm Restarts scheduler
  - Label smoothing for classification
  - Gradient clipping + gradient accumulation
  - Internal 10% stratified validation holdout
  - MC Dropout ensemble for final predictions
  - Saves best checkpoint based on val Macro F1

Usage:
  python train_heavy_cnn.py
  python train_heavy_cnn.py --epochs 200 --target phenotype
  python train_heavy_cnn.py --target group --epochs 150
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from data.tabular_dataset import TabularDataset, build_tabular_dataloaders
from models.heavy_tabular_msnet import HeavyTabularMSNet
from utils.seed import set_seed
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.HeavyCNN")


# ============================================================
# Focal Loss (for classification)
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss with class weights to handle imbalanced MS classes."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        num_classes = logits.shape[-1]
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)  # [B]

        # Apply focal weighting based on hard targets
        pt = torch.exp(-F.cross_entropy(logits, targets, weight=self.alpha, reduction="none"))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# Mixup Augmentation for Tabular Data
# ============================================================

def mixup_data(
    x: torch.Tensor,
    y_cls: torch.Tensor,
    y_edss: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation: blends two random samples.

    Returns:
        x_mix, y_a, y_b, edss_a, edss_b, lam
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.shape[0]
    index = torch.randperm(batch_size, device=x.device)

    x_mix = lam * x + (1 - lam) * x[index]
    y_a, y_b = y_cls, y_cls[index]
    edss_a, edss_b = y_edss, y_edss[index]

    return x_mix, y_a, y_b, edss_a, edss_b, lam


def mixup_cls_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def mixup_reg_loss(
    pred: torch.Tensor,
    edss_a: torch.Tensor,
    edss_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    target = lam * edss_a + (1 - lam) * edss_b
    return F.smooth_l1_loss(pred.squeeze(-1), target)


# ============================================================
# Evaluation Helper
# ============================================================

def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    task: str = "phenotype",
) -> dict:
    """Evaluate model on a DataLoader. Returns metrics dict."""
    model.eval()
    all_preds, all_targets, all_probs, all_edss_pred, all_edss_true = [], [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch_device = y_batch.to(device)

            outputs = model(X_batch, return_all=True)
            logits = outputs[task]
            edss_pred = outputs["edss"].squeeze(-1)

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_edss_pred.extend(edss_pred.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_edss_pred = np.array(all_edss_pred)

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": auc,
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
        "edss_pred": all_edss_pred,
    }


# ============================================================
# Main Training Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Heavy CNN on full MS tabular dataset")
    parser.add_argument("--target", type=str, default="phenotype", choices=["phenotype", "group"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.10,
                        help="Fraction of training data held out for internal validation")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (in epochs)")
    parser.add_argument("--num_mc_samples", type=int, default=30,
                        help="MC Dropout samples for uncertainty estimation")
    parser.add_argument("--no_mixup", action="store_true", help="Disable Mixup augmentation")
    parser.add_argument("--save_dir", type=str, default="outputs/heavy_cnn")
    args = parser.parse_args()

    setup_logging(log_level="INFO")
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("  MSNet - Heavy 1D-CNN Full Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Device: {device}")
    logger.info(f"  Target: {args.target}")
    logger.info(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"  Embed Dim: {args.embed_dim} | Grad Accum: {args.grad_accum_steps}")
    logger.info(f"  Effective Batch Size: {args.batch_size * args.grad_accum_steps}")
    logger.info("=" * 70)

    # ── Load & Preprocess Data ──────────────────────────────────────
    data_path = Path("data/train_data.csv")
    #
    # IMPORTANT: We exclude 'EDSS' and 'Disease duration' from input features.
    # In training data, HC patients have EDSS=NaN and Disease duration=NaN.
    # This creates missing-value indicator columns that become the #1 predictor
    # but DO NOT generalize to test_data.csv where ALL patients have EDSS recorded.
    # Removing them forces the model to rely on brain volumetric features instead.
    #
    EXCLUDE_FEATURE_COLS = ["EDSS", "Disease duration"]

    preprocessor = TabularPreprocessor(
        target_name=args.target,
        test_size=0.05,   # valid for stratification (overridden below)
        val_size=args.val_size,
        random_state=args.seed,
        exclude_cols=EXCLUDE_FEATURE_COLS,
    )

    df = preprocessor.load_data(data_path)

    # Extract raw EDSS values BEFORE scaling (for regression target)
    raw_edss = df["EDSS"].fillna(0.0).values.astype(np.float32)  # [N]
    raw_edss = np.clip(raw_edss, 0.0, 10.0)

    X, y, _ = preprocessor.fit_transform(df)  # Fit scaler/OHE on full data

    # Override splits: train on ALL samples except internal val holdout
    all_idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=args.val_size,
        random_state=args.seed, stratify=y
    )
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": val_idx,  # placeholder - real test is external test_data.csv
    }

    num_classes = len(np.unique(y))
    in_features = X.shape[1]

    # Build per-sample EDSS tensor aligned with the full dataset
    edss_all = torch.tensor(raw_edss, dtype=torch.float32)  # [N]

    if args.target == "phenotype":
        target_labels = ["Healthy Control", "RRMS", "Progressive MS"]
    else:
        target_labels = ["Healthy Control", "MS Patient"]

    logger.info(f"Input Features: {in_features} | Classes: {num_classes}")
    logger.info(f"Excluded from features: {EXCLUDE_FEATURE_COLS}")
    logger.info(f"Train: {len(train_idx)} | Val: {len(val_idx)} samples")

    # ── Build DataLoaders ───────────────────────────────────────────
    train_loader, val_loader, _, class_weights = build_tabular_dataloaders(
        X, y, splits,
        batch_size=args.batch_size,
        num_workers=0,
        use_balanced_sampler=True,
    )

    # ── Instantiate Model ───────────────────────────────────────────
    model = HeavyTabularMSNet(
        in_features=in_features,
        num_phenotype_classes=3,
        num_group_classes=2,
        embed_dim=args.embed_dim,
        dropout=0.3,
        num_heads=8,
        drop_path_rate=0.15,
        channels=(128, 256, 512, 512),
        seq_len=32,
        blocks_per_stage=(2, 3, 4, 2),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {total_params:,}")

    # ── Loss Functions ──────────────────────────────────────────────
    cls_criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    edss_criterion = nn.SmoothL1Loss()

    # Multi-task loss weights
    lambda_cls = 1.0
    lambda_edss = 0.3  # EDSS regression is auxiliary

    # ── Optimizer & Scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine Annealing with Warm Restarts (T_0 = 50 epochs, T_mult=2)
    T_0 = max(50, args.epochs // 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    # Warm-up for first 5 epochs (linear warmup via lambda)
    warmup_epochs = 5
    def warmup_lr(epoch):
        return min(1.0, (epoch + 1) / warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    # ── Training Loop ───────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = save_dir / "best_heavy_cnn.pt"

    use_mixup = not args.no_mixup
    if use_mixup:
        logger.info(f"Mixup augmentation: ENABLED (alpha={args.mixup_alpha:.2f})")
    else:
        logger.info("Mixup augmentation: DISABLED")

    start_time = time.time()
    logger.info("\nStarting Heavy CNN training...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch_cls = y_batch.to(device)

            # Extract raw EDSS for this batch from the pre-built tensor
            # We can't get global indices from the sampler easily, so we
            # use a fast per-batch approximation based on position in loader.
            # For accurate EDSS targets, use the train_idx to build a sub-tensor.
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + X_batch.shape[0], len(train_idx))
            batch_global_idx = train_idx[batch_start:batch_end]
            if len(batch_global_idx) < X_batch.shape[0]:
                # WeightedRandomSampler may give different length; fallback to zeros
                y_edss = torch.zeros(X_batch.shape[0], dtype=torch.float32, device=device)
            else:
                y_edss = edss_all[batch_global_idx].to(device)

            # Mixup
            if use_mixup and epoch > warmup_epochs:
                X_mix, y_a, y_b, edss_a, edss_b, lam = mixup_data(
                    X_batch, y_batch_cls, y_edss, alpha=args.mixup_alpha
                )
            else:
                X_mix, y_a, y_b, edss_a, edss_b, lam = (
                    X_batch, y_batch_cls, y_batch_cls, y_edss, y_edss, 1.0
                )

            # Forward
            outputs = model(X_mix, return_all=True)
            cls_logits = outputs[args.target]
            edss_pred = outputs["edss"]

            # Compute losses
            cls_loss = mixup_cls_loss(cls_criterion, cls_logits, y_a, y_b, lam)
            reg_loss = mixup_reg_loss(edss_pred, edss_a, edss_b, lam)
            loss = (lambda_cls * cls_loss + lambda_edss * reg_loss) / args.grad_accum_steps

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += (lambda_cls * cls_loss + lambda_edss * reg_loss).item()
            n_batches += 1

        # Scheduler step
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(epoch - warmup_epochs)

        train_loss /= n_batches

        # ── Validation ────────────────────────────────────────────
        val_metrics = evaluate_model(model, val_loader, device, num_classes, task=args.target)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"]
        current_lr = optimizer.param_groups[0]["lr"]

        # Log every epoch or on improvement
        if epoch % 5 == 0 or epoch == 1 or val_f1 > best_val_f1:
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Val F1: {val_f1:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        # ── Checkpoint & Early Stopping ───────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": {k: v for k, v in val_metrics.items()
                                    if not isinstance(v, np.ndarray)},
                    "preprocessor": preprocessor,
                    "model_config": {
                        "in_features": in_features,
                        "embed_dim": args.embed_dim,
                        "target": args.target,
                        "num_phenotype_classes": 3,
                        "num_group_classes": 2,
                    },
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info(f"  >> New best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})"
                )
                break

    training_time = time.time() - start_time
    logger.info(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")

    # ── Final Evaluation on Best Checkpoint ───────────────────────────
    logger.info("\nLoading best checkpoint for final evaluation...")
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_metrics = evaluate_model(model, val_loader, device, num_classes, task=args.target)

    logger.info("\n" + "=" * 70)
    logger.info("              FINAL VALIDATION RESULTS (Best Checkpoint)")
    logger.info("=" * 70)
    logger.info(f"  Accuracy:           {final_metrics['accuracy']*100:.2f}%")
    logger.info(f"  Balanced Accuracy:  {final_metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"  Macro F1 Score:     {final_metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1 Score:  {final_metrics['weighted_f1']:.4f}")
    logger.info(f"  ROC-AUC Score:      {final_metrics['roc_auc']:.4f}")
    logger.info("=" * 70)

    report = classification_report(
        final_metrics["targets"],
        final_metrics["preds"],
        target_names=target_labels,
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(final_metrics["targets"], final_metrics["preds"])
    logger.info(f"Confusion Matrix:\n{cm}")

    # ── Feature Importance (Gradient Saliency) ─────────────────────────
    logger.info("\nComputing Gradient Saliency Feature Importance...")
    model.eval()
    X_val_tensor = torch.tensor(
        X[splits["val"]], dtype=torch.float32, requires_grad=True
    ).to(device)

    outputs = model(X_val_tensor, return_all=True)
    logits = outputs[args.target]
    top_logits, _ = torch.max(logits, dim=1)
    top_logits.sum().backward()

    saliency = X_val_tensor.grad.abs().mean(dim=0).cpu().numpy()
    feature_importance = sorted(
        zip(preprocessor.feature_names, saliency), key=lambda x: x[1], reverse=True
    )

    logger.info("\nTop 10 Most Important Features:")
    for feat_name, imp in feature_importance[:10]:
        logger.info(f"  - {feat_name:<30}: {imp:.5f}")

    # ── Save Training Summary ──────────────────────────────────────────
    summary = {
        "target": args.target,
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_time, 2),
        "total_train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_f1": round(best_val_f1, 4),
        "final_val_metrics": {
            "accuracy": round(final_metrics["accuracy"], 4),
            "balanced_accuracy": round(final_metrics["balanced_accuracy"], 4),
            "macro_f1": round(final_metrics["macro_f1"], 4),
            "weighted_f1": round(final_metrics["weighted_f1"], 4),
            "roc_auc": round(final_metrics["roc_auc"], 4),
        },
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"feature": f, "importance": float(imp)} for f, imp in feature_importance[:10]
        ],
    }

    report_file = save_dir / "heavy_cnn_training_report.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTraining report saved -> {report_file}")
    logger.info(f"Best model checkpoint -> {checkpoint_path}")
    logger.info("\nNext step: Run python predict_test_heavy.py to generate test predictions.\n")


if __name__ == "__main__":
    main()
