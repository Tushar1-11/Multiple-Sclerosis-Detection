"""
MSNet -- CNN + MLP Hybrid Training Pipeline
============================================
Trains CNNMLPHybrid on ALL 1660 train samples with a 2-phase strategy:

  Phase 1 (Warm-up): Freeze CNN, train only the MLP tower fast (20 epochs)
  Phase 2 (Fine-tune): Unfreeze CNN, train full model end-to-end (180 epochs)

Advanced techniques:
  - 2-phase training (frozen CNN warm-up -> full end-to-end)
  - Mixup augmentation on tabular features
  - Label Smoothing + Focal Loss
  - Cosine Annealing with Warm Restarts
  - Gradient accumulation (effective batch 128)
  - MC Dropout ensemble for final predictions
  - Feature importance via gradient saliency

Usage:
  python train_cnn_mlp.py
  python train_cnn_mlp.py --target phenotype --epochs 200
  python train_cnn_mlp.py --target group --warmup_epochs 15
  python train_cnn_mlp.py --skip_warmup   # train end-to-end from start
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from data.tabular_dataset import build_tabular_dataloaders
from models.cnn_mlp_hybrid import CNNMLPHybrid
from utils.seed import set_seed
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.CNNMLPHybrid")

# Features excluded to prevent biased missing-value indicators
# (HC patients have EDSS/Disease duration missing in train but not in test)
EXCLUDE_FEATURE_COLS = ["EDSS", "Disease duration"]


# ============================================================
# Focal Loss with Label Smoothing
# ============================================================

class FocalLoss(nn.Module):
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
        num_classes = logits.shape[-1]
        smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth * log_probs).sum(dim=-1)

        pt = torch.exp(-F.cross_entropy(logits, targets, weight=self.alpha, reduction="none"))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# Mixup Augmentation
# ============================================================

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    edss: torch.Tensor,
    alpha: float = 0.4,
):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.shape[0], device=x.device)
    return (
        lam * x + (1 - lam) * x[idx],
        y, y[idx],
        edss, edss[idx],
        lam,
    )


def cls_loss_mixup(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def reg_loss_mixup(pred, ea, eb, lam):
    target = lam * ea + (1 - lam) * eb
    return F.smooth_l1_loss(pred.squeeze(-1), target)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device, num_classes, task="phenotype"):
    model.eval()
    preds_all, targets_all, probs_all = [], [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            out = model(X_b, return_all=True)
            logits = out[task]
            probs = F.softmax(logits, dim=-1)
            preds_all.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            targets_all.extend(y_b.numpy())
            probs_all.extend(probs.cpu().numpy())

    p = np.array(preds_all)
    t = np.array(targets_all)
    pr = np.array(probs_all)

    try:
        auc = (roc_auc_score(t, pr[:, 1]) if num_classes == 2
               else roc_auc_score(t, pr, multi_class="ovr", average="macro"))
    except Exception:
        auc = 0.0

    return {
        "accuracy":          accuracy_score(t, p),
        "balanced_accuracy": balanced_accuracy_score(t, p),
        "macro_f1":          f1_score(t, p, average="macro",    zero_division=0),
        "weighted_f1":       f1_score(t, p, average="weighted", zero_division=0),
        "roc_auc":           auc,
        "targets": t, "preds": p, "probs": pr,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train CNN+MLP Hybrid on MS tabular data")
    parser.add_argument("--target",         type=str,   default="phenotype",
                        choices=["phenotype", "group"])
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--warmup_epochs",  type=int,   default=20,
                        help="Phase-1 epochs with CNN frozen (MLP warm-up)")
    parser.add_argument("--skip_warmup",    action="store_true",
                        help="Skip Phase 1, train end-to-end from the start")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--mlp_lr",         type=float, default=1e-3,
                        help="LR for MLP-only warmup phase")
    parser.add_argument("--embed_dim",      type=int,   default=512)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--val_size",       type=float, default=0.10)
    parser.add_argument("--mixup_alpha",    type=float, default=0.4)
    parser.add_argument("--focal_gamma",    type=float, default=2.0)
    parser.add_argument("--label_smooth",   type=float, default=0.1)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--grad_accum",     type=int,   default=2)
    parser.add_argument("--patience",       type=int,   default=30)
    parser.add_argument("--no_mixup",       action="store_true")
    parser.add_argument("--save_dir",       type=str,   default="outputs/cnn_mlp")
    args = parser.parse_args()

    setup_logging(log_level="INFO")
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 72)
    logger.info("  MSNet - CNN + MLP Hybrid Training Pipeline")
    logger.info("=" * 72)
    logger.info(f"  Device        : {device}")
    logger.info(f"  Target        : {args.target}")
    logger.info(f"  Total Epochs  : {args.epochs}")
    logger.info(f"  Warmup Epochs : {args.warmup_epochs} (CNN frozen)")
    logger.info(f"  Batch Size    : {args.batch_size} x {args.grad_accum} = {args.batch_size*args.grad_accum}")
    logger.info(f"  LR (full)     : {args.lr} | LR (warmup) : {args.mlp_lr}")
    logger.info("=" * 72)

    # ── Data ────────────────────────────────────────────────────────
    data_path = Path("data/train_data.csv")
    preprocessor = TabularPreprocessor(
        target_name=args.target,
        test_size=0.05,
        val_size=args.val_size,
        random_state=args.seed,
        exclude_cols=EXCLUDE_FEATURE_COLS,
    )

    df = preprocessor.load_data(data_path)

    # Raw EDSS for regression target (before scaling)
    raw_edss = np.clip(df["EDSS"].fillna(0.0).values.astype(np.float32), 0.0, 10.0)

    X, y, _ = preprocessor.fit_transform(df)
    in_features = X.shape[1]
    num_classes = len(np.unique(y))

    # Full-data split: all samples except val holdout for training
    all_idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=args.val_size,
        random_state=args.seed, stratify=y
    )
    splits = {"train": train_idx, "val": val_idx, "test": val_idx}

    edss_all = torch.tensor(raw_edss, dtype=torch.float32)

    target_labels = (
        ["Healthy Control", "RRMS", "Progressive MS"]
        if args.target == "phenotype" else
        ["Healthy Control", "MS Patient"]
    )

    logger.info(f"  Features: {in_features} | Classes: {num_classes}")
    logger.info(f"  Excluded: {EXCLUDE_FEATURE_COLS}")
    logger.info(f"  Train samples: {len(train_idx)} | Val: {len(val_idx)}")

    train_loader, val_loader, _, class_weights = build_tabular_dataloaders(
        X, y, splits,
        batch_size=args.batch_size,
        num_workers=0,
        use_balanced_sampler=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = CNNMLPHybrid(
        in_features=in_features,
        # CNN backbone
        cnn_embed_dim=args.embed_dim,
        cnn_channels=(128, 256, 512, 512),
        cnn_seq_len=32,
        cnn_num_heads=8,
        cnn_drop_path_rate=0.15,
        cnn_blocks_per_stage=(2, 3, 4, 2),
        # MLP tower
        mlp_layer_dims=(512, 512, 256, 256, 128),
        mlp_dropout=0.35,
        mlp_expansion=2.0,
        # Global
        shared_dropout=0.3,
        num_phenotype_classes=3,
        num_group_classes=2,
        freeze_cnn=(not args.skip_warmup),   # frozen in Phase 1
    ).to(device)

    # ── Loss ─────────────────────────────────────────────────────────
    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=args.focal_gamma,
        label_smoothing=args.label_smooth,
    )
    edss_criterion = nn.SmoothL1Loss()
    lambda_edss = 0.25

    # ── Training State ───────────────────────────────────────────────
    best_val_f1   = 0.0
    best_epoch    = 0
    patience_ctr  = 0
    checkpoint    = save_dir / "best_cnn_mlp.pt"
    use_mixup     = not args.no_mixup
    phase         = 1  # 1 = warmup (CNN frozen), 2 = full end-to-end

    def make_optimizer(lr):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
        )

    # Phase 1 optimizer: only MLP params
    optimizer = make_optimizer(args.mlp_lr if not args.skip_warmup else args.lr)

    T_0 = max(40, args.epochs // 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
    )

    start = time.time()

    # ── Training Loop ─────────────────────────────────────────────
    logger.info("\nStarting CNN + MLP Hybrid training...\n")

    for epoch in range(1, args.epochs + 1):

        # ── Phase Transition: unfreeze CNN after warmup ───────────
        if phase == 1 and (args.skip_warmup or epoch > args.warmup_epochs):
            phase = 2
            model.unfreeze_cnn()
            logger.info(f"\n[Phase 2 Start - Epoch {epoch}] CNN unfrozen. Full end-to-end training.\n")
            # Rebuild optimizer with lower LR for CNN, higher for MLP
            optimizer = torch.optim.AdamW([
                {"params": model.cnn_encoder.parameters(), "lr": args.lr * 0.1},
                {"params": model.phenotype_mlp.parameters(), "lr": args.lr},
                {"params": model.group_mlp.parameters(),    "lr": args.lr},
                {"params": model.edss_mlp.parameters(),     "lr": args.lr},
            ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=2, eta_min=1e-7
            )

        model.train()
        train_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        for batch_idx, (X_b, y_b) in enumerate(train_loader):
            X_b       = X_b.to(device)
            y_b_cls   = y_b.to(device)

            # EDSS target from raw values
            b_start = batch_idx * args.batch_size
            b_end   = min(b_start + X_b.shape[0], len(train_idx))
            b_idx   = train_idx[b_start:b_end]
            if len(b_idx) < X_b.shape[0]:
                y_edss = torch.zeros(X_b.shape[0], device=device)
            else:
                y_edss = edss_all[b_idx].to(device)

            # Mixup
            if use_mixup and epoch > args.warmup_epochs:
                X_mix, ya, yb, ea, eb, lam = mixup_batch(X_b, y_b_cls, y_edss, args.mixup_alpha)
            else:
                X_mix, ya, yb, ea, eb, lam = X_b, y_b_cls, y_b_cls, y_edss, y_edss, 1.0

            # Forward
            out       = model(X_mix, return_all=True)
            cls_logit = out[args.target]
            edss_pred = out["edss"]

            # Loss
            cls_loss = cls_loss_mixup(criterion, cls_logit, ya, yb, lam)
            reg_loss = reg_loss_mixup(edss_pred, ea, eb, lam)
            loss     = (cls_loss + lambda_edss * reg_loss) / args.grad_accum

            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += (cls_loss + lambda_edss * reg_loss).item()
            n_batches  += 1

        scheduler.step(epoch)
        train_loss /= n_batches

        # ── Validation ────────────────────────────────────────────
        vm      = evaluate(model, val_loader, device, num_classes, task=args.target)
        val_f1  = vm["macro_f1"]
        val_acc = vm["accuracy"]
        val_auc = vm["roc_auc"]
        lr_now  = optimizer.param_groups[0]["lr"]
        ph_tag  = f"[P{phase}]"

        if epoch % 5 == 0 or epoch == 1 or val_f1 > best_val_f1:
            logger.info(
                f"{ph_tag} Epoch {epoch:03d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Val F1: {val_f1:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"LR: {lr_now:.2e}"
            )

        # ── Checkpoint & Early Stopping ───────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_ctr = 0
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "val_metrics":       {k: v for k, v in vm.items()
                                      if not isinstance(v, np.ndarray)},
                "preprocessor":      preprocessor,
                "model_config": {
                    "in_features":           in_features,
                    "cnn_embed_dim":         args.embed_dim,
                    "target":                args.target,
                    "num_phenotype_classes": 3,
                    "num_group_classes":     2,
                    "mlp_layer_dims":        [512, 512, 256, 256, 128],
                    "exclude_cols":          EXCLUDE_FEATURE_COLS,
                },
                "args": vars(args),
            }, checkpoint)
            logger.info(f"  >> Best model saved (Val F1: {best_val_f1:.4f} @ Epoch {epoch})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})"
                )
                break

    elapsed = time.time() - start
    logger.info(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Final Evaluation ─────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    fm = evaluate(model, val_loader, device, num_classes, task=args.target)
    logger.info("\n" + "=" * 72)
    logger.info("  FINAL VALIDATION RESULTS  (CNN + MLP Hybrid, Best Checkpoint)")
    logger.info("=" * 72)
    logger.info(f"  Accuracy          : {fm['accuracy']*100:.2f}%")
    logger.info(f"  Balanced Accuracy : {fm['balanced_accuracy']*100:.2f}%")
    logger.info(f"  Macro F1          : {fm['macro_f1']:.4f}")
    logger.info(f"  Weighted F1       : {fm['weighted_f1']:.4f}")
    logger.info(f"  ROC-AUC           : {fm['roc_auc']:.4f}")
    logger.info("=" * 72)

    report = classification_report(fm["targets"], fm["preds"],
                                   target_names=target_labels, digits=4)
    logger.info(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(fm["targets"], fm["preds"])
    logger.info(f"Confusion Matrix:\n{cm}")

    # ── Feature Importance (Gradient Saliency) ────────────────────────
    logger.info("\nComputing gradient saliency feature importance...")
    model.eval()
    X_val_t = torch.tensor(X[splits["val"]], dtype=torch.float32,
                            requires_grad=True).to(device)
    out = model(X_val_t, return_all=True)
    top, _ = torch.max(out[args.target], dim=1)
    top.sum().backward()

    saliency = X_val_t.grad.abs().mean(dim=0).cpu().numpy()
    feat_imp = sorted(
        zip(preprocessor.feature_names, saliency), key=lambda x: x[1], reverse=True
    )

    logger.info("\nTop 10 Features (gradient saliency):")
    for name, imp in feat_imp[:10]:
        logger.info(f"  {name:<30} : {imp:.5f}")

    # ── Save Report ───────────────────────────────────────────────────
    summary = {
        "model":        "CNN+MLP Hybrid",
        "target":       args.target,
        "best_epoch":   best_epoch,
        "training_sec": round(elapsed, 2),
        "train_samples": len(train_idx),
        "val_samples":   len(val_idx),
        "model_params":  sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_f1":   round(best_val_f1, 4),
        "val_metrics": {
            "accuracy":          round(fm["accuracy"],          4),
            "balanced_accuracy": round(fm["balanced_accuracy"], 4),
            "macro_f1":          round(fm["macro_f1"],          4),
            "weighted_f1":       round(fm["weighted_f1"],       4),
            "roc_auc":           round(fm["roc_auc"],           4),
        },
        "confusion_matrix": cm.tolist(),
        "top_features": [{"feature": n, "importance": float(i)} for n, i in feat_imp[:10]],
    }

    report_file = save_dir / "cnn_mlp_training_report.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nReport saved -> {report_file}")
    logger.info(f"Checkpoint   -> {checkpoint}")
    logger.info("\nNext: Run python predict_cnn_mlp.py to generate test predictions.\n")


if __name__ == "__main__":
    main()
