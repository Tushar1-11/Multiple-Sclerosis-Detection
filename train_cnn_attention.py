"""
MSNet -- CNN + Attention Hybrid Training Pipeline
==================================================
Trains CNNAttentionHybrid on ALL 1660 train samples.

Phase 1 (warm-up, 15 epochs):
  CNN frozen -- only Transformer + heads trained at high LR.
  Lets attention layers learn meaningful patterns before CNN moves.

Phase 2 (full fine-tune, remaining epochs):
  CNN unfrozen -- differential LR:
    CNN backbone  : lr * 0.05  (very small to not destroy features)
    Transformer   : lr * 0.5
    Heads         : lr * 1.0

Advanced techniques:
  - 2-phase training with differential LR
  - Mixup augmentation
  - Label smoothing + Focal Loss
  - Cosine Annealing Warm Restarts
  - Gradient accumulation (effective batch 128)
  - Gradient clipping

Usage:
  python train_cnn_attention.py
  python train_cnn_attention.py --target phenotype --epochs 200
  python train_cnn_attention.py --skip_warmup --epochs 150
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from data.tabular_dataset import build_tabular_dataloaders
from models.cnn_attention import CNNAttentionHybrid
from utils.seed import set_seed
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.CNNAttention")

# Exclude columns that cause biased missing-value indicators
# (HC patients have EDSS/Disease duration missing in train but NOT in test)
EXCLUDE_FEATURE_COLS = ["EDSS", "Disease duration"]


# ============================================================
# Focal Loss + Label Smoothing
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        C = logits.shape[-1]
        smooth = torch.full_like(logits, self.ls / (C - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p  = F.log_softmax(logits, -1)
        ce     = -(smooth * log_p).sum(-1)
        pt     = torch.exp(-F.cross_entropy(logits, targets,
                                             weight=self.alpha, reduction="none"))
        return (((1 - pt) ** self.gamma) * ce).mean()


# ============================================================
# Mixup
# ============================================================

def mixup(x, y, edss, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.shape[0], device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], edss, edss[idx], lam


def cls_loss_mix(crit, logits, ya, yb, lam):
    return lam*crit(logits, ya) + (1-lam)*crit(logits, yb)


def reg_loss_mix(pred, ea, eb, lam):
    return F.smooth_l1_loss(pred.squeeze(-1), lam*ea + (1-lam)*eb)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device, num_classes, task="phenotype"):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out    = model(Xb.to(device), return_all=True)
            logits = out[task]
            preds.extend(torch.argmax(logits, -1).cpu().numpy())
            targets.extend(yb.numpy())
            probs.extend(F.softmax(logits, -1).cpu().numpy())

    p, t, pr = np.array(preds), np.array(targets), np.array(probs)
    try:
        auc = (roc_auc_score(t, pr[:,1]) if num_classes==2
               else roc_auc_score(t, pr, multi_class="ovr", average="macro"))
    except Exception:
        auc = 0.0
    return {
        "accuracy":          accuracy_score(t, p),
        "balanced_accuracy": balanced_accuracy_score(t, p),
        "macro_f1":          f1_score(t, p, average="macro",    zero_division=0),
        "weighted_f1":       f1_score(t, p, average="weighted", zero_division=0),
        "roc_auc": auc,
        "targets": t, "preds": p, "probs": pr,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",        type=str,   default="phenotype",
                    choices=["phenotype", "group"])
    ap.add_argument("--epochs",        type=int,   default=200)
    ap.add_argument("--warmup_epochs", type=int,   default=15,
                    help="Phase-1 epochs with CNN frozen")
    ap.add_argument("--skip_warmup",   action="store_true")
    ap.add_argument("--batch_size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--warmup_lr",     type=float, default=1e-3,
                    help="LR for Phase-1 (Attention + heads only)")
    ap.add_argument("--token_dim",     type=int,   default=256)
    ap.add_argument("--attn_dim",      type=int,   default=256)
    ap.add_argument("--num_layers",    type=int,   default=6,
                    help="Number of Transformer blocks")
    ap.add_argument("--num_heads",     type=int,   default=8)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--val_size",      type=float, default=0.10)
    ap.add_argument("--mixup_alpha",   type=float, default=0.4)
    ap.add_argument("--focal_gamma",   type=float, default=2.0)
    ap.add_argument("--label_smooth",  type=float, default=0.1)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--grad_accum",    type=int,   default=2)
    ap.add_argument("--patience",      type=int,   default=30)
    ap.add_argument("--no_mixup",      action="store_true")
    ap.add_argument("--save_dir",      type=str,   default="outputs/cnn_attention")
    args = ap.parse_args()

    setup_logging(log_level="INFO")
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 72)
    logger.info("  MSNet - CNN + Attention Hybrid Training")
    logger.info("=" * 72)
    logger.info(f"  Device         : {device}")
    logger.info(f"  Target         : {args.target}")
    logger.info(f"  Epochs         : {args.epochs} (warmup {args.warmup_epochs})")
    logger.info(f"  Transformer    : {args.num_layers} layers x {args.num_heads} heads")
    logger.info(f"  Token dim      : {args.token_dim} | Attn dim: {args.attn_dim}")
    logger.info(f"  Batch size     : {args.batch_size} x {args.grad_accum} accum")
    logger.info("=" * 72)

    # ── Load Discrete Train & Val Split Files ──────────────────────────
    train_path = Path("data/train_split.csv")
    val_path   = Path("data/val_data.csv")

    preprocessor = TabularPreprocessor(
        target_name=args.target,
        random_state=args.seed,
        exclude_cols=EXCLUDE_FEATURE_COLS,
        use_feature_engineering=True,
    )

    df_train = preprocessor.load_data(train_path)
    df_val   = preprocessor.load_data(val_path)

    # Raw EDSS values for regression targets
    raw_edss_train = np.clip(df_train["EDSS"].fillna(0.0).values.astype(np.float32), 0.0, 10.0)
    raw_edss_val   = np.clip(df_val["EDSS"].fillna(0.0).values.astype(np.float32), 0.0, 10.0)

    # Fit preprocessor on train_split.csv only, transform val_data.csv
    X_train, y_train, _ = preprocessor.fit_transform(df_train)
    X_val, y_val        = preprocessor.transform(df_val)

    # Combine into single array with explicit index mapping
    n_train = len(y_train)
    n_val   = len(y_val)

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    edss_all = torch.tensor(np.concatenate([raw_edss_train, raw_edss_val]), dtype=torch.float32)

    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    splits    = {"train": train_idx, "val": val_idx, "test": val_idx}

    in_features = X.shape[1]
    num_classes = len(np.unique(y_train))
    target_labels = (["Healthy Control", "RRMS", "Progressive MS"]
                     if args.target == "phenotype"
                     else ["Healthy Control", "MS Patient"])

    logger.info(f"  Input features : {in_features} | Classes: {num_classes}")
    logger.info(f"  Excluded cols  : {EXCLUDE_FEATURE_COLS}")
    logger.info(f"  Train split    : {n_train} samples (data/train_split.csv)")
    logger.info(f"  Val split      : {n_val} samples (data/val_data.csv)")

    train_loader, val_loader, _, class_weights = build_tabular_dataloaders(
        X, y, splits, batch_size=args.batch_size,
        num_workers=0, use_balanced_sampler=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = CNNAttentionHybrid(
        in_features=in_features,
        token_dim=args.token_dim,
        cnn_channels=(128, 256, 256, 256),
        cnn_seq_len=32,
        cnn_dropout=0.2,
        cnn_drop_path_rate=0.10,
        cnn_blocks_per_stage=(2, 3, 3, 2),
        num_attn_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_expansion=4,
        attn_dropout=0.1,
        attn_drop_path_rate=0.10,
        attn_dim=args.attn_dim,
        head_dropout=0.3,
        num_phenotype_classes=3,
        num_group_classes=2,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=args.focal_gamma,
        label_smoothing=args.label_smooth,
    )
    lambda_edss = 0.2

    # ── Phase 1: freeze CNN, train attention + heads ──────────────────
    if not args.skip_warmup:
        for p in model.branch1_cnn.parameters():
            p.requires_grad = False
        logger.info("[Phase 1] CNN stem frozen -- training Parallel Embeddings + Recurrent + Attention + Heads only")

    def make_opt(lr, p2=False):
        if p2:
            return torch.optim.AdamW([
                {"params": model.branch1_cnn.parameters(),       "lr": lr * 0.05},
                {"params": model.branch2_ft.parameters(),        "lr": lr * 0.2},
                {"params": model.branch3_tab.parameters(),       "lr": lr * 0.2},
                {"params": model.recurrent_fusion.parameters(),  "lr": lr * 0.5},
                {"params": model.transformer_blocks.parameters(),"lr": lr * 0.5},
                {"params": model.cls_token,                      "lr": lr * 0.5},
                {"params": model.pos_embed,                      "lr": lr * 0.5},
                {"params": model.latent_fusion.parameters(),     "lr": lr},
                {"params": model.phenotype_head.parameters(),    "lr": lr},
                {"params": model.group_head.parameters(),        "lr": lr},
                {"params": model.edss_head.parameters(),         "lr": lr},
            ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
        else:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
            )

    optimizer = make_opt(args.warmup_lr if not args.skip_warmup else args.lr)
    T_0       = max(40, args.epochs // 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=2, eta_min=1e-7
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_f1  = 0.0
    best_ep  = 0
    pat_ctr  = 0
    ckpt     = save_dir / "best_cnn_attention.pt"
    phase    = 1
    use_mix  = not args.no_mixup
    t0       = time.time()

    logger.info("\nStarting CNN + Attention training...\n")

    for epoch in range(1, args.epochs + 1):

        # Phase transition
        if phase == 1 and (args.skip_warmup or epoch > args.warmup_epochs):
            phase = 2
            for p in model.branch1_cnn.parameters():
                p.requires_grad = True
            optimizer = make_opt(args.lr, p2=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=2, eta_min=1e-7
            )
            logger.info(f"\n[Phase 2 - Epoch {epoch}] CNN unfrozen. Full end-to-end multi-branch training.\n")

        model.train()
        tr_loss = 0.0
        nb      = 0
        optimizer.zero_grad()

        for bidx, (Xb, yb) in enumerate(train_loader):
            Xb     = Xb.to(device)
            yb_cls = yb.to(device)

            # Raw EDSS for this batch
            bs = bidx * args.batch_size
            be = min(bs + Xb.shape[0], len(train_idx))
            bi = train_idx[bs:be]
            y_edss = (edss_all[bi].to(device)
                      if len(bi) == Xb.shape[0]
                      else torch.zeros(Xb.shape[0], device=device))

            # Mixup
            if use_mix and epoch > args.warmup_epochs:
                Xm, ya, yb_, ea, eb, lam = mixup(Xb, yb_cls, y_edss, args.mixup_alpha)
            else:
                Xm, ya, yb_, ea, eb, lam = Xb, yb_cls, yb_cls, y_edss, y_edss, 1.0

            out    = model(Xm, return_all=True)
            c_loss = cls_loss_mix(criterion, out[args.target], ya, yb_, lam)
            r_loss = reg_loss_mix(out["edss"], ea, eb, lam)
            loss   = (c_loss + lambda_edss * r_loss) / args.grad_accum
            loss.backward()

            if (bidx + 1) % args.grad_accum == 0 or (bidx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            tr_loss += (c_loss + lambda_edss * r_loss).item()
            nb      += 1

        scheduler.step(epoch)
        tr_loss /= nb

        vm      = evaluate(model, val_loader, device, num_classes, args.target)
        vf1     = vm["macro_f1"]
        vacc    = vm["accuracy"]
        vauc    = vm["roc_auc"]
        lr_now  = optimizer.param_groups[0]["lr"]
        ph_tag  = f"[P{phase}]"

        if epoch % 5 == 0 or epoch == 1 or vf1 > best_f1:
            logger.info(
                f"{ph_tag} Epoch {epoch:03d}/{args.epochs} | "
                f"Loss: {tr_loss:.4f} | "
                f"Val Acc: {vacc*100:.2f}% | "
                f"Val F1: {vf1:.4f} | "
                f"Val AUC: {vauc:.4f} | "
                f"LR: {lr_now:.2e}"
            )

        if vf1 > best_f1:
            best_f1  = vf1
            best_ep  = epoch
            pat_ctr  = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics":      {k:v for k,v in vm.items()
                                     if not isinstance(v, np.ndarray)},
                "preprocessor":     preprocessor,
                "model_config": {
                    "in_features":    in_features,
                    "token_dim":      args.token_dim,
                    "attn_dim":       args.attn_dim,
                    "num_layers":     args.num_layers,
                    "num_heads":      args.num_heads,
                    "target":         args.target,
                    "exclude_cols":   EXCLUDE_FEATURE_COLS,
                },
                "args": vars(args),
            }, ckpt)
            logger.info(f"  >> Best saved (Val F1: {best_f1:.4f} @ Epoch {epoch})")
        else:
            pat_ctr += 1
            if pat_ctr >= args.patience:
                logger.info(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best Val F1: {best_f1:.4f} (Epoch {best_ep})"
                )
                break

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Final Evaluation ──────────────────────────────────────────────
    logger.info("\nLoading best checkpoint...")
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])

    fm = evaluate(model, val_loader, device, num_classes, args.target)
    logger.info("\n" + "=" * 72)
    logger.info("  FINAL RESULTS -- CNN + Attention Hybrid")
    logger.info("=" * 72)
    logger.info(f"  Accuracy          : {fm['accuracy']*100:.2f}%")
    logger.info(f"  Balanced Accuracy : {fm['balanced_accuracy']*100:.2f}%")
    logger.info(f"  Macro F1          : {fm['macro_f1']:.4f}")
    logger.info(f"  Weighted F1       : {fm['weighted_f1']:.4f}")
    logger.info(f"  ROC-AUC           : {fm['roc_auc']:.4f}")
    logger.info("=" * 72)
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(fm['targets'], fm['preds'], target_names=target_labels, digits=4)}")
    cm = confusion_matrix(fm["targets"], fm["preds"])
    logger.info(f"Confusion Matrix:\n{cm}")

    # Feature importance
    model.eval()
    Xv = torch.tensor(X[splits["val"]], dtype=torch.float32, requires_grad=True).to(device)
    out = model(Xv, return_all=True)
    torch.max(out[args.target], 1)[0].sum().backward()
    sal = Xv.grad.abs().mean(0).cpu().numpy()
    fi  = sorted(zip(preprocessor.feature_names, sal), key=lambda z: z[1], reverse=True)
    logger.info("\nTop 10 Features (saliency):")
    for n, s in fi[:10]:
        logger.info(f"  {n:<30}: {s:.5f}")

    # Save report
    summary = {
        "model":         "CNN + Attention Hybrid",
        "target":        args.target,
        "best_epoch":    best_ep,
        "training_sec":  round(elapsed, 2),
        "train_samples": len(train_idx),
        "val_samples":   len(val_idx),
        "total_params":  sum(p.numel() for p in model.parameters()),
        "best_val_f1":   round(best_f1, 4),
        "val_metrics": {
            "accuracy":          round(fm["accuracy"],          4),
            "balanced_accuracy": round(fm["balanced_accuracy"], 4),
            "macro_f1":          round(fm["macro_f1"],          4),
            "weighted_f1":       round(fm["weighted_f1"],       4),
            "roc_auc":           round(fm["roc_auc"],           4),
        },
        "confusion_matrix": cm.tolist(),
        "top_features":     [{"feature": n, "importance": float(s)} for n,s in fi[:10]],
    }
    report_f = save_dir / "cnn_attention_report.json"
    with open(report_f, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nReport     -> {report_f}")
    logger.info(f"Checkpoint -> {ckpt}")
    logger.info("\nNext: python predict_cnn_attention.py\n")


if __name__ == "__main__":
    main()
