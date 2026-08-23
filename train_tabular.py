"""
MSNet — Fast 1D-CNN Training Pipeline for Tabular MS Data
=========================================================
Trains and evaluates the 1D-CNN ResNet model on `train_data.csv`.

Usage:
  python train_tabular.py
  python train_tabular.py --config configs/tabular_config.yaml --target group
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.tabular_preprocessor import TabularPreprocessor
from data.tabular_dataset import build_tabular_dataloaders
from models.tabular_msnet import TabularMSNet
from utils.seed import set_seed
from utils.logger import setup_logging

logger = logging.getLogger("MSNet.Tabular")


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in tabular classification.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# Metric Evaluation Helper
# ============================================================

def evaluate_model(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int
) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = F.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    # Compute ROC-AUC
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
    }


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN on tabular MS dataset")
    parser.add_argument("--config", type=str, default="configs/tabular_config.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None, choices=["phenotype", "group"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--full_train", action="store_true", help="Train on 100% of train_data.csv")
    args = parser.parse_args()

    # Setup logger
    setup_logging(log_level="INFO")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {"data_path": "data/train_data.csv", "target": "phenotype", "batch_size": 32},
            "model": {"embed_dim": 256, "dropout": 0.25, "num_heads": 4},
            "training": {"epochs": 50, "lr": 1e-3, "seed": 42, "save_dir": "outputs/tabular_1dcnn"},
        }

    # Override CLI args
    if args.data_path:
        config["data"]["data_path"] = args.data_path
    if args.target:
        config["data"]["target"] = args.target
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    save_dir = Path(train_cfg.get("save_dir", "outputs/tabular_1dcnn"))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  MSNet 1D-CNN Training Pipeline (Tabular MS Biomarkers)")
    logger.info("=" * 65)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Load & Preprocess Data
    preprocessor = TabularPreprocessor(
        target_name=data_cfg.get("target", "phenotype"),
        test_size=data_cfg.get("test_size", 0.15),
        val_size=data_cfg.get("val_size", 0.15),
        random_state=seed,
    )

    df = preprocessor.load_data(data_cfg["data_path"])
    X, y, splits = preprocessor.fit_transform(df)

    if args.full_train:
        logger.info("--> Running FULL TRAINING mode on 100% of train_data.csv (all 1659 samples) <--")
        splits["train"] = np.arange(len(y))

    num_classes = len(np.unique(y))
    in_features = X.shape[1]

    target_labels = (
        ["Healthy Control", "RRMS", "Progressive MS"]
        if num_classes == 3
        else ["Healthy Control", "MS Patient"]
    )
    logger.info(
        f"Input features: {in_features} | Target classes: {num_classes} ({', '.join(target_labels)})"
    )

    # Build DataLoaders
    train_loader, val_loader, test_loader, class_weights = build_tabular_dataloaders(
        X,
        y,
        splits,
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 0),
        use_balanced_sampler=data_cfg.get("use_balanced_sampler", True),
    )

    # Instantiate Model
    model = TabularMSNet(
        in_features=in_features,
        num_classes=num_classes,
        embed_dim=model_cfg.get("embed_dim", 256),
        dropout=model_cfg.get("dropout", 0.25),
        num_heads=model_cfg.get("num_heads", 4),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture: Tabular 1D-CNN (Trainable parameters: {num_params:,})")

    # Loss Function & Optimizer
    if train_cfg.get("use_focal_loss", True):
        criterion = FocalLoss(
            alpha=class_weights.to(device), gamma=float(train_cfg.get("focal_gamma", 2.0))
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop with Early Stopping
    best_val_f1 = 0.0
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 12)
    patience_counter = 0
    checkpoint_path = save_dir / "best_tabular_1dcnn.pt"

    start_time = time.time()

    logger.info("\nStarting 1D-CNN model training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, num_classes)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        if epoch % 5 == 0 or epoch == 1 or val_f1 > best_val_f1:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}"
            )

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "preprocessor": preprocessor,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
                break

    training_time = time.time() - start_time
    logger.info(f"\nTraining completed in {training_time:.2f} seconds.")

    # Final Test Set Evaluation
    logger.info("\nEvaluating Best Model on Test Set...")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, device, num_classes)

    logger.info("\n" + "=" * 65)
    logger.info("              TEST SET EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info(f"Accuracy:           {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"Macro F1 Score:     {test_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1 Score:  {test_metrics['weighted_f1']:.4f}")
    logger.info(f"ROC-AUC Score:      {test_metrics['roc_auc']:.4f}")
    logger.info("=" * 65)

    # Print Detailed Classification Report
    report = classification_report(
        test_metrics["targets"], test_metrics["preds"], target_names=target_labels, digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])
    logger.info(f"Confusion Matrix:\n{cm}")

    # Feature Importance Attribution (Saliency via Gradients)
    logger.info("\nComputing Feature Importance Attribution...")
    model.eval()
    X_test_tensor = torch.tensor(X[splits["test"]], dtype=torch.float32, requires_grad=True).to(device)
    logits = model(X_test_tensor)

    # Backward gradient of top logit
    top_logits, _ = torch.max(logits, dim=1)
    top_logits.sum().backward()

    saliency = X_test_tensor.grad.abs().mean(dim=0).cpu().numpy()
    feature_importance = sorted(
        zip(preprocessor.feature_names, saliency), key=lambda x: x[1], reverse=True
    )

    logger.info("\nTop 10 Most Important Features:")
    for feat_name, imp in feature_importance[:10]:
        logger.info(f"  - {feat_name:<25}: {imp:.5f}")

    # Save summary report
    summary = {
        "target": data_cfg.get("target"),
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_time, 2),
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "balanced_accuracy": round(test_metrics["balanced_accuracy"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "weighted_f1": round(test_metrics["weighted_f1"], 4),
            "roc_auc": round(test_metrics["roc_auc"], 4),
        },
        "confusion_matrix": cm.tolist(),
        "top_features": [
            {"feature": f, "importance": float(imp)} for f, imp in feature_importance[:10]
        ],
    }

    report_file = save_dir / "tabular_training_report.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nReport saved to {report_file}")
    logger.info(f"Model saved to {checkpoint_path}\n")


if __name__ == "__main__":
    main()
