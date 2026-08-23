"""
MSNet — Main Training Entry Point
===================================
Orchestrates the full training pipeline:
  1. Load and validate config
  2. Set seeds for reproducibility
  3. Load and split dataset
  4. Build DataLoaders
  5. Initialize MSNet model
  6. Train with early stopping
  7. Evaluate best model on test set
  8. Save final report

Usage:
  python train.py --config configs/default_config.yaml

  # Override config via CLI (Hydra-style):
  python train.py --config configs/default_config.yaml \
      training.learning_rate=5e-5 \
      data.batch_size=8
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing.clinical_processor import ClinicalProcessor, stratified_split
from data.dataset import build_dataloaders
from models.msnet import MSNet
from training.trainer import MSNetTrainer
from evaluation.evaluator import MSNetEvaluator
from utils.seed import set_seed
from utils.logger import setup_logging
from utils.checkpoint import load_checkpoint


# ============================================================
# Argument Parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MSNet Training Pipeline — MS Detection from Multimodal MRI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; only evaluate checkpoint on test set",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path (required if --eval_only)",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    # Allow inline config overrides
    parser.add_argument(
        "overrides", nargs="*",
        help="Key=value config overrides (e.g., training.learning_rate=1e-4)",
    )
    return parser.parse_args()


# ============================================================
# Config Loading & Override
# ============================================================

def load_config(config_path: str, overrides: list) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply dot-notation overrides: key.subkey=value
    for override in overrides:
        if "=" not in override:
            continue
        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the correct nested dict
        ref = config
        for k in keys[:-1]:
            ref = ref[k]

        # Parse value type
        raw = value
        try:
            if "." in raw:
                raw = float(raw)
            else:
                raw = int(raw)
        except ValueError:
            if raw.lower() in ("true", "false"):
                raw = raw.lower() == "true"
            # else keep as string

        ref[keys[-1]] = raw
        logging.info(f"[Config] Override: {key_path} = {raw}")

    return config


# ============================================================
# Main Training Function
# ============================================================

def main():
    args = parse_args()

    # ---- Logging ----
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    # ---- Config ----
    config = load_config(args.config, args.overrides)
    logger.info(f"[Config] Loaded from: {args.config}")

    # ---- Reproducibility ----
    seed = config["experiment"]["seed"]
    set_seed(seed)
    logger.info(f"[Seed] Set to {seed}")

    # ---- Device ----
    device_str = config["hardware"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("[Device] CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"[Device] Using: {device}")

    # ============================================================
    # Data Loading
    # ============================================================

    logger.info("[Data] Loading metadata CSV...")
    df = pd.read_csv(config["data"]["csv_path"])
    logger.info(f"[Data] Total subjects: {len(df)}")

    # ---- Clinical Processing ----
    clinical_proc = ClinicalProcessor(
        features=config["data"]["clinical_features"],
        label_col="label",
    )

    # ---- Stratified Split ----
    train_df, val_df, test_df = stratified_split(
        df,
        label_col="label",
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        seed=seed,
    )

    # Fit & transform clinical features
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["__clinical_processed__"] = clinical_proc.fit_transform(train_df).tolist()
    val_df["__clinical_processed__"]   = clinical_proc.transform(val_df).tolist()
    test_df["__clinical_processed__"]  = clinical_proc.transform(test_df).tolist()

    # ---- Build DataLoaders ----
    logger.info("[Data] Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
    )
    logger.info(
        f"[Data] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}"
    )

    # ============================================================
    # Model Initialization
    # ============================================================

    logger.info("[Model] Initializing MSNet...")
    model = MSNet(config)

    # Optional: Compile model for PyTorch 2.0+ performance
    if config["hardware"].get("compile_model", False):
        logger.info("[Model] Compiling with torch.compile...")
        model = torch.compile(model)

    # ---- Resume from Checkpoint ----
    start_epoch = 1
    if args.resume:
        logger.info(f"[Checkpoint] Resuming from: {args.resume}")
        start_epoch = load_checkpoint(model, args.resume, device) + 1

    # ============================================================
    # Training (unless eval_only)
    # ============================================================

    if not args.eval_only:
        # ---- Class Weights for Imbalanced Data ----
        class_weights = None
        if config["training"]["loss"].get("use_class_weights", False):
            class_weights = train_loader.dataset.get_class_weights()
            logger.info(f"[Loss] Class weights: {class_weights.tolist()}")

        # ---- Initialize Trainer ----
        trainer = MSNetTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_weights=class_weights,
        )

        # ---- Run Training ----
        history = trainer.train()

        # ---- Load Best Checkpoint for Evaluation ----
        best_ckpt = str(Path(config["experiment"]["output_dir"]) / "best_model.pth")
        logger.info(f"[Eval] Loading best model from: {best_ckpt}")
        load_checkpoint(model, best_ckpt, device)

    elif args.checkpoint:
        logger.info(f"[Eval Only] Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, device)
    else:
        raise ValueError("--eval_only requires --checkpoint to be specified")

    # ============================================================
    # Final Test Evaluation
    # ============================================================

    logger.info("\n[Eval] Running final test set evaluation...")
    evaluator = MSNetEvaluator(
        model=model,
        config=config,
        device=device,
        output_dir=config["experiment"]["output_dir"] + "/evaluation",
    )

    eval_results = evaluator.evaluate(
        loader=test_loader,
        save_gradcam=config["explainability"]["gradcam"]["enabled"],
        n_gradcam_samples=10,
        bootstrap_n=config["evaluation"]["bootstrap"]["n_bootstraps"],
    )

    # ---- Print Final Summary ----
    m = eval_results["metrics"]
    logger.info(
        f"\n{'='*60}\n"
        f"FINAL TEST SET RESULTS\n"
        f"{'='*60}\n"
        f"  Accuracy:    {m.get('accuracy',  0):.4f}\n"
        f"  Sensitivity: {m.get('sensitivity', 0):.4f}\n"
        f"  Specificity: {m.get('specificity', 0):.4f}\n"
        f"  F1-Score:    {m.get('f1_score', 0):.4f}\n"
        f"  ROC-AUC:     {m.get('roc_auc',  0):.4f}\n"
        f"  PR-AUC:      {m.get('pr_auc',   0):.4f}\n"
        f"  MCC:         {m.get('mcc',      0):.4f}\n"
        f"  Kappa:       {m.get('cohen_kappa', 0):.4f}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
