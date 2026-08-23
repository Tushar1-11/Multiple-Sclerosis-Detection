"""
MSNet — Training Engine
========================
Full training pipeline with:
  - AMP (mixed-precision FP16/BF16) training
  - Gradient accumulation for larger effective batch sizes
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Early stopping with patience
  - Best model checkpointing
  - W&B / TensorBoard logging
  - Per-epoch metric reporting

Designed for MICCAI / TMI publication-quality results.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.msnet import MSNet
from training.losses import build_loss_function
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ============================================================
# LR Scheduler with Warmup
# ============================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build cosine annealing LR scheduler with linear warmup.

    Warmup helps stabilize early training with large learning rates
    when the model's representations are still random.

    Args:
        optimizer: Optimizer to schedule
        config: Training config dict
        steps_per_epoch: Number of gradient steps per epoch

    Returns:
        Composed sequential scheduler
    """
    train_cfg = config["training"]
    warmup_epochs = train_cfg["warmup_epochs"]
    total_epochs = train_cfg["epochs"]
    min_lr = train_cfg["min_lr"]
    base_lr = train_cfg["learning_rate"]

    # Linear warmup over warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # Chain them
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    logger.info(
        f"[Scheduler] Warmup {warmup_epochs} epochs -> "
        f"Cosine {total_epochs - warmup_epochs} epochs"
    )
    return scheduler


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping based on monitored validation metric.

    Args:
        patience: Epochs to wait for improvement before stopping
        mode: 'max' (higher is better) or 'min' (lower is better)
        min_delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Call each epoch with current metric score.

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")

        self.should_stop = (self.counter >= self.patience)
        return self.should_stop


# ============================================================
# Trainer
# ============================================================

class MSNetTrainer:
    """
    Full training engine for MSNet.

    Handles:
      - Training loop (AMP, grad accumulation)
      - Validation loop (deterministic)
      - Metric computation and logging
      - Model checkpointing (top-K)
      - Early stopping
      - Experiment tracking (W&B / TensorBoard)

    Args:
        model: MSNet model
        config: Full YAML config dict
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch.device
        class_weights: Optional class weights [n_classes]
    """

    def __init__(
        self,
        model: MSNet,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = config["data"]["num_classes"]
        self.task = config["data"]["task"]

        # ---- Directories ----
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # ---- Loss ----
        self.criterion = build_loss_function(config, class_weights)
        if class_weights is not None:
            self.criterion = self.criterion.to(device)

        # ---- Optimizer ----
        train_cfg = config["training"]
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            betas=tuple(train_cfg["betas"]),
        )

        # ---- Scheduler ----
        self.scheduler = build_scheduler(
            self.optimizer, config, len(train_loader)
        )

        # ---- AMP Scaler ----
        self.use_amp = train_cfg["use_amp"]
        self.amp_dtype = torch.float16 if train_cfg["amp_dtype"] == "float16" else torch.bfloat16
        _device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        # GradScaler only functional with CUDA; disable on CPU regardless
        _amp_enabled = self.use_amp and _device_type == "cuda"
        self.scaler = GradScaler(_device_type, enabled=_amp_enabled)
        self.use_amp = _amp_enabled  # disable AMP if not on CUDA
        self._device_type = _device_type

        # ---- Training Setup ----
        self.grad_accum_steps = train_cfg["grad_accum_steps"]
        self.grad_clip_norm = train_cfg["gradient_clip_norm"]
        self.max_epochs = train_cfg["epochs"]

        # ---- Early Stopping ----
        es_cfg = train_cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=es_cfg["patience"],
            mode=es_cfg["mode"],
            min_delta=es_cfg["min_delta"],
        ) if es_cfg["enabled"] else None

        # ---- Logging ----
        self.use_wandb = config["experiment"].get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb()

        # ---- State ----
        self.best_val_score = 0.0
        self.best_epoch = 0
        self.history = {"train": [], "val": []}

        logger.info(f"[Trainer] Initialized. AMP={self.use_amp}, Device={device}")

    def _init_wandb(self):
        """Initialize Weights & Biases experiment tracking."""
        try:
            import wandb
            exp = self.config["experiment"]
            wandb.init(
                project=exp.get("wandb_project", "msnet"),
                entity=exp.get("wandb_entity", None),
                name=exp["name"],
                config=self.config,
            )
            self.wandb = wandb
            logger.info("[W&B] Initialized successfully")
        except ImportError:
            logger.warning("[W&B] wandb not installed. Skipping.")
            self.use_wandb = False

    # ----------------------------------------------------------
    # Training Step
    # ----------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            Dict of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.max_epochs} [train]",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            # ---- Move to Device ----
            volumes = batch["volumes"].to(self.device, non_blocking=True)
            clinical = batch["clinical"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            modality_mask = batch["modality_mask"].to(self.device, non_blocking=True)

            # ---- Forward Pass (AMP) ----
            with autocast(device_type=self._device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(
                    volumes=volumes,
                    clinical=clinical,
                    modality_mask=modality_mask,
                )
                loss = self.criterion(output["logits"], labels)
                # Scale for gradient accumulation
                loss = loss / self.grad_accum_steps

            # ---- Backward ----
            self.scaler.scale(loss).backward()

            # ---- Gradient Accumulation ----
            if (step + 1) % self.grad_accum_steps == 0:
                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # ---- Accumulate Predictions ----
            total_loss += loss.item() * self.grad_accum_steps
            preds = output["pred_class"].cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}"})

        # ---- Compute Epoch Metrics ----
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_all_metrics(
            labels=all_labels,
            preds=all_preds,
            num_classes=self.num_classes,
            probs=None,  # Train metrics don't need AUC for speed
        )
        metrics["loss"] = avg_loss

        logger.info(
            f"\n[Train] Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | Acc={metrics.get('accuracy', 0):.4f} | "
            f"F1={metrics.get('f1_macro', 0):.4f} | "
            f"LR={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        return metrics

    # ----------------------------------------------------------
    # Validation Step
    # ----------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run deterministic validation.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.max_epochs} [val]",
            leave=False,
        )

        for batch in pbar:
            volumes = batch["volumes"].to(self.device, non_blocking=True)
            clinical = batch["clinical"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            modality_mask = batch["modality_mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self._device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(volumes=volumes, clinical=clinical,
                                    modality_mask=modality_mask)
                loss = self.criterion(output["logits"], labels)

            total_loss += loss.item()
            all_preds.extend(output["pred_class"].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(output["probabilities"].cpu().numpy().tolist())

        # ---- Compute Full Validation Metrics ----
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_all_metrics(
            labels=all_labels,
            preds=all_preds,
            num_classes=self.num_classes,
            probs=all_probs,
        )
        metrics["loss"] = avg_loss

        logger.info(
            f"[Val]   Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | Acc={metrics.get('accuracy', 0):.4f} | "
            f"AUC={metrics.get('roc_auc', 0):.4f} | "
            f"F1={metrics.get('f1_macro', 0):.4f}"
        )
        return metrics

    # ----------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        val_score: float,
        metrics: Dict,
        tag: str = "best",
    ):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_score": val_score,
            "metrics": metrics,
            "config": self.config,
        }
        path = self.checkpoint_dir / f"msnet_{tag}_epoch{epoch:03d}.pth"
        torch.save(ckpt, path)
        logger.info(f"[Checkpoint] Saved → {path}")

        # Also save a fixed 'best_model.pth' for convenient loading
        if tag == "best":
            best_path = self.output_dir / "best_model.pth"
            torch.save(ckpt, best_path)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    def _log_metrics(self, metrics: Dict, split: str, epoch: int):
        """Log metrics to W&B / TensorBoard."""
        prefixed = {f"{split}/{k}": v for k, v in metrics.items()}
        prefixed["epoch"] = epoch

        if self.use_wandb:
            self.wandb.log(prefixed)

    # ----------------------------------------------------------
    # Main Training Loop
    # ----------------------------------------------------------

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Training history dict with per-epoch metrics
        """
        logger.info(f"\n{'='*60}\nStarting MSNet Training\n{'='*60}")
        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            # ---- Train ----
            train_metrics = self._train_epoch(epoch)
            self._log_metrics(train_metrics, "train", epoch)
            self.history["train"].append(train_metrics)

            # ---- Validate ----
            val_metrics = self._validate_epoch(epoch)
            self._log_metrics(val_metrics, "val", epoch)
            self.history["val"].append(val_metrics)

            # ---- LR Scheduler Step ----
            self.scheduler.step()

            # ---- Best Model Saving ----
            monitor_metric = self.config["training"]["early_stopping"]["monitor"]
            # Strip 'val/' prefix if present
            metric_key = monitor_metric.replace("val/", "")
            val_score = val_metrics.get(metric_key, val_metrics.get("roc_auc", 0.0))

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_score, val_metrics, tag="best")
                logger.info(
                    f"[Best] New best {monitor_metric}={val_score:.4f} at epoch {epoch}"
                )

            # ---- Periodic Checkpoint ----
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_score, val_metrics, tag="periodic")

            # ---- Early Stopping ----
            if self.early_stopping and self.early_stopping(val_score):
                logger.info(
                    f"[EarlyStopping] Stopping at epoch {epoch}. "
                    f"Best was epoch {self.best_epoch} with {monitor_metric}={self.best_val_score:.4f}"
                )
                break

        # ---- Training Complete ----
        elapsed = time.time() - start_time
        logger.info(
            f"\n{'='*60}\n"
            f"Training Complete\n"
            f"  Total time:    {elapsed / 60:.1f} minutes\n"
            f"  Best epoch:    {self.best_epoch}\n"
            f"  Best val score: {self.best_val_score:.4f}\n"
            f"{'='*60}"
        )

        if self.use_wandb:
            self.wandb.finish()

        return self.history
