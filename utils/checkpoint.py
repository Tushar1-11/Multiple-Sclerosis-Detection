"""
MSNet — Checkpoint Utilities
==============================
Handles model checkpoint saving and loading with:
  - Full state preservation (model, optimizer, scheduler, scaler)
  - Version info and training metadata
  - Graceful missing key handling for partial loads
  - Top-K checkpoint management
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
) -> int:
    """
    Load model (and optionally optimizer/scheduler) from a checkpoint.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to .pth checkpoint file
        device: Target device
        optimizer: Optional optimizer to restore
        scheduler: Optional LR scheduler to restore
        scaler: Optional GradScaler to restore
        strict: Whether to enforce strict key matching

    Returns:
        epoch: The epoch at which this checkpoint was saved
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"[Checkpoint] Loading from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    state_dict = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Optionally restore optimizer state
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    val_score = ckpt.get("val_score", None)
    logger.info(
        f"[Checkpoint] Loaded epoch={epoch}"
        + (f", val_score={val_score:.4f}" if val_score else "")
    )

    return epoch
