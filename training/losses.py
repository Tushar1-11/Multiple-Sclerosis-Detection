"""
MSNet — Custom Loss Functions
================================
Implements:
  1. Focal Loss — addresses severe class imbalance in MS datasets
  2. Focal Cross-Entropy — combines focal weighting with CE
  3. Label Smoothing Cross-Entropy — prevents overconfidence
  4. Evidential Deep Learning Loss — for EDL uncertainty training
  5. Composite Loss — combines multiple objectives

Motivation:
  MS datasets typically exhibit severe class imbalance:
  - HC >> RRMS >> SPMS >> PPMS (rarer subtypes harder to get)
  Focal loss down-weights easy negative samples and forces the
  model to focus on hard-to-classify (minority) cases.

Reference:
  Lin et al. (2017) — "Focal Loss for Dense Object Detection", ICCV 2017
  Müller et al. (2019) — "When Does Label Smoothing Help?", NeurIPS
  Sensoy et al. (2018) — "Evidential Deep Learning", NeurIPS
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for addressing class imbalance.

    Modulates CE loss by (1 - p_t)^gamma, where p_t is the model's
    estimated probability for the TRUE class. Easy-to-classify samples
    (p_t → 1) contribute less to the total loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        num_classes: Number of output classes
        gamma: Focusing parameter (gamma=0 → standard CE; gamma=2 typical)
        alpha: Per-class weight Tensor [n_classes] or None (uniform)
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: Smooth target labels (0.0–0.2 typical)
    """

    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Register alpha as buffer (device-agnostic)
        if alpha is not None:
            assert len(alpha) == num_classes
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer(
                "alpha", torch.ones(num_classes, dtype=torch.float32)
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [B, n_classes] — raw model outputs
            targets: [B] — integer class labels (0-based)

        Returns:
            Scalar loss
        """
        B = logits.size(0)

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [B, C]

        # One-hot + label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / self.num_classes
            target_soft = torch.full_like(probs, smooth)
            target_soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
        else:
            target_soft = F.one_hot(targets, self.num_classes).float()

        # Log probabilities (numerical stability via log_softmax)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]

        # p_t: probability of true class for focal weighting
        p_t = (probs * target_soft).sum(dim=-1)  # [B]

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma  # [B]

        # Alpha weight (per-class)
        alpha_t = (self.alpha.unsqueeze(0) * target_soft).sum(dim=-1)  # [B]

        # Focal loss
        ce = -(log_probs * target_soft).sum(dim=-1)   # [B]
        fl = alpha_t * focal_weight * ce              # [B]

        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl


# ============================================================
# Label Smoothing Cross-Entropy
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for overconfidence prevention.

    Converts hard one-hot targets to soft targets:
      y_smooth = (1 - eps) * y_hard + eps / C

    This prevents the model from assigning probability 1.0 to any class,
    improving calibration and generalization.

    Args:
        smoothing: Smoothing factor epsilon (0.1 typical)
        reduction: 'mean' | 'sum'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_val = self.smoothing / n_classes
            target_smooth = torch.full_like(log_p, smooth_val)
            target_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + smooth_val)

        loss = -(target_smooth * log_p).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ============================================================
# Evidential Deep Learning Loss
# ============================================================

class EvidentialLoss(nn.Module):
    """
    Loss for Evidential Deep Learning training.

    Combines:
      1. NLL loss on Dirichlet predictive distribution
      2. KL regularization toward uniform Dirichlet (prior)

    L_EDL = L_NLL + lambda * L_KL

    Args:
        num_classes: Number of output classes
        kl_weight: KL regularization weight (annealed during training)
    """

    def __init__(self, num_classes: int = 4, kl_weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 1,
        max_epochs: int = 100,
    ) -> torch.Tensor:
        """
        Compute evidential loss.

        Args:
            alpha: [B, C] — Dirichlet parameters
            targets: [B] — integer class labels
            epoch: Current epoch (for KL annealing)
            max_epochs: Total epochs (for annealing schedule)

        Returns:
            Scalar loss
        """
        C = self.num_classes
        S = alpha.sum(dim=-1, keepdim=True)         # [B, 1]

        # One-hot targets
        y = F.one_hot(targets, C).float()           # [B, C]

        # NLL loss: E[log S - log alpha_k]
        nll = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        # KL divergence from Dirichlet prior (uniform -> all alpha = 1)
        # KL[Dir(alpha') || Dir(1)] where alpha' = alpha after removing true class evidence
        alpha_tilde = y + (1 - y) * alpha           # Adjust evidence
        kl = self._kl_dirichlet(alpha_tilde)        # [B]

        # KL annealing: ramp from 0 to 1 over first max_epochs/2
        annealing = min(1.0, epoch / (max_epochs * 0.5))
        loss = (nll + self.kl_weight * annealing * kl).mean()

        return loss

    def _kl_dirichlet(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        KL divergence: Dir(alpha) || Dir(1) (uniform prior).

        KL = lgamma(sum alpha) - sum(lgamma(alpha)) - lgamma(C)
             + sum((alpha - 1) * (digamma(alpha) - digamma(sum alpha)))
        """
        C = alpha.size(-1)
        S = alpha.sum(dim=-1, keepdim=True)  # [B, 1]

        kl = (
            torch.lgamma(S.squeeze(-1))
            - torch.lgamma(alpha).sum(dim=-1)
            - torch.lgamma(torch.tensor(float(C), device=alpha.device))
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1)
        )
        return kl


# ============================================================
# Composite Loss Builder
# ============================================================

def build_loss_function(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build the loss function from config.

    Args:
        config: Training loss config dict
        class_weights: [n_classes] tensor of class weights (optional)

    Returns:
        Loss module
    """
    loss_cfg = config["training"]["loss"]
    loss_type = loss_cfg["type"]
    num_classes = config["data"]["num_classes"]

    if loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )

    elif loss_type == "focal":
        alpha = class_weights if class_weights is not None else None
        loss_fn = FocalLoss(
            num_classes=num_classes,
            gamma=loss_cfg["focal_gamma"],
            alpha=alpha,
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )

    elif loss_type == "focal_cross_entropy":
        # Weighted sum of focal + CE
        alpha = class_weights if class_weights is not None else None
        focal = FocalLoss(
            num_classes=num_classes,
            gamma=loss_cfg["focal_gamma"],
            alpha=alpha,
        )
        ce = LabelSmoothingCrossEntropy(
            smoothing=loss_cfg.get("label_smoothing", 0.1)
        )

        class _ComboLoss(nn.Module):
            def forward(self, logits, targets):
                return 0.5 * focal(logits, targets) + 0.5 * ce(logits, targets)

        loss_fn = _ComboLoss()

    elif loss_type == "label_smoothing":
        loss_fn = LabelSmoothingCrossEntropy(
            smoothing=loss_cfg.get("label_smoothing", 0.1)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    logger.info(f"[Loss] Using: {loss_type} | classes={num_classes}")
    return loss_fn
