"""
MSNet — Uncertainty-Aware Classifier Head
===========================================
Implements:
  1. Standard classification head (binary/multi-class)
  2. Monte Carlo Dropout for epistemic uncertainty estimation
  3. Temperature scaling for calibration
  4. Evidential Deep Learning (alternative uncertainty method)

Uncertainty estimation:
  Epistemic uncertainty reflects model uncertainty (data scarcity).
  Aleatoric uncertainty reflects inherent data noise/ambiguity.

  MC Dropout: Run N forward passes with dropout ACTIVE during inference.
  Variance across passes = epistemic uncertainty estimate.

Reference:
  Gal & Ghahramani (2016) — "Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning", ICML 2016.

  Sensoy et al. (2018) — "Evidential Deep Learning to Quantify
  Classification Uncertainty", NeurIPS 2018.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Standard Classifier Head
# ============================================================

class ClassifierHead(nn.Module):
    """
    Standard MLP classification head.

    Includes:
      - Multi-layer hidden projection
      - Label smoothing support (via training loss)
      - Temperature scaling for probability calibration
      - MC Dropout for uncertainty estimation at inference

    Args:
        in_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate (used for MC Dropout at inference)
        temperature: Temperature for softmax calibration (>1 = softer)
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 4,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.4,
        temperature: float = 1.0,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [512, 256]
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # Build hidden layers
        layers = []
        in_ch = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_ch, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),  # MC Dropout: active in training AND test
            ])
            in_ch = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x: [B, in_dim] — fused embedding
            return_logits: If True returns raw logits; else probabilities

        Returns:
            [B, num_classes] — logits or softmax probabilities
        """
        h = self.hidden(x)
        logits = self.output(h)

        if not return_logits:
            # Temperature scaling before softmax
            return F.softmax(logits / self.temperature, dim=-1)
        return logits


# ============================================================
# MC Dropout Uncertainty Head
# ============================================================

class MCDropoutUncertaintyHead(nn.Module):
    """
    Uncertainty estimation via Monte Carlo Dropout.

    Runs N stochastic forward passes with dropout ACTIVE,
    then computes:
      - Mean prediction (expected softmax)
      - Epistemic uncertainty (variance over MC samples)
      - Predictive entropy (total uncertainty)

    This is theoretically equivalent to approximate Bayesian inference
    over the network weights (Gal & Ghahramani, 2016).

    Args:
        classifier_head: Trained ClassifierHead module
        n_mc_samples: Number of Monte Carlo forward passes
        num_classes: Number of output classes
    """

    def __init__(
        self,
        classifier_head: ClassifierHead,
        n_mc_samples: int = 20,
        num_classes: int = 4,
    ):
        super().__init__()
        self.head = classifier_head
        self.n_mc_samples = n_mc_samples
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run MC Dropout inference.

        Args:
            x: [B, in_dim] — fused embedding

        Returns:
            Dict with:
              - 'mean_probs': [B, n_classes] — mean predictive probability
              - 'epistemic_uncertainty': [B] — variance (epistemic)
              - 'predictive_entropy': [B] — entropy of mean prediction
              - 'logits': [B, n_classes] — logits from mean pass
              - 'mc_probs': [B, n_samples, n_classes] — all MC samples
        """
        # Force dropout to be ACTIVE during inference
        self.head.train()

        mc_probs = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                logits = self.head(x, return_logits=True)
                probs = F.softmax(logits / self.head.temperature, dim=-1)
                mc_probs.append(probs.unsqueeze(1))  # [B, 1, C]

        mc_probs = torch.cat(mc_probs, dim=1)  # [B, n_samples, C]

        # Mean predictive probability
        mean_probs = mc_probs.mean(dim=1)  # [B, C]

        # Variance across MC samples (epistemic uncertainty)
        variance = mc_probs.var(dim=1)  # [B, C]
        epistemic = variance.mean(dim=-1)  # [B] — scalar per sample

        # Predictive entropy (total uncertainty = aleatoric + epistemic)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)

        # Deterministic logits (from mean probs)
        logits = torch.log(mean_probs + 1e-8)

        self.head.eval()

        return {
            "mean_probs": mean_probs,             # [B, C]
            "epistemic_uncertainty": epistemic,    # [B]
            "predictive_entropy": entropy,         # [B]
            "logits": logits,                      # [B, C]
            "mc_probs": mc_probs,                  # [B, n_samples, C]
        }


# ============================================================
# Evidential Deep Learning Head (Optional Alternative)
# ============================================================

class EvidentialClassifierHead(nn.Module):
    """
    Evidential Deep Learning (EDL) uncertainty estimation.

    Instead of outputting class logits, outputs Dirichlet distribution
    parameters (alpha > 1). The uncertainty comes from the distribution
    of distributions — high uncertainty when evidence is low.

    Advantages over MC Dropout:
      - Single forward pass (fast)
      - Explicit uncertainty decomposition

    Reference:
      Sensoy et al. (2018) — NeurIPS

    Args:
        in_dim: Input embedding dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),  # Ensure alpha > 0
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evidential forward pass.

        Args:
            x: [B, in_dim]

        Returns:
            Dict with:
              - 'alpha': [B, C] — Dirichlet parameters
              - 'mean_probs': [B, C] — expected class probabilities
              - 'uncertainty': [B] — total uncertainty (K / sum(alpha))
              - 'logits': [B, C] — log of expected probs
        """
        # Dirichlet parameters: alpha = evidence + 1
        evidence = self.head(x)   # [B, C], all > 0
        alpha = evidence + 1.0    # [B, C]

        # Total Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)  # [B, 1]

        # Expected class probabilities
        mean_probs = alpha / S  # [B, C]

        # Total uncertainty: K/S (high when S is small = low evidence)
        uncertainty = self.num_classes / S.squeeze(-1)  # [B]

        return {
            "alpha": alpha,
            "mean_probs": mean_probs,
            "uncertainty": uncertainty,
            "logits": torch.log(mean_probs + 1e-8),
        }
