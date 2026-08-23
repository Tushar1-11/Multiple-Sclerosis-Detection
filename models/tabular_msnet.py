"""
MSNet — Tabular MSNet Architecture with 1D-CNN Encoder
======================================================
Complete Tabular Neural Network model combining 1D-CNN ResNet Encoder,
Squeeze-and-Excitation Attention, Self-Attention, and Classifier Head.

Supports:
  - 3-class Subtype classification (HC vs RRMS vs Progressive MS)
  - 2-class Binary classification (HC vs MS Patient)
  - MC Dropout for Monte Carlo Uncertainty estimation
"""

import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.tabular_cnn_1d import Tabular1DCNNEncoder

logger = logging.getLogger(__name__)


class TabularMSNet(nn.Module):
    """
    End-to-End Tabular 1D-CNN Model for Multiple Sclerosis Diagnosis & Subtyping.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        embed_dim: int = 256,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        """
        Args:
            in_features: Number of input tabular features
            num_classes: Number of target classes (3 for phenotype, 2 for group)
            embed_dim: Intermediate embedding dimension
            dropout: Dropout probability
            num_heads: Number of attention heads in self-attention
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1D-CNN Encoder
        self.encoder = Tabular1DCNNEncoder(
            in_features=in_features,
            embed_dim=embed_dim,
            channels=(64, 128, 256),
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # EDSS Score Regressor Head (Predicts EDSS 0.0 - 10.0)
        self.edss_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.ReLU(),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False, return_edss: bool = False
    ) -> Any:
        """
        Args:
            x: [B, in_features] — preprocessed tabular features
            return_features: If True, returns (logits, features)
            return_edss: If True, returns (logits, edss_pred)

        Returns:
            logits: [B, num_classes]
            edss_pred: [B, 1] (optional)
        """
        feats = self.encoder(x)
        logits = self.classifier(feats)

        if return_edss:
            edss_pred = self.edss_regressor(feats)
            return logits, edss_pred

        if return_features:
            return logits, feats
        return logits

    def predict_with_uncertainty(
        self, x: torch.Tensor, num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference to estimate predictive uncertainty.

        Returns:
            mean_probs: [B, num_classes] — mean class probabilities
            predictive_entropy: [B] — epistemic uncertainty measure
            std_probs: [B, num_classes] — standard deviation of predictions
        """
        self.train()  # Enable dropout during evaluation

        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.unsqueeze(0))

        probs_tensor = torch.cat(probs_list, dim=0)  # [num_samples, B, num_classes]
        mean_probs = probs_tensor.mean(dim=0)  # [B, num_classes]
        std_probs = probs_tensor.std(dim=0)    # [B, num_classes]

        # Epistemic uncertainty via predictive entropy: -sum(p * log(p))
        eps = 1e-8
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)

        self.eval()
        return mean_probs, predictive_entropy, std_probs
