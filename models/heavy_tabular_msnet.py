"""
MSNet — Heavy Tabular MSNet with Multi-Task Heads
==================================================
Full model wrapper combining the Heavy 1D-CNN Encoder with:
  - 3-class Phenotype classification head (HC / RRMS / Progressive MS)
  - 2-class Group classification head (HC vs MS Patient)
  - EDSS regression head (0.0 – 10.0 continuous score)
  - MC Dropout uncertainty estimation

Architecture:
  HeavyTabular1DCNNEncoder → shared embedding
      ├── PhenotypeHead   → [B, 3]   logits
      ├── GroupHead       → [B, 2]   logits
      └── EDSSHead        → [B, 1]   regression score
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.heavy_cnn_1d import HeavyTabular1DCNNEncoder

logger = logging.getLogger(__name__)


# ============================================================
# Multi-Task Classifier Head
# ============================================================

class ClassificationHead(nn.Module):
    """Dense classification head with residual bottleneck."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        mid = embed_dim // 2
        self.block = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.LayerNorm(mid),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2),
            nn.LayerNorm(mid // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# EDSS Regression Head
# ============================================================

class EDSSRegressionHead(nn.Module):
    """Dense regression head for EDSS score prediction (0.0–10.0)."""

    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        mid = embed_dim // 2
        self.block = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.LayerNorm(mid),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 4),
            nn.LayerNorm(mid // 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid // 4, 1),
            nn.Sigmoid(),  # output in [0, 1], scale by 10 in post-processing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) * 10.0  # Scale to [0, 10]


# ============================================================
# Heavy Tabular MSNet — Main Model
# ============================================================

class HeavyTabularMSNet(nn.Module):
    """
    End-to-End Heavy 1D-CNN Multi-Task Model for MS Diagnosis, Subtyping & EDSS Prediction.

    Multi-task outputs:
      - phenotype_logits: [B, 3] — HC / RRMS / Progressive MS
      - group_logits:     [B, 2] — HC vs MS Patient
      - edss_pred:        [B, 1] — EDSS score (0–10)
    """

    def __init__(
        self,
        in_features: int,
        num_phenotype_classes: int = 3,
        num_group_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3,
        num_heads: int = 8,
        drop_path_rate: float = 0.15,
        channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
        seq_len: int = 32,
        blocks_per_stage: Tuple[int, int, int, int] = (2, 3, 4, 2),
    ):
        """
        Args:
            in_features: Number of preprocessed input features
            num_phenotype_classes: Number of phenotype classes (default 3)
            num_group_classes: Number of group classes (default 2)
            embed_dim: Shared embedding dimension from encoder
            dropout: Dropout probability for heads
            num_heads: Attention heads in encoder
            drop_path_rate: Stochastic depth max drop rate
            channels: Channels per stage in encoder
            seq_len: Sequence length in encoder stem
            blocks_per_stage: Number of bottleneck blocks per stage
        """
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.num_phenotype_classes = num_phenotype_classes
        self.num_group_classes = num_group_classes

        # Shared Heavy 1D-CNN Encoder (backbone)
        self.encoder = HeavyTabular1DCNNEncoder(
            in_features=in_features,
            embed_dim=embed_dim,
            channels=channels,
            seq_len=seq_len,
            num_heads=num_heads,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            blocks_per_stage=blocks_per_stage,
        )

        # Task-specific heads
        self.phenotype_head = ClassificationHead(embed_dim, num_phenotype_classes, dropout=dropout)
        self.group_head = ClassificationHead(embed_dim, num_group_classes, dropout=dropout)
        self.edss_head = EDSSRegressionHead(embed_dim, dropout=dropout)

        # Initialize weights
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"HeavyTabularMSNet initialized | Params: {total_params:,} | embed_dim: {embed_dim}")

    def _init_weights(self):
        """Kaiming initialization for Conv1d, Xavier for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        task: str = "phenotype",
        return_features: bool = False,
        return_all: bool = False,
    ) -> Any:
        """
        Args:
            x: [B, in_features] — preprocessed tabular input
            task: 'phenotype', 'group', or 'edss'
            return_features: If True, also return the encoder embedding
            return_all: If True, return all task outputs as a dict

        Returns:
            If return_all=True: dict with 'phenotype', 'group', 'edss', 'features'
            If return_features=True: (logits, features)
            Else: logits (or edss scalar)
        """
        feats = self.encoder(x)  # [B, embed_dim]

        if return_all:
            return {
                "phenotype": self.phenotype_head(feats),   # [B, 3]
                "group": self.group_head(feats),            # [B, 2]
                "edss": self.edss_head(feats),              # [B, 1]
                "features": feats,                          # [B, embed_dim]
            }

        if task == "phenotype":
            logits = self.phenotype_head(feats)
        elif task == "group":
            logits = self.group_head(feats)
        elif task == "edss":
            logits = self.edss_head(feats)
        else:
            raise ValueError(f"Unknown task: '{task}'. Choose 'phenotype', 'group', or 'edss'.")

        if return_features:
            return logits, feats
        return logits

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        task: str = "phenotype",
        num_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference for epistemic uncertainty estimation.

        Args:
            x: [B, in_features]
            task: 'phenotype' or 'group'
            num_samples: Number of stochastic forward passes

        Returns:
            mean_probs:         [B, num_classes]
            predictive_entropy: [B]       — epistemic uncertainty
            std_probs:          [B, num_classes]
        """
        self.train()  # Enable dropout during inference

        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, task=task)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.unsqueeze(0))

        probs_tensor = torch.cat(probs_list, dim=0)     # [S, B, C]
        mean_probs = probs_tensor.mean(dim=0)            # [B, C]
        std_probs = probs_tensor.std(dim=0)              # [B, C]

        eps = 1e-8
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + eps), dim=-1
        )  # [B]

        self.eval()
        return mean_probs, predictive_entropy, std_probs
