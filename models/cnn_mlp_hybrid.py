"""
MSNet -- CNN + MLP Hybrid Model
=================================
Architecture:
  Heavy 1D-CNN backbone (feature extractor)
        |
        v
  [B, embed_dim]  -- frozen or fine-tunable CNN features
        |
        v
  Deep Residual MLP (classifier)
    - Multiple dense residual blocks
    - BatchNorm + Dropout + GELU
    - Skip connections between MLP layers
    - Squeeze-and-Excitation in dense space
        |
        v
  Multi-task heads: Phenotype / Group / EDSS

Key philosophy:
  CNN  = extract powerful spatial/sequential feature representations from
         tabular data treated as a 1D sequence.
  MLP  = deep non-linear decision boundary learner on CNN embeddings,
         with residual connections to avoid gradient vanishing in deep layers.
"""

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.heavy_cnn_1d import HeavyTabular1DCNNEncoder

logger = logging.getLogger(__name__)


# ============================================================
# Dense Squeeze-and-Excitation (Feature Recalibration in MLP space)
# ============================================================

class DenseSEBlock(nn.Module):
    """
    Squeeze-and-Excitation adapted for 1D dense (flat) feature vectors.
    Learns to reweight individual feature dimensions.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ============================================================
# Residual MLP Block
# ============================================================

class ResMLPBlock(nn.Module):
    """
    Residual dense block for MLP:
      Linear -> BN -> GELU -> Dropout -> Linear -> BN -> SE -> + residual

    Uses a projection shortcut when in_dim != out_dim.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        expansion: float = 2.0,
    ):
        super().__init__()
        mid_dim = int(out_dim * expansion)

        self.block = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

        self.se = DenseSEBlock(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout * 0.5)

        # Projection shortcut for dimension change
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.block(x)
        out = self.se(out)
        out = self.dropout(out)
        return self.act(out + residual)


# ============================================================
# Deep Residual MLP Classifier
# ============================================================

class DeepResidualMLP(nn.Module):
    """
    Deep Residual MLP for classification/regression on CNN embeddings.

    Architecture:
      embed_dim -> [layer_dims[0]] -> ... -> [layer_dims[-1]] -> num_classes

    Each transition uses a ResMLPBlock with projection shortcut.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        layer_dims: List[int] = (512, 512, 256, 256, 128),
        dropout: float = 0.35,
        expansion: float = 2.0,
    ):
        super().__init__()

        layers = []
        current_dim = in_dim
        for out_dim in layer_dims:
            layers.append(ResMLPBlock(current_dim, out_dim, dropout=dropout, expansion=expansion))
            current_dim = out_dim

        self.mlp_tower = nn.Sequential(*layers)

        # Final classification head (no activation, raw logits)
        self.head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(current_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.mlp_tower(x)
        return self.head(feats)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate MLP features before the classification head."""
        return self.mlp_tower(x)


# ============================================================
# Deep Residual MLP Regressor (for EDSS)
# ============================================================

class DeepResidualMLPRegressor(nn.Module):
    """Deep Residual MLP for EDSS regression (0.0 - 10.0)."""

    def __init__(
        self,
        in_dim: int,
        layer_dims: List[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        current_dim = in_dim
        for out_dim in layer_dims:
            layers.append(ResMLPBlock(current_dim, out_dim, dropout=dropout))
            current_dim = out_dim

        self.mlp_tower = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(current_dim, 1),
            nn.Sigmoid(),  # output [0, 1] -> scaled to [0, 10]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp_tower(x)) * 10.0


# ============================================================
# CNN + MLP Hybrid Model  (main class)
# ============================================================

class CNNMLPHybrid(nn.Module):
    """
    CNN + MLP Hybrid Model for MS Diagnosis, Subtyping & EDSS Prediction.

    Stage 1 -- CNN Encoder (Heavy 1D-CNN):
      Extracts rich spatial/sequence features from tabular MS biomarkers.
      Produces a fixed-size embedding vector of shape [B, embed_dim].

    Stage 2 -- Residual MLP Tower:
      Deep non-linear decision boundary learner on top of CNN features.
      Uses residual connections, BatchNorm, GELU, Dense-SE attention.

    Stage 3 -- Task-specific heads:
      - Phenotype: 3-class (HC / RRMS / Progressive MS)
      - Group:     2-class (HC vs MS Patient)
      - EDSS:      Regression head (0.0 - 10.0)

    The CNN backbone can be:
      a) Trained end-to-end (default, best accuracy)
      b) Frozen (fast fine-tuning of MLP only)
    """

    def __init__(
        self,
        in_features: int,
        # CNN encoder config
        cnn_embed_dim: int = 512,
        cnn_channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
        cnn_seq_len: int = 32,
        cnn_num_heads: int = 8,
        cnn_drop_path_rate: float = 0.15,
        cnn_blocks_per_stage: Tuple[int, int, int, int] = (2, 3, 4, 2),
        # MLP classifier config
        mlp_layer_dims: Tuple[int, ...] = (512, 512, 256, 256, 128),
        mlp_dropout: float = 0.35,
        mlp_expansion: float = 2.0,
        # Global config
        shared_dropout: float = 0.3,
        num_phenotype_classes: int = 3,
        num_group_classes: int = 2,
        freeze_cnn: bool = False,
    ):
        """
        Args:
            in_features: Number of preprocessed input features
            cnn_embed_dim: CNN backbone output embedding dimension
            cnn_channels: Channel widths for 4 CNN stages
            cnn_seq_len: Stem sequence length in CNN encoder
            cnn_num_heads: Self-attention heads in CNN
            cnn_drop_path_rate: Stochastic depth rate for CNN
            cnn_blocks_per_stage: Bottleneck blocks per CNN stage
            mlp_layer_dims: Hidden layer widths for the Residual MLP tower
            mlp_dropout: Dropout in MLP blocks
            mlp_expansion: Expansion ratio in ResMLPBlock
            shared_dropout: Dropout applied between CNN and MLP
            num_phenotype_classes: Output classes for phenotype head
            num_group_classes: Output classes for group head
            freeze_cnn: If True, CNN backbone weights are frozen
        """
        super().__init__()
        self.in_features = in_features
        self.cnn_embed_dim = cnn_embed_dim
        self.freeze_cnn = freeze_cnn

        # ── Stage 1: CNN Encoder (Feature Extractor) ─────────────────
        self.cnn_encoder = HeavyTabular1DCNNEncoder(
            in_features=in_features,
            embed_dim=cnn_embed_dim,
            channels=cnn_channels,
            seq_len=cnn_seq_len,
            num_heads=cnn_num_heads,
            dropout=shared_dropout,
            drop_path_rate=cnn_drop_path_rate,
            blocks_per_stage=cnn_blocks_per_stage,
        )

        if freeze_cnn:
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False
            logger.info("CNN encoder FROZEN - only MLP will be trained")

        # Bridge: optional dropout between CNN output and MLP input
        self.bridge_dropout = nn.Dropout(shared_dropout)

        # ── Stage 2: Deep Residual MLP Tower (Classifier) ────────────
        self.phenotype_mlp = DeepResidualMLP(
            in_dim=cnn_embed_dim,
            num_classes=num_phenotype_classes,
            layer_dims=list(mlp_layer_dims),
            dropout=mlp_dropout,
            expansion=mlp_expansion,
        )

        self.group_mlp = DeepResidualMLP(
            in_dim=cnn_embed_dim,
            num_classes=num_group_classes,
            layer_dims=list(mlp_layer_dims),
            dropout=mlp_dropout,
            expansion=mlp_expansion,
        )

        # ── Stage 3: EDSS Regression MLP ─────────────────────────────
        self.edss_mlp = DeepResidualMLPRegressor(
            in_dim=cnn_embed_dim,
            layer_dims=[256, 128, 64],
            dropout=mlp_dropout,
        )

        # Weight initialization
        self._init_mlp_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        mlp_params = total_params - cnn_params
        logger.info(
            f"CNNMLPHybrid | Total: {total_params:,} params "
            f"(CNN: {cnn_params:,} | MLP: {mlp_params:,})"
        )

    def _init_mlp_weights(self):
        """Xavier uniform for all Linear layers in MLP tower."""
        for m in [self.phenotype_mlp, self.group_mlp, self.edss_mlp]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        task: str = "phenotype",
        return_features: bool = False,
        return_all: bool = False,
    ) -> Any:
        """
        Args:
            x: [B, in_features]
            task: 'phenotype', 'group', or 'edss'
            return_features: If True, also return CNN embedding
            return_all: If True, return all task outputs as dict

        Returns:
            If return_all: dict('phenotype', 'group', 'edss', 'features')
            If return_features: (logits, cnn_feats)
            Else: logits (or edss scalar for task='edss')
        """
        # Stage 1: CNN feature extraction
        cnn_feats = self.cnn_encoder(x)          # [B, cnn_embed_dim]
        cnn_feats = self.bridge_dropout(cnn_feats)

        if return_all:
            return {
                "phenotype": self.phenotype_mlp(cnn_feats),
                "group":     self.group_mlp(cnn_feats),
                "edss":      self.edss_mlp(cnn_feats),
                "features":  cnn_feats,
            }

        # Stage 2: Deep Residual MLP
        if task == "phenotype":
            logits = self.phenotype_mlp(cnn_feats)
        elif task == "group":
            logits = self.group_mlp(cnn_feats)
        elif task == "edss":
            logits = self.edss_mlp(cnn_feats)
        else:
            raise ValueError(f"Unknown task: '{task}'. Use 'phenotype', 'group', or 'edss'.")

        if return_features:
            return logits, cnn_feats
        return logits

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        task: str = "phenotype",
        num_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty estimation.

        Returns:
            mean_probs:         [B, num_classes]
            predictive_entropy: [B]
            std_probs:          [B, num_classes]
        """
        self.train()  # enable dropout

        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, task=task)
                probs_list.append(F.softmax(logits, dim=-1).unsqueeze(0))

        probs_tensor = torch.cat(probs_list, dim=0)       # [S, B, C]
        mean_probs = probs_tensor.mean(dim=0)              # [B, C]
        std_probs  = probs_tensor.std(dim=0)               # [B, C]

        eps = 1e-8
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)

        self.eval()
        return mean_probs, entropy, std_probs

    def unfreeze_cnn(self):
        """Unfreeze CNN encoder for end-to-end fine-tuning."""
        for param in self.cnn_encoder.parameters():
            param.requires_grad = True
        logger.info("CNN encoder UNFROZEN - full end-to-end training enabled")

    def freeze_cnn(self):
        """Freeze CNN encoder (train MLP only)."""
        for param in self.cnn_encoder.parameters():
            param.requires_grad = False
        logger.info("CNN encoder FROZEN")
