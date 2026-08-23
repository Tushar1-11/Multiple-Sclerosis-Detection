"""
MSNet — Clinical Metadata MLP Encoder
=======================================
Encodes structured clinical features into a dense embedding vector
that can be fused with MRI image features via cross-modal attention.

Architecture:
  Input:  [B, n_features]   (normalized clinical feature vector)
  Output: [B, out_dim]      (clinical embedding)

Design:
  - Residual MLP blocks for deep feature transformation
  - LayerNorm + GELU for stable gradient flow
  - Attention-weighted feature gating (clinical feature importance)
  - Dropout for regularization

Clinical features encoded:
  age | gender | edss | disease_duration | igg_index |
  oligoclonal_bands | t_cell_count
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Residual MLP Block
# ============================================================

class ResMLPBlock(nn.Module):
    """
    Residual MLP block for clinical feature transformation.

    Structure: LayerNorm → Linear → GELU → Linear → Dropout → + residual

    Args:
        dim: Feature dimension (input = output)
        dropout: Dropout rate
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + residual


# ============================================================
# Feature Importance Gate
# ============================================================

class ClinicalFeatureGate(nn.Module):
    """
    Soft attention gate for clinical feature importance weighting.

    Learns a scalar importance weight for each input clinical feature.
    This provides interpretability: which clinical variables most
    influence the embedding (analogous to feature importance).

    Args:
        n_features: Number of clinical input features
    """

    def __init__(self, n_features: int):
        super().__init__()
        # Scalar attention weight per feature
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, n_features] — raw clinical features

        Returns:
            gated_x: [B, n_features] — feature-gated clinical vector
            weights: [B, n_features] — attention weights (for interpretation)
        """
        weights = self.gate(x)   # [B, n_features]
        gated   = x * weights    # Element-wise gate
        return gated, weights


# ============================================================
# Clinical Encoder
# ============================================================

class ClinicalEncoder(nn.Module):
    """
    Deep MLP encoder for structured clinical metadata.

    Transforms raw clinical feature vectors into rich embeddings
    that capture complex clinical phenotype patterns, enabling
    effective fusion with high-dimensional MRI features.

    Processing flow:
      1. Feature importance gating
      2. Input projection to hidden_dim
      3. Stacked Residual MLP blocks
      4. Output projection to out_dim
      5. L2 normalization of final embedding

    Args:
        input_dim: Number of clinical features (e.g., 7)
        hidden_dims: List of hidden layer dimensions
        out_dim: Output embedding dimension (should match MRI encoder)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: Optional[List[int]] = None,
        out_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        hidden_dims = hidden_dims or [64, 128, 64]
        self.out_dim = out_dim

        # 1. Feature importance gate
        self.gate = ClinicalFeatureGate(input_dim)

        # 2. Input embedding projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Intermediate projection layers
        mlp_layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 4. Per-dimension residual refinement blocks
        self.res_blocks = nn.Sequential(
            ResMLPBlock(in_dim, dropout),
            ResMLPBlock(in_dim, dropout),
        )

        # 5. Output projection to out_dim
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self._init_weights()
        logger.info(
            f"[ClinicalEncoder] input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, out_dim={out_dim}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, input_dim] — normalized clinical features

        Returns:
            embedding: [B, out_dim] — clinical feature embedding
            gate_weights: [B, input_dim] — per-feature importance weights
        """
        # 1. Soft feature importance gate
        x_gated, gate_weights = self.gate(x)

        # 2. Project to hidden_dim
        h = self.input_proj(x_gated)

        # 3. MLP layers
        h = self.mlp(h)

        # 4. Residual refinement
        h = self.res_blocks(h)

        # 5. Output projection
        embedding = self.out_proj(h)   # [B, out_dim]

        # L2 normalization for stable fusion with MRI features
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding, gate_weights
