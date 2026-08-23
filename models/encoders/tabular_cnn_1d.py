"""
MSNet — 1D-CNN ResNet + SE Attention + Self-Attention Encoder
==============================================================
Specialized 1D Convolutional Neural Network architecture designed for high-dimensional
tabular clinical & brain volumetric biomarkers.

Key Design Elements:
  - Feature Projection Stem: Transforms 1D feature vectors into channelized 1D sequence representations [B, C, L]
  - Multi-stage 1D Bottleneck ResNet Blocks (Conv1D + GroupNorm + GELU)
  - 1D Squeeze-and-Excitation (SE1D) Channel Attention for feature recalibration
  - 1D Multi-Head Self-Attention for capturing non-local cross-feature dependencies
  - Dual Adaptive Pooling (AvgPool1d + MaxPool1d) for comprehensive feature aggregation
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# 1D Squeeze-and-Excitation (SE) Block
# ============================================================

class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation block for channel-wise feature recalibration.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        Returns: [B, C, L] scaled by channel weights
        """
        scale = self.fc(x)  # [B, C]
        scale = scale.unsqueeze(-1)  # [B, C, 1]
        return x * scale


# ============================================================
# 1D Residual Convolutional Block
# ============================================================

class ResBlock1D(nn.Module):
    """
    1D Residual Block with GroupNorm, GELU activation, and SE Attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        reduction: int = 16,
    ):
        super().__init__()
        padding = kernel_size // 2

        num_groups = min(8, in_channels)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )

        num_groups_out = min(8, out_channels)
        self.gn2 = nn.GroupNorm(num_groups_out, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )

        self.act = nn.GELU()
        self.se = SEBlock1D(out_channels, reduction=reduction)

        # Residual shortcut matching
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups_out, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.act(self.gn1(x))
        out = self.conv1(out)
        out = self.act(self.gn2(out))
        out = self.conv2(out)
        out = self.se(out)

        return out + residual


# ============================================================
# Tabular 1D-CNN Encoder Engine
# ============================================================

class Tabular1DCNNEncoder(nn.Module):
    """
    1D-CNN Encoder with ResNet backbone, Squeeze-and-Excitation, and Self-Attention.
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int = 256,
        channels: Tuple[int, int, int] = (64, 128, 256),
        seq_len: int = 16,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_features: Number of input tabular features
            embed_dim: Output embedding dimension
            channels: Channels for the 3 ResNet stages
            seq_len: Sequence length after projection stem
            num_heads: Number of attention heads for multi-head self-attention
            dropout: Dropout probability
        """
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        c1, c2, c3 = channels

        # Stem: Project flat tabular vector [B, in_features] to 1D channel sequence [B, c1, seq_len]
        self.stem_proj = nn.Sequential(
            nn.Linear(in_features, c1 * seq_len),
            nn.LayerNorm(c1 * seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stage 1: [B, c1, seq_len]
        self.stage1 = nn.Sequential(
            ResBlock1D(c1, c1, stride=1),
            ResBlock1D(c1, c1, stride=1),
        )

        # Stage 2: [B, c2, seq_len / 2]
        self.stage2 = nn.Sequential(
            ResBlock1D(c1, c2, stride=2),
            ResBlock1D(c2, c2, stride=1),
        )

        # Stage 3: [B, c3, seq_len / 4]
        self.stage3 = nn.Sequential(
            ResBlock1D(c2, c3, stride=2),
            ResBlock1D(c3, c3, stride=1),
        )

        # Self-Attention Layer (over sequence dimension)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=c3, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(c3)

        # Dual Adaptive Pooling (Avg + Max)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Final projection layer to embed_dim
        pooled_dim = c3 * 2
        self.fc_out = nn.Sequential(
            nn.Linear(pooled_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] — preprocessed tabular features

        Returns:
            embedding: [B, embed_dim] — dense 1D-CNN feature representation
        """
        batch_size = x.shape[0]

        # Stem projection -> [B, c1, seq_len]
        stem = self.stem_proj(x)
        h = stem.view(batch_size, -1, self.seq_len)

        # ResNet stages
        h = self.stage1(h)  # [B, c1, seq_len]
        h = self.stage2(h)  # [B, c2, seq_len / 2]
        h = self.stage3(h)  # [B, c3, seq_len / 4]

        # Multi-Head Self-Attention
        # Rearrange to [B, L, C] for nn.MultiheadAttention
        h_perm = h.permute(0, 2, 1)  # [B, L_curr, c3]
        attn_out, _ = self.self_attn(h_perm, h_perm, h_perm)
        h_attn = self.attn_norm(h_perm + attn_out).permute(0, 2, 1)  # Back to [B, c3, L_curr]

        # Dual Pooling
        avg_feats = self.avg_pool(h_attn).flatten(1)  # [B, c3]
        max_feats = self.max_pool(h_attn).flatten(1)  # [B, c3]
        concat_feats = torch.cat([avg_feats, max_feats], dim=1)  # [B, c3 * 2]

        # Project to embed_dim
        out = self.fc_out(concat_feats)  # [B, embed_dim]
        return out
