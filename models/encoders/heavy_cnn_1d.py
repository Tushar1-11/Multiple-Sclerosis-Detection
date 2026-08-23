"""
MSNet — Heavy 1D-CNN Encoder with Dilated Bottleneck ResNet
============================================================
A dramatically deeper and more powerful 1D-CNN encoder designed for
maximum representational capacity on clinical MS tabular biomarker data.

Key Design Elements:
  - Feature Projection Stem (larger seq_len=32)
  - 4-Stage Bottleneck ResNet (128 → 256 → 512 → 512 channels)
  - Dilated Convolutions in stages 3 & 4 for long-range feature interactions
  - Dual Squeeze-and-Excitation per block (local + global recalibration)
  - Multi-Scale Feature Fusion (concat all stage outputs)
  - Stochastic Depth (DropPath) regularization
  - Dual Adaptive Pooling (AvgPool1d + MaxPool1d) aggregation
  - Final MLP projection to embed_dim
"""

import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Stochastic Depth (DropPath)
# ============================================================

class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drops the entire path (residual branch)
    during training. Improves regularization for deep networks.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # [B, 1, 1] — broadcast over C and L
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ============================================================
# 1D Squeeze-and-Excitation (SE) Block
# ============================================================

class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation block for channel-wise feature recalibration.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_channels = max(channels // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] → [B, C, L] scaled by channel attention weights"""
        scale = self.se(x).unsqueeze(-1)  # [B, C, 1]
        return x * scale


# ============================================================
# 1D Bottleneck Residual Block (ResNet-50 style)
# ============================================================

class BottleneckResBlock1D(nn.Module):
    """
    1D Bottleneck Residual Block:
      1×1 Conv (reduce) → 3×1 Conv (dilation) → 1×1 Conv (expand)
    With GroupNorm, SiLU activation, SE attention, and DropPath.
    """

    expansion: int = 2  # expand bottleneck output by this factor

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        drop_path_rate: float = 0.0,
        reduction: int = 8,
    ):
        super().__init__()
        padding = dilation  # keep same length with dilation

        def _norm(c: int) -> nn.GroupNorm:
            num_groups = min(8, c) if c >= 8 else 1
            return nn.GroupNorm(num_groups, c)

        # 1×1 bottleneck reduce
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = _norm(mid_channels)

        # 3×1 depthwise-style conv with optional dilation
        self.conv2 = nn.Conv1d(
            mid_channels, mid_channels, kernel_size=3,
            stride=stride, padding=padding, dilation=dilation,
            groups=1, bias=False
        )
        self.bn2 = _norm(mid_channels)

        # 1×1 bottleneck expand
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = _norm(out_channels)

        self.act = nn.SiLU(inplace=True)
        self.se = SEBlock1D(out_channels, reduction=reduction)
        self.drop_path = DropPath(drop_path_rate)

        # Residual shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.drop_path(out)

        return self.act(out + residual)


# ============================================================
# Multi-Scale Feature Fusion Module
# ============================================================

class MultiScaleFusion(nn.Module):
    """
    Fuses feature maps from multiple CNN stages by upsampling all to
    the same length and concatenating channel-wise.
    """

    def __init__(self, stage_channels: List[int], fused_dim: int):
        super().__init__()
        total_channels = sum(stage_channels)
        self.proj = nn.Sequential(
            nn.Conv1d(total_channels, fused_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, fused_dim), fused_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        feature_maps: List of [B, C_i, L_i] tensors from each stage
        Returns: [B, fused_dim, L_max]
        """
        target_len = feature_maps[0].shape[-1]  # Use first stage's (largest) length
        aligned = []
        for f in feature_maps:
            if f.shape[-1] != target_len:
                f = F.interpolate(f, size=target_len, mode="linear", align_corners=False)
            aligned.append(f)
        concat = torch.cat(aligned, dim=1)  # [B, sum(C_i), target_len]
        return self.proj(concat)


# ============================================================
# Heavy Tabular 1D-CNN Encoder
# ============================================================

class HeavyTabular1DCNNEncoder(nn.Module):
    """
    Heavy 4-stage 1D-CNN Encoder with:
      - Bottleneck ResNet blocks (ResNet-50 style)
      - Dilated convolutions in deep stages
      - Multi-scale feature fusion
      - Stochastic Depth regularization
      - Multi-Head Self-Attention
      - Dual Global Pooling
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int = 512,
        channels: Tuple[int, int, int, int] = (128, 256, 512, 512),
        seq_len: int = 32,
        num_heads: int = 8,
        dropout: float = 0.2,
        drop_path_rate: float = 0.15,
        blocks_per_stage: Tuple[int, int, int, int] = (2, 3, 4, 2),
    ):
        """
        Args:
            in_features: Number of input tabular features
            embed_dim: Output embedding dimension
            channels: Output channels for each of the 4 stages
            seq_len: Sequence length after projection stem
            num_heads: Number of attention heads in self-attention
            dropout: Dropout probability
            drop_path_rate: Maximum stochastic depth drop rate
            blocks_per_stage: Number of bottleneck blocks per stage
        """
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        c1, c2, c3, c4 = channels
        total_blocks = sum(blocks_per_stage)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        block_idx = 0

        # ── Stem Projection ──────────────────────────────────────────
        # Projects flat vector [B, in_features] → [B, c1, seq_len]
        self.stem = nn.Sequential(
            nn.Linear(in_features, c1 * seq_len),
            nn.LayerNorm(c1 * seq_len),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Stage 1: [B, c1, seq_len] — no downsampling ──────────────
        s1_blocks = []
        for i in range(blocks_per_stage[0]):
            s1_blocks.append(BottleneckResBlock1D(
                in_channels=c1, mid_channels=c1 // 2, out_channels=c1,
                stride=1, dilation=1,
                drop_path_rate=dp_rates[block_idx],
            ))
            block_idx += 1
        self.stage1 = nn.Sequential(*s1_blocks)

        # ── Stage 2: [B, c2, seq_len/2] — stride=2 ──────────────────
        s2_blocks = []
        for i in range(blocks_per_stage[1]):
            s2_blocks.append(BottleneckResBlock1D(
                in_channels=c1 if i == 0 else c2,
                mid_channels=c2 // 2,
                out_channels=c2,
                stride=2 if i == 0 else 1,
                dilation=1,
                drop_path_rate=dp_rates[block_idx],
            ))
            block_idx += 1
        self.stage2 = nn.Sequential(*s2_blocks)

        # ── Stage 3: [B, c3, seq_len/4] — stride=2, dilation=2 ─────
        s3_blocks = []
        for i in range(blocks_per_stage[2]):
            s3_blocks.append(BottleneckResBlock1D(
                in_channels=c2 if i == 0 else c3,
                mid_channels=c3 // 2,
                out_channels=c3,
                stride=2 if i == 0 else 1,
                dilation=2,
                drop_path_rate=dp_rates[block_idx],
            ))
            block_idx += 1
        self.stage3 = nn.Sequential(*s3_blocks)

        # ── Stage 4: [B, c4, seq_len/4] — stride=1, dilation=3 ─────
        s4_blocks = []
        for i in range(blocks_per_stage[3]):
            s4_blocks.append(BottleneckResBlock1D(
                in_channels=c3 if i == 0 else c4,
                mid_channels=c4 // 2,
                out_channels=c4,
                stride=1,
                dilation=3,
                drop_path_rate=dp_rates[block_idx],
            ))
            block_idx += 1
        self.stage4 = nn.Sequential(*s4_blocks)

        # ── Multi-Scale Feature Fusion ────────────────────────────────
        fused_dim = c4  # fuse all into c4
        self.fusion = MultiScaleFusion(
            stage_channels=[c1, c2, c3, c4],
            fused_dim=fused_dim,
        )

        # ── Multi-Head Self-Attention over sequence ───────────────────
        # Ensure embed_dim is divisible by num_heads
        attn_dim = fused_dim
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # Feed-Forward after attention
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )
        self.ff_norm = nn.LayerNorm(attn_dim)

        # ── Dual Adaptive Pooling ─────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # ── Final Projection to embed_dim ─────────────────────────────
        pooled_dim = attn_dim * 2  # avg + max concatenated
        self.fc_out = nn.Sequential(
            nn.Linear(pooled_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] — preprocessed tabular feature vector

        Returns:
            embedding: [B, embed_dim] — rich 1D-CNN feature representation
        """
        B = x.shape[0]

        # Stem → 1D sequence
        h = self.stem(x).view(B, -1, self.seq_len)  # [B, c1, seq_len]

        # 4 ResNet stages — collect all outputs for multi-scale fusion
        h1 = self.stage1(h)   # [B, c1, seq_len]
        h2 = self.stage2(h1)  # [B, c2, seq_len/2]
        h3 = self.stage3(h2)  # [B, c3, seq_len/4]
        h4 = self.stage4(h3)  # [B, c4, seq_len/4]

        # Multi-Scale Fusion
        fused = self.fusion([h1, h2, h3, h4])  # [B, fused_dim, seq_len]

        # Multi-Head Self-Attention (Pre-LN transformer style)
        h_perm = fused.permute(0, 2, 1)  # [B, L, C]
        attn_out, _ = self.self_attn(h_perm, h_perm, h_perm)
        h_attn = self.attn_norm(h_perm + attn_out)  # residual + norm

        # Feed-Forward
        ff_out = self.ff(h_attn)
        h_ff = self.ff_norm(h_attn + ff_out)  # [B, L, C]

        h_out = h_ff.permute(0, 2, 1)  # [B, C, L]

        # Dual Pooling
        avg_feats = self.avg_pool(h_out).flatten(1)  # [B, C]
        max_feats = self.max_pool(h_out).flatten(1)  # [B, C]
        concat_feats = torch.cat([avg_feats, max_feats], dim=1)  # [B, C*2]

        # Final projection
        out = self.fc_out(concat_feats)  # [B, embed_dim]
        return out
