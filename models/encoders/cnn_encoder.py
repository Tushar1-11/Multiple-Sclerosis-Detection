"""
MSNet — 3D CNN Encoder (Fallback / Ensemble Component)
========================================================
Implements a 3D ResNet-50-based CNN encoder as an alternative
or complement to the Swin Transformer encoder.

Architecture:
  Input:  [B, 1, D, H, W]   (single-channel MRI volume)
  Output: [B, out_dim]       (global feature vector)

Design choices:
  - 3D ResNet-50 with GroupNorm (works on ANY spatial size, unlike InstanceNorm
    which crashes when spatial dims collapse to 1x1x1 in deep stages)
  - Squeeze-and-Excitation blocks for channel recalibration
  - Global Average + Max Pooling for richer descriptors
  - Modality-specific stem convolutions

Reference:
  He et al. (2016) — "Deep Residual Learning for Image Recognition"
  Hu et al. (2018) — "Squeeze-and-Excitation Networks", CVPR 2018
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Squeeze-and-Excitation Block
# ============================================================

class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation block for channel-wise recalibration.

    SE blocks learn to selectively emphasize informative feature
    channels (e.g., lesion-sensitive feature maps in FLAIR)
    and suppress less useful ones, with negligible parameter overhead.

    Args:
        in_channels: Number of input channels
        reduction: SE reduction ratio (r=16 default)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale channels by SE weights."""
        scale = self.fc(x)                    # [B, C]
        scale = scale.view(*scale.shape, 1, 1, 1)  # [B, C, 1, 1, 1]
        return x * scale


# ============================================================
# 3D Residual Block
# ============================================================

class ResBlock3D(nn.Module):
    """
    3D Bottleneck Residual Block with SE attention.

    Structure:
      1x1x1 Conv → 3x3x3 Conv → 1x1x1 Conv → SE → + residual → GELU

    Instance Normalization is preferred over Batch Normalization for
    3D medical images since small batch sizes (2-4) make BN statistics
    unreliable.

    Args:
        in_channels: Input channels
        mid_channels: Bottleneck channels (typically in_channels // 4)
        out_channels: Output channels
        stride: Spatial stride (2 for downsampling)
        reduction: SE reduction ratio
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
    ):
        super().__init__()

        # GroupNorm works on ANY spatial size including 1x1x1.
        # InstanceNorm3d crashes when spatial dims collapse in deep stages.
        def gn(ch): return nn.GroupNorm(min(8, ch), ch)

        # Bottleneck: 1x1 -> 3x3 -> 1x1
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 1, bias=False)
        self.norm1 = gn(mid_channels)

        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.norm2 = gn(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, 1, bias=False)
        self.norm3 = gn(out_channels)

        self.act = nn.GELU()

        # SE block for channel recalibration
        self.se = SEBlock3D(out_channels, reduction)

        # Skip connection (if dims differ)
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                gn(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        out = self.se(out)  # Channel recalibration

        if self.shortcut is not None:
            residual = self.shortcut(x)

        return self.act(out + residual)


# ============================================================
# 3D ResNet-50 Encoder
# ============================================================

class ResNet3DEncoder(nn.Module):
    """
    3D ResNet-50 encoder with SE blocks for volumetric MRI.

    Architecture (following ResNet-50 bottleneck design):
      Stem: 7×7×7 Conv, stride=2, MaxPool (stride=2)
      Stage 1: 3 ResBlocks, 64→256 channels
      Stage 2: 4 ResBlocks, 256→512 channels, stride=2
      Stage 3: 6 ResBlocks, 512→1024 channels, stride=2
      Stage 4: 3 ResBlocks, 1024→2048 channels, stride=2
      Head: GAP + GMP → Linear projection

    Args:
        in_channels: Input channels (1 for MRI)
        base_channels: Number of channels in stem (default 64)
        out_dim: Output embedding dimension
        dropout: Dropout before final projection
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        out_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.out_dim = out_dim
        c = base_channels  # 64

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, c), c),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # ---- Stage 1: 3 blocks, 64 → 256 ----
        self.stage1 = self._make_stage(c, c//4, c*4, n_blocks=3, stride=1)
        # ---- Stage 2: 4 blocks, 256 → 512 ----
        self.stage2 = self._make_stage(c*4, c//2, c*8, n_blocks=4, stride=2)
        # ---- Stage 3: 6 blocks, 512 → 1024 ----
        self.stage3 = self._make_stage(c*8, c, c*16, n_blocks=6, stride=2)
        # ---- Stage 4: 3 blocks, 1024 → 2048 ----
        self.stage4 = self._make_stage(c*16, c*2, c*32, n_blocks=3, stride=2)

        # ---- Aggregation ----
        final_channels = c * 32  # 2048

        # Both GAP and GMP for richer descriptor
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)

        # Projection to output_dim
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * 2, out_dim),  # *2 for GAP+GMP concat
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout),
        )

        self._init_weights()
        logger.info(
            f"[ResNet3D] Initialized: base_channels={c}, out_dim={out_dim}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _make_stage(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage of n_blocks ResBlocks."""
        blocks = []
        # First block handles channel expansion and stride
        blocks.append(ResBlock3D(in_ch, mid_ch, out_ch, stride=stride))
        # Remaining blocks maintain dimensions
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock3D(out_ch, mid_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        """Initialize weights using Kaiming normal for Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        modality_present: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, 1, D, H, W]
            modality_present: [B] — 1 if modality present, 0 if missing

        Returns:
            features: [B, out_dim]
            last_conv_feat: [B, 2048, D', H', W'] — for Grad-CAM
        """
        x = self.stem(x)           # [B, 64, D/4, H/4, W/4]
        x = self.stage1(x)         # [B, 256, D/4, H/4, W/4]
        x = self.stage2(x)         # [B, 512, D/8, H/8, W/8]
        x = self.stage3(x)         # [B, 1024, D/16, H/16, W/16]
        last_conv = self.stage4(x) # [B, 2048, D/32, H/32, W/32]

        # ---- Dual Pooling ----
        x = torch.cat([
            self.gap(last_conv),
            self.gmp(last_conv),
        ], dim=1)  # [B, 4096, 1, 1, 1]

        features = self.head(x)  # [B, out_dim]

        # Zero out for missing modalities
        if modality_present is not None:
            mask = modality_present.float().unsqueeze(1)
            features = features * mask

        return features, last_conv
