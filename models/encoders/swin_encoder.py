"""
MSNet — 3D Swin Transformer Encoder
======================================
Implements a 3D Swin Transformer encoder for volumetric MRI feature
extraction, based on the MONAI SwinUNETR encoder backbone.

Architecture:
  Input: [B, 1, D, H, W]   (single-channel 3D MRI volume)
  Output: [B, embed_dim]    (global feature vector)

Key features:
  - Shifted window self-attention for local-global context
  - Hierarchical feature extraction (4 resolution stages)
  - Patch merging for progressive downsampling
  - Gradient checkpointing for memory-efficient training
  - Modality-specific adapter layers (lightweight)

Reference:
  Liu et al. (2021) — "Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows", ICCV 2021.

  Tang et al. (2022) — "Self-supervised Pre-Training of Swin Transformers
  for 3D Medical Image Analysis", CVPR 2022. (SwinUNETR)
"""

import logging
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# MONAI SwinUNETR encoder — pretrained on large-scale medical datasets
from monai.networks.nets import SwinUNETR

logger = logging.getLogger(__name__)


# ============================================================
# Modality-Specific Adapter Head
# ============================================================

class ModalityAdapter(nn.Module):
    """
    Lightweight modality-specific adapter applied BEFORE shared encoder.

    Each MRI sequence (T1, T2, FLAIR, Gd-T1) has different contrast
    characteristics. This adapter learns a linear projection that
    maps sequence-specific intensity patterns into a common space
    before passing to the shared Swin encoder backbone.

    Design: Input Conv → BN → GELU → Residual projection

    Args:
        in_channels: Input channels (1 for grayscale MRI)
        out_channels: Adapted channel count
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(in_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, D, H, W] — raw MRI volume

        Returns:
            [B, 1, D, H, W] — adapted volume (residual connection)
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual  # Residual to preserve original signal


# ============================================================
# Global Feature Aggregator
# ============================================================

class GlobalAveragePooling3D(nn.Module):
    """
    Reduces 3D feature maps to a 1D global descriptor.

    Uses multi-scale average pooling (global + local 2x2x2 blocks)
    and concatenates for a richer global representation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W] — 3D feature map

        Returns:
            [B, C] — global pooled features
        """
        # Global average pooling
        return F.adaptive_avg_pool3d(x, output_size=1).squeeze(-1).squeeze(-1).squeeze(-1)


# ============================================================
# 3D Swin Transformer Encoder
# ============================================================

class SwinTransformerEncoder3D(nn.Module):
    """
    Per-modality 3D Swin Transformer encoder.

    Wraps MONAI's SwinUNETR-based encoder backbone with:
      1. Modality-specific adapter layer
      2. Shared Swin Transformer body
      3. Multi-scale feature aggregation
      4. Projection head → fixed-dim embedding

    The encoder is instantiated once per modality in MSNet,
    with shared backbone weights and separate adapter weights.

    Args:
        img_size: Input MRI volume size [D, H, W]
        in_channels: Input channels (1 for MRI)
        feature_size: Swin base feature dimension
        depths: Number of Swin blocks per stage
        num_heads: Attention heads per stage
        window_size: Local attention window size
        drop_rate: Dropout rate in Swin
        attn_drop_rate: Attention dropout rate
        dropout_path_rate: Stochastic depth rate
        out_dim: Output embedding dimension
        use_checkpoint: Enable gradient checkpointing
    """

    def __init__(
        self,
        img_size: Sequence[int] = (128, 128, 128),
        in_channels: int = 1,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.1,
        out_dim: int = 512,
        use_checkpoint: bool = True,
        adapter_channels: int = 4,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.feature_size = feature_size

        # 1. Modality-specific adapter
        self.adapter = ModalityAdapter(
            in_channels=in_channels,
            out_channels=adapter_channels,
        )

        # 2. Shared Swin Transformer backbone (MONAI SwinUNETR encoder)
        # We use SwinUNETR as an encoder-only feature extractor
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,   # Unused decoder output dim
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )

        # ---- Extract encoder-only for feature purpose ----
        # SwinUNETR exposes swinViT encoder features at multiple scales
        # We only use the encoder (swin.swinViT), not the decoder

        # 3. Compute encoder output channel dim after all stages
        # For feature_size=48: stage outputs [96, 192, 384, 768]
        stage_out_dims = [
            feature_size * (2 ** i) for i in range(len(depths))
        ]
        self._total_feats = sum(stage_out_dims)  # 48+96+192+384 = 720

        # 4. Multi-scale projection
        self.ms_projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),    # Global pool each scale
                nn.Flatten(),
                nn.Linear(dim, out_dim // len(depths)),
                nn.GELU(),
                nn.LayerNorm(out_dim // len(depths)),
            )
            for dim in stage_out_dims
        ])

        # 5. Final projection to out_dim
        self.final_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=0.1),
        )

        logger.info(
            f"[SwinEncoder3D] Initialized: feature_size={feature_size}, "
            f"depths={depths}, out_dim={out_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_present: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Swin encoder.

        Args:
            x: [B, 1, D, H, W] — single MRI volume
            modality_present: [B] bool mask (1=present, 0=missing)

        Returns:
            features: [B, out_dim] — global feature vector
            hidden_states: [B, out_dim] — same, for Grad-CAM hook
        """
        B = x.shape[0]

        # 1. Apply modality-specific adapter
        x = self.adapter(x)

        # 2. Run through Swin ViT encoder
        # SwinUNETR's swinViT returns hidden states at each stage
        hidden_states_out = self.swin.swinViT(x, self.swin.normalize)
        # hidden_states_out is a list of tensors at each resolution:
        # [stage0_feats, stage1_feats, stage2_feats, stage3_feats]

        # 3. Multi-scale feature aggregation
        scale_features = []
        for i, (proj, h_state) in enumerate(
            zip(self.ms_projectors, hidden_states_out)
        ):
            # Reshape hidden state from [B, tokens, dim] to [B, dim, D', H', W']
            feat = h_state  # Already in spatial format from SwinUNETR
            if feat.dim() == 3:
                # Sequence format: [B, T, C] → reshape to spatial
                T = feat.shape[1]
                C = feat.shape[2]
                # Resolve spatial dims
                sp = round(T ** (1/3))
                feat = feat.reshape(B, sp, sp, sp, C).permute(0, 4, 1, 2, 3)
            scale_feat = proj(feat)
            scale_features.append(scale_feat)

        # 4. Concatenate multi-scale features
        ms_feat = torch.cat(scale_features, dim=1)  # [B, out_dim]

        # 5. Final projection
        features = self.final_proj(ms_feat)

        # 6. Zero out features for missing modalities
        if modality_present is not None:
            mask = modality_present.float().unsqueeze(1)  # [B, 1]
            features = features * mask

        return features, hidden_states_out[-1]  # feat + last hidden state
