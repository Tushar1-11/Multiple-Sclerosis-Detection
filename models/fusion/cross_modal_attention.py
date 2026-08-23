"""
MSNet — Cross-Modal Attention Fusion Module
============================================
Implements bidirectional cross-modal attention to fuse features
from multiple MRI modalities and clinical metadata.

Architecture Overview:
  Modality features:  [B, n_mod, embed_dim]
  Clinical features:  [B, 1, embed_dim]
  → Concatenate as a token sequence
  → Apply Transformer layers (multi-head self-attention)
  → Extract [CLS] token or mean-pool as fused representation

Key design decisions:
  1. CLS Token — Learnable global summary token prepended to sequence
  2. Modality Masking — Attention masking for missing modalities
  3. Positional Encoding — Learnable per-modality positions
  4. Gating Mechanism — Sigmoid gate controlling cross-modal info flow

Reference:
  Vaswani et al. (2017) — "Attention Is All You Need", NeurIPS
  Tsai et al. (2019) — "Multimodal Transformer for Unaligned Multimodal
  Language Sequences", ACL 2019
  Chen et al. (2021) — "TransUNet: Transformers Make Strong Encoders for
  Medical Image Segmentation"
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ============================================================
# Learnable Positional Encoding
# ============================================================

class ModalityPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for each modality token.

    Instead of fixed sinusoidal encodings, we use learnable embeddings
    that allow the model to encode modality-specific positional structure.

    Tokens (in order):
      [CLS] | T1 | T2 | FLAIR | GD_T1 | Clinical

    Args:
        n_tokens: Total number of tokens (n_modalities + 2)
        embed_dim: Embedding dimension
        dropout: Positional dropout rate
    """

    def __init__(self, n_tokens: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n_tokens, embed_dim) * 0.02
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to token sequence.

        Args:
            x: [B, n_tokens, embed_dim]

        Returns:
            [B, n_tokens, embed_dim]
        """
        return self.drop(x + self.pos_embedding[:, :x.size(1), :])


# ============================================================
# Cross-Modal Attention Layer
# ============================================================

class CrossModalAttentionLayer(nn.Module):
    """
    Single Transformer encoder layer for cross-modal fusion.

    Combines:
      - Multi-head self-attention (full cross-modal interaction)
      - Pre-LayerNorm for stable training (Pre-LN > Post-LN for deep nets)
      - GELU feedforward network
      - Optional residual gating

    Args:
        embed_dim: Token embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: FFN hidden dim = embed_dim * mlp_ratio
        dropout: Attention and FFN dropout
        attn_dropout: Attention weights dropout
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-LayerNorm for self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # Pre-LayerNorm for FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        # Residual gate — learn to balance residual vs. new information
        self.gate_alpha = nn.Parameter(torch.ones(1))
        self.drop_path_rate = 0.0  # Can be set for stochastic depth

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, n_tokens, embed_dim] — token sequence
            key_padding_mask: [B, n_tokens] — True = IGNORE this token
                              (used to mask missing modality tokens)

        Returns:
            x: [B, n_tokens, embed_dim] — Updated tokens
            attn_weights: [B, n_tokens, n_tokens] — Attention map
        """
        # ---- Self-Attention with Pre-LN ----
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )
        x = x + torch.sigmoid(self.gate_alpha) * attn_out

        # ---- FFN with Pre-LN ----
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights


# ============================================================
# Full Cross-Modal Attention Fusion Module
# ============================================================

class CrossModalAttentionFusion(nn.Module):
    """
    Full multi-layer cross-modal Transformer fusion module.

    Fuses MRI modality features and clinical features into a
    single unified representation using Transformer self-attention.

    Token sequence structure:
      [CLS] [T1] [T2] [FLAIR] [GD_T1] [Clinical]
       ↑                                    ↑
    global summary                    one token for all clinical

    The [CLS] token aggregates cross-modal context after L
    attention layers and is used as the final fused representation.

    Args:
        n_modalities: Number of MRI modalities (4 for T1/T2/FLAIR/GdT1)
        embed_dim: Token embedding dimension (must match encoder output)
        num_heads: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dropout: Dropout rate
        use_clinical: Whether to include clinical token in sequence
        use_positional_encoding: Whether to add positional encodings
    """

    def __init__(
        self,
        n_modalities: int = 4,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_clinical: bool = True,
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.n_modalities = n_modalities
        self.embed_dim = embed_dim
        self.use_clinical = use_clinical

        # Number of tokens = 1 (CLS) + n_modalities + 1 (clinical, optional)
        n_clinical_tokens = 1 if use_clinical else 0
        self.n_tokens = 1 + n_modalities + n_clinical_tokens

        # ---- CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ---- Positional Encoding ----
        self.pos_enc = (
            ModalityPositionalEncoding(self.n_tokens, embed_dim, dropout)
            if use_positional_encoding
            else nn.Identity()
        )

        # ---- Transformer Layers ----
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])

        # ---- Output Projection ----
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()
        logger.info(
            f"[CrossModalFusion] tokens={self.n_tokens}, "
            f"embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}"
        )

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_attention_mask(
        self,
        modality_mask: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Build key_padding_mask for missing modalities.

        key_padding_mask: [B, n_tokens] — True = IGNORE this position

        CLS and Clinical tokens are never masked.
        Missing modality tokens (mask==0) ARE masked from attention.

        Args:
            modality_mask: [B, n_modalities] — 1=present, 0=missing
            batch_size: Batch size

        Returns:
            padding_mask or None
        """
        if modality_mask is None:
            return None

        # True means "ignore" in PyTorch's convention
        # [B, 1(CLS)] + [B, n_mod(MRI)] + [B, 1(clinical)]
        cls_mask = torch.zeros(batch_size, 1, device=modality_mask.device, dtype=torch.bool)
        mri_mask = (modality_mask == 0)  # [B, n_modalities]
        
        if self.use_clinical:
            clinical_mask = torch.zeros(batch_size, 1, device=modality_mask.device, dtype=torch.bool)
            padding_mask = torch.cat([cls_mask, mri_mask, clinical_mask], dim=1)
        else:
            padding_mask = torch.cat([cls_mask, mri_mask], dim=1)

        return padding_mask  # [B, n_tokens]

    def forward(
        self,
        mri_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse MRI and clinical features via cross-modal attention.

        Args:
            mri_features: [B, n_modalities, embed_dim]
            clinical_features: [B, embed_dim] or [B, 1, embed_dim] (optional)
            modality_mask: [B, n_modalities] — 1=present, 0=missing

        Returns:
            fused: [B, embed_dim] — global fused representation (CLS token)
            all_attn_weights: list of [B, heads, n_tokens, n_tokens]
        """
        B = mri_features.shape[0]

        # ---- Build Token Sequence ----
        # CLS token (replicated across batch)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]

        # Start with CLS + MRI modality tokens
        tokens = [cls, mri_features]  # [B, 1+n_mod, embed_dim]

        # Append clinical token if available
        if self.use_clinical and clinical_features is not None:
            if clinical_features.dim() == 2:
                clinical_features = clinical_features.unsqueeze(1)  # [B, 1, embed_dim]
            tokens.append(clinical_features)

        token_seq = torch.cat(tokens, dim=1)  # [B, n_tokens, embed_dim]

        # ---- Add Positional Encoding ----
        token_seq = self.pos_enc(token_seq)

        # ---- Build Attention Mask ----
        attn_mask = self._build_attention_mask(modality_mask, B)

        # ---- Transformer Layers ----
        all_attn_weights = []
        for layer in self.layers:
            token_seq, attn_w = layer(token_seq, key_padding_mask=attn_mask)
            all_attn_weights.append(attn_w)

        # ---- Extract CLS Token ----
        cls_output = token_seq[:, 0, :]           # [B, embed_dim]
        cls_output = self.out_norm(cls_output)
        fused = self.out_proj(cls_output)          # [B, embed_dim]

        return fused, all_attn_weights
