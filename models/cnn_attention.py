"""
MSNet -- Multi-Branch Tabular Deep Neural Network (CNNAttentionHybrid)
====================================================================
Implements the exact multi-branch architecture:

1. Input Features: Tabular Biomarkers [B, in_features]
2. Three Parallel Tokenizer/Embedding Branches:
   - 1D-ResNet CNN + SE-Blocks (ResNet1DSEBranch)
   - FT-Transformer Numerical Tokenizer (FTTransformerNumericalTokenizer)
   - TabTransformer Column Embeddings (TabTransformerColumnEmbeddings)
3. Combined Feature Token Sequence + CLS Token + Learned Positional Encoding
4. Dual Recurrent Branches (BiLSTM + BiGRU)
5. Recurrent Residual Fusion (Gated residual skip connection)
6. Stacked Transformer Self-Attention Blocks (Pre-LN MHSA + GeGLU FFN + DropPath)
7. Latent Fusion: CLS + Attention Pool + Stacking
8. Multi-Task Output Heads:
   - Head 1: Phenotype Classifier (3-class: HC / RRMS / Progressive MS)
   - Head 2: EDSS Score Regressor (continuous 0.0 - 10.0 score)
   - Head 3: Binary Group Classifier (2-class: HC vs MS Patient)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# Stochastic Depth (DropPath)
# ============================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x * mask / keep


# ============================================================
# Squeeze-and-Excitation 1D Block
# ============================================================

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


# ============================================================
# 1D Bottleneck Residual Block
# ============================================================

class BottleneckResBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        stride: int = 1,
        dilation: int = 1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        pad = dilation

        def _norm(c):
            return nn.GroupNorm(min(8, c) if c >= 8 else 1, c)

        self.c1 = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
        self.n1 = _norm(mid_ch)

        self.c2 = nn.Conv1d(
            mid_ch, mid_ch, 3, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.n2 = _norm(mid_ch)

        self.c3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.n3 = _norm(out_ch)

        self.se = SEBlock1D(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop_path = DropPath(drop_path_rate)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        y = self.act(self.n1(self.c1(x)))
        y = self.act(self.n2(self.c2(y)))
        y = self.n3(self.c3(y))
        y = self.se(y)
        return self.act(res + self.drop_path(y))


# ============================================================
# Branch 1: 1D-ResNet CNN + SE-Blocks
# ============================================================

class ResNet1DSEBranch(nn.Module):
    """
    1D-ResNet CNN Backbone with Squeeze-and-Excitation Attention.
    Transforms raw tabular vector into a multi-channel feature token sequence.
    """

    def __init__(
        self,
        in_features: int,
        token_dim: int = 256,
        channels: Tuple[int, int, int, int] = (128, 256, 256, 256),
        seq_len: int = 32,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        blocks_per_stage: Tuple[int, int, int, int] = (2, 3, 3, 2),
    ):
        super().__init__()
        self.seq_len = seq_len
        c0 = channels[0]

        # Input Stem: projects 1D vector -> 2D sequence [B, c0, seq_len]
        self.stem = nn.Sequential(
            nn.Linear(in_features, c0 * seq_len),
            nn.LayerNorm(c0 * seq_len),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        stages = []
        in_c = c0
        total_blocks = sum(blocks_per_stage)
        block_idx = 0

        for stage_i, (out_c, num_b) in enumerate(zip(channels, blocks_per_stage)):
            stage_blocks = []
            for b in range(num_b):
                dp = drop_path_rate * (block_idx / max(total_blocks - 1, 1))
                mid_c = out_c // 2
                stage_blocks.append(
                    BottleneckResBlock1D(
                        in_ch=in_c if b == 0 else out_c,
                        mid_ch=mid_c,
                        out_ch=out_c,
                        drop_path_rate=dp,
                    )
                )
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
            in_c = out_c

        self.stages = nn.ModuleList(stages)
        self.proj_out = nn.Sequential(
            nn.Conv1d(channels[-1], token_dim, 1, bias=False),
            nn.GroupNorm(min(8, token_dim), token_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]
        Returns: [B, seq_len, token_dim]
        """
        B = x.shape[0]
        h = self.stem(x).view(B, -1, self.seq_len)
        for stage in self.stages:
            h = stage(h)
        h = self.proj_out(h)  # [B, token_dim, seq_len]
        return h.transpose(1, 2)  # [B, seq_len, token_dim]


# ============================================================
# Branch 2: FT-Transformer Numerical Tokenizer
# ============================================================

class FTTransformerNumericalTokenizer(nn.Module):
    """
    Feature Tokenizer from FT-Transformer (Gorishniy et al., 2021).
    Maps each scalar numerical feature into a d-dimensional token vector.
    """

    def __init__(self, in_features: int, token_dim: int = 256):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(in_features, token_dim))
        self.bias = nn.Parameter(torch.empty(in_features, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.act = nn.GELU()

        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]
        Returns: [B, in_features, token_dim]
        """
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.act(self.norm(tokens))


# ============================================================
# Branch 3: TabTransformer Column Embeddings
# ============================================================

class TabTransformerColumnEmbeddings(nn.Module):
    """
    Learned Column Identifiers & Feature Value Embeddings (TabTransformer inspired).
    Combines feature identity embeddings with non-linear feature projections.
    """

    def __init__(self, in_features: int, token_dim: int = 256):
        super().__init__()
        self.col_embeddings = nn.Parameter(torch.empty(in_features, token_dim))
        self.val_weight = nn.Parameter(torch.empty(in_features, token_dim))
        self.val_bias = nn.Parameter(torch.empty(in_features, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.act = nn.GELU()

        nn.init.normal_(self.col_embeddings, std=0.02)
        nn.init.kaiming_uniform_(self.val_weight, a=math.sqrt(5))
        nn.init.zeros_(self.val_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]
        Returns: [B, in_features, token_dim]
        """
        val_proj = x.unsqueeze(-1) * self.val_weight.unsqueeze(0) + self.val_bias.unsqueeze(0)
        tokens = self.col_embeddings.unsqueeze(0) + self.act(val_proj)
        return self.norm(tokens)


# ============================================================
# Dual Recurrent Branches (BiLSTM & BiGRU)
# ============================================================

class BiLSTMBranch(nn.Module):
    """Bidirectional LSTM Sequential Branch."""

    def __init__(self, token_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=token_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.dropout(self.proj(out))


class BiGRUBranch(nn.Module):
    """Bidirectional GRU Gated Branch."""

    def __init__(self, token_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim, token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.dropout(self.proj(out))


# ============================================================
# Recurrent Residual Fusion
# ============================================================

class RecurrentResidualFusion(nn.Module):
    """
    Recurrent Residual Fusion:
    Fuses BiLSTM and BiGRU representations via a learned gating mechanism
    and merges back into the input sequence via a residual skip connection.
    """

    def __init__(self, token_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bilstm = BiLSTMBranch(token_dim=token_dim, hidden_dim=token_dim, dropout=dropout)
        self.bigru = BiGRUBranch(token_dim=token_dim, hidden_dim=token_dim, dropout=dropout)

        # Learned Gating Mechanism
        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: [B, L, token_dim]
        Returns: [B, L, token_dim]
        """
        lstm_out = self.bilstm(x_tokens)
        gru_out = self.bigru(x_tokens)

        # Compute fusion gate
        g = self.gate(torch.cat([lstm_out, gru_out], dim=-1))
        recurrent_fused = g * lstm_out + (1.0 - g) * gru_out

        # Residual connection
        return self.norm(x_tokens + self.dropout(recurrent_fused))


# ============================================================
# Stacked Transformer Self-Attention Blocks
# ============================================================

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder Block with GeGLU Feed-Forward Network."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = dim * ffn_expansion

        # GeGLU (Gated Linear Unit with GELU) FFN
        self.ffn_proj = nn.Linear(dim, ffn_dim * 2)
        self.ffn_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Self-Attention
        nx = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx)
        x = x + self.drop_path1(attn_out)

        # GeGLU FFN
        nx = self.norm2(x)
        proj = self.ffn_proj(nx)
        val, gate = proj.chunk(2, dim=-1)
        ffn_out = self.ffn_out(val * F.gelu(gate))
        x = x + self.drop_path2(ffn_out)
        return x


# ============================================================
# Latent Fusion: CLS + Attention Pool + Stacking
# ============================================================

class LatentFusionPool(nn.Module):
    """
    Latent Fusion Pooling:
    Combines CLS token embedding, Multi-Head Attention Pooling across non-CLS tokens,
    and sequence Mean/Max pooling into a unified latent representation vector.
    """

    def __init__(self, token_dim: int = 256, attn_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.token_dim = token_dim

        # Multi-Head Attention Pooling Query
        self.pool_query = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True
        )

        # Fusion MLP Projection (CLS [token_dim] + AttnPool [token_dim] + MeanPool [token_dim] + MaxPool [token_dim])
        self.fusion_mlp = nn.Sequential(
            nn.Linear(token_dim * 4, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        seq_tokens: [B, 1 + L, token_dim]
        Returns: [B, attn_dim]
        """
        B = seq_tokens.shape[0]

        # 1. CLS Token
        cls_embed = seq_tokens[:, 0, :]  # [B, token_dim]

        # Non-CLS sequence tokens
        non_cls_tokens = seq_tokens[:, 1:, :]  # [B, L, token_dim]

        # 2. Multi-Head Attention Pooling
        q = self.pool_query.expand(B, -1, -1)  # [B, 1, token_dim]
        attn_pooled, _ = self.pool_attn(q, non_cls_tokens, non_cls_tokens)
        attn_embed = attn_pooled.squeeze(1)  # [B, token_dim]

        # 3. Sequence Stacking (Mean Pool + Max Pool)
        mean_pooled = non_cls_tokens.mean(dim=1)  # [B, token_dim]
        max_pooled, _ = non_cls_tokens.max(dim=1)   # [B, token_dim]

        # Concatenate all 4 representation views
        stacked = torch.cat([cls_embed, attn_embed, mean_pooled, max_pooled], dim=-1)  # [B, 4 * token_dim]

        return self.fusion_mlp(stacked)  # [B, attn_dim]


# ============================================================
# Multi-Task Output Head
# ============================================================

class TaskHead(nn.Module):
    """Task-specific Multi-Task Prediction Head."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        sigmoid_scale: Optional[float] = None,
    ):
        super().__init__()
        self.sigmoid_scale = sigmoid_scale
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.sigmoid_scale is not None:
            out = torch.sigmoid(out) * self.sigmoid_scale
        return out


# ============================================================
# Main Neural Network: CNNAttentionHybrid
# ============================================================

class CNNAttentionHybrid(nn.Module):
    """
    Complete Multi-Branch Neural Network Architecture for Tabular MS Biomarkers.

    Architecture summary:
      1. Input Features: Tabular Biomarkers [B, in_features]
      2. 3 Parallel Branches:
         - Branch 1: 1D-ResNet CNN + SE-Blocks -> [B, L_cnn, token_dim]
         - Branch 2: FT-Transformer Numerical Tokenizer -> [B, in_features, token_dim]
         - Branch 3: TabTransformer Column Embeddings -> [B, in_features, token_dim]
      3. Combined Feature Token Sequence + CLS Token + Positional Encoding -> [B, 1+L, token_dim]
      4. Dual Recurrent Branches (BiLSTM + BiGRU)
      5. Recurrent Residual Fusion -> [B, 1+L, token_dim]
      6. Stacked Transformer Self-Attention Blocks -> [B, 1+L, token_dim]
      7. Latent Fusion: CLS + Attention Pool + Stacking -> [B, attn_dim]
      8. Multi-Task Output Heads:
         - Head 1: Phenotype Classifier -> [B, num_phenotype_classes] (logits)
         - Head 2: EDSS Score Regressor -> [B, 1] (score 0-10)
         - Head 3: Binary Group Classifier -> [B, num_group_classes] (logits)
    """

    def __init__(
        self,
        in_features: int,
        token_dim: int = 256,
        cnn_channels: Tuple[int, int, int, int] = (128, 256, 256, 256),
        cnn_seq_len: int = 32,
        cnn_dropout: float = 0.2,
        cnn_drop_path_rate: float = 0.1,
        cnn_blocks_per_stage: Tuple[int, int, int, int] = (2, 3, 3, 2),
        num_attn_layers: int = 6,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        attn_dropout: float = 0.1,
        attn_drop_path_rate: float = 0.1,
        attn_dim: int = 256,
        head_dropout: float = 0.3,
        num_phenotype_classes: int = 3,
        num_group_classes: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.token_dim = token_dim
        self.attn_dim = attn_dim

        # ── 1. Three Parallel Feature Tokenizer / Embedding Branches ──
        # Branch 1: 1D-ResNet CNN + SE-Blocks
        self.branch1_cnn = ResNet1DSEBranch(
            in_features=in_features,
            token_dim=token_dim,
            channels=cnn_channels,
            seq_len=cnn_seq_len,
            dropout=cnn_dropout,
            drop_path_rate=cnn_drop_path_rate,
            blocks_per_stage=cnn_blocks_per_stage,
        )

        # Branch 2: FT-Transformer Numerical Tokenizer
        self.branch2_ft = FTTransformerNumericalTokenizer(
            in_features=in_features,
            token_dim=token_dim,
        )

        # Branch 3: TabTransformer Column Embeddings
        self.branch3_tab = TabTransformerColumnEmbeddings(
            in_features=in_features,
            token_dim=token_dim,
        )

        # ── 2. Combined Token Sequence + CLS Token + Positional Encoding ──
        total_tokens = cnn_seq_len + in_features + in_features  # L_cnn + 2 * in_features
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens + 1, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(attn_dropout)

        # ── 3. Dual Recurrent Branches & Recurrent Residual Fusion ──
        self.recurrent_fusion = RecurrentResidualFusion(
            token_dim=token_dim, dropout=attn_dropout
        )

        # ── 4. Stacked Transformer Self-Attention Blocks ──
        dp_rates = [
            x.item() for x in torch.linspace(0, attn_drop_path_rate, num_attn_layers)
        ]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=token_dim,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                dropout=attn_dropout,
                drop_path_rate=dp_rates[i],
            )
            for i in range(num_attn_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)

        # ── 5. Latent Fusion (CLS + Attention Pool + Stacking) ──
        self.latent_fusion = LatentFusionPool(
            token_dim=token_dim, attn_dim=attn_dim, dropout=head_dropout
        )

        # ── 6. Three Output Heads ──
        self.phenotype_head = TaskHead(attn_dim, num_phenotype_classes, head_dropout)
        self.group_head = TaskHead(attn_dim, num_group_classes, head_dropout)
        self.edss_head = TaskHead(attn_dim, 1, head_dropout, sigmoid_scale=10.0)

        # Parameter summary logging
        self._log_params()

    def _log_params(self):
        b1_p = sum(p.numel() for p in self.branch1_cnn.parameters())
        b2_p = sum(p.numel() for p in self.branch2_ft.parameters())
        b3_p = sum(p.numel() for p in self.branch3_tab.parameters())
        rec_p = sum(p.numel() for p in self.recurrent_fusion.parameters())
        trans_p = sum(p.numel() for p in self.transformer_blocks.parameters())
        lat_p = sum(p.numel() for p in self.latent_fusion.parameters())
        heads_p = (
            sum(p.numel() for p in self.phenotype_head.parameters())
            + sum(p.numel() for p in self.group_head.parameters())
            + sum(p.numel() for p in self.edss_head.parameters())
        )
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CNNAttentionHybrid | Total: {total:,} params "
            f"(CNN: {b1_p:,} | FT: {b2_p:,} | Tab: {b3_p:,} | Recurrent: {rec_p:,} | "
            f"Transformer: {trans_p:,} | LatentFusion: {lat_p:,} | Heads: {heads_p:,})"
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features]
        Returns: [B, attn_dim] latent fused vector
        """
        B = x.shape[0]

        # Stage 1: 3 Parallel Branches
        tok_cnn = self.branch1_cnn(x)  # [B, L_cnn, token_dim]
        tok_ft = self.branch2_ft(x)    # [B, in_features, token_dim]
        tok_tab = self.branch3_tab(x)  # [B, in_features, token_dim]

        # Stage 2: Combine tokens + CLS token + Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, token_dim]
        seq_tokens = torch.cat([cls_tokens, tok_cnn, tok_ft, tok_tab], dim=1)  # [B, 1 + L_total, token_dim]
        seq_tokens = self.pos_drop(seq_tokens + self.pos_embed)

        # Stage 3: Dual Recurrent Branches & Recurrent Residual Fusion
        seq_tokens = self.recurrent_fusion(seq_tokens)  # [B, 1 + L_total, token_dim]

        # Stage 4: Stacked Transformer Self-Attention Blocks
        for blk in self.transformer_blocks:
            seq_tokens = blk(seq_tokens)
        seq_tokens = self.final_norm(seq_tokens)

        # Stage 5: Latent Fusion (CLS + Attention Pool + Stacking)
        latent_vector = self.latent_fusion(seq_tokens)  # [B, attn_dim]
        return latent_vector

    def forward(
        self,
        x: torch.Tensor,
        task: str = "phenotype",
        return_all_heads: bool = False,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with support for single task or multi-task prediction.

        Args:
            x: [B, in_features]
            task: 'phenotype', 'group', or 'edss'
            return_all_heads: If True, returns dict with all task outputs
            return_all: Alias for return_all_heads

        Returns:
            Logits / predictions tensor OR dictionary of output tensors
        """
        latent = self.forward_features(x)

        if return_all_heads or return_all:
            pheno = self.phenotype_head(latent)
            group = self.group_head(latent)
            edss = self.edss_head(latent)
            return {
                "phenotype": pheno,
                "phenotype_logits": pheno,
                "group": group,
                "group_logits": group,
                "edss": edss,
                "edss_pred": edss,
                "latent_features": latent,
            }

        if task == "phenotype":
            return self.phenotype_head(latent)
        elif task == "group":
            return self.group_head(latent)
        elif task == "edss":
            return self.edss_head(latent)
        else:
            raise ValueError(f"Unknown task '{task}'. Use 'phenotype', 'group', 'edss', or return_all=True")
