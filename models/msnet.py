"""
MSNet — Main Multimodal MS Detection Model
============================================
The MSNet model integrates all components into a unified architecture:

  1. Per-modality 3D encoders (Swin Transformer or 3D CNN)
  2. Clinical metadata MLP encoder
  3. Cross-modal attention fusion Transformer
  4. Uncertainty-aware classification head

Model supports:
  - Binary classification: HC vs MS
  - Multi-class: HC / RRMS / SPMS / PPMS
  - Missing modality handling via masking
  - MC Dropout or Evidential uncertainty estimation
  - Modality-specific feature inspection
  - Feature extraction mode (for downstream tasks)

Architecture summary (Swin variant, 4-class):
  ┌──────────────────────────────────────────────────────┐
  │        Input: [B, 4, 1, 128, 128, 128] MRI           │
  │                + [B, 7] Clinical                     │
  │                + [B, 4] Modality mask                │
  └──────────────────────────────────────────────────────┘
                          ↓
  ┌──────── Per-Modality Encoders (shared weights) ───────┐
  │    T1    T2    FLAIR    GD_T1  →  [B, 4, embed_dim]  │
  └───────────────────────────────────────────────────────┘
                          ↓
  ┌──────── Clinical Encoder ──────────────────────────────┐
  │    [B, 7] → ClinicalMLP → [B, embed_dim]              │
  └───────────────────────────────────────────────────────┘
                          ↓
  ┌──────── Cross-Modal Attention Fusion ─────────────────┐
  │  [CLS|T1|T2|FLAIR|GdT1|Clinical] → CLS: [B, embed]   │
  └───────────────────────────────────────────────────────┘
                          ↓
  ┌──────── Uncertainty Classifier ───────────────────────┐
  │   [B, embed_dim] → {logits, probs, uncertainty}       │
  └───────────────────────────────────────────────────────┘
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Internal module imports
from models.encoders.swin_encoder import SwinTransformerEncoder3D
from models.encoders.cnn_encoder import ResNet3DEncoder
from models.encoders.clinical_encoder import ClinicalEncoder
from models.fusion.cross_modal_attention import CrossModalAttentionFusion
from models.heads.uncertainty_head import (
    ClassifierHead,
    MCDropoutUncertaintyHead,
    EvidentialClassifierHead,
)

logger = logging.getLogger(__name__)

# Canonical modality ordering
MODALITIES = ["T1", "T2", "FLAIR", "GD_T1"]


class MSNet(nn.Module):
    """
    MSNet: Multimodal Deep Learning Framework for MS Detection.

    A unified architecture that fuses 3D MRI features from multiple
    sequences with structured clinical metadata for accurate Multiple
    Sclerosis detection and subtype classification.

    Args:
        config: Full configuration dictionary (from YAML)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        model_cfg = config["model"]
        data_cfg  = config["data"]

        # ---- Key Parameters ----
        self.n_modalities  = len(MODALITIES)                  # 4
        self.num_classes   = data_cfg["num_classes"]          # 2 or 4
        self.task          = data_cfg["task"]                 # binary/multiclass
        self.encoder_type  = model_cfg["encoder_type"]        # swin/resnet3d
        embed_dim          = model_cfg["fusion"]["embed_dim"] # 512
        self.embed_dim     = embed_dim

        # ---- Build MRI Encoders ----
        # One encoder per modality. By default, they SHARE the backbone
        # weights (parameter efficient) but have separate adapter layers.
        if self.encoder_type == "swin_transformer":
            self._build_swin_encoders(model_cfg)
        elif self.encoder_type == "resnet3d":
            self._build_cnn_encoders(model_cfg)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        # ---- Projection: Encoder Out → Fusion Embed Dim ----
        # Projects each modality's encoder output to the shared embed_dim
        encoder_out_dim = self.mri_encoders[0].out_dim
        self.mri_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_out_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            for _ in MODALITIES
        ])

        # ---- Clinical Encoder ----
        clinical_cfg = model_cfg["clinical"]
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_cfg["input_dim"],
            hidden_dims=clinical_cfg["hidden_dims"],
            out_dim=clinical_cfg["out_dim"],
            dropout=clinical_cfg["dropout"],
        )

        # Projection: clinical out → fusion embed_dim
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_cfg["out_dim"], embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ---- Cross-Modal Attention Fusion ----
        fusion_cfg = model_cfg["fusion"]
        self.fusion = CrossModalAttentionFusion(
            n_modalities=self.n_modalities,
            embed_dim=embed_dim,
            num_heads=fusion_cfg["num_heads"],
            num_layers=fusion_cfg["num_layers"],
            dropout=fusion_cfg["dropout"],
            use_clinical=True,
            use_positional_encoding=fusion_cfg["use_positional_encoding"],
        )

        # ---- Classification Head ----
        classifier_cfg = model_cfg["classifier"]
        self.classifier = ClassifierHead(
            in_dim=embed_dim,
            num_classes=self.num_classes,
            hidden_dims=classifier_cfg["hidden_dims"],
            dropout=classifier_cfg["dropout"],
        )

        # ---- Uncertainty Head ----
        unc_cfg = model_cfg["uncertainty"]
        self.use_uncertainty = unc_cfg["enabled"]
        self.uncertainty_method = unc_cfg.get("method", "mc_dropout")

        if self.use_uncertainty:
            if self.uncertainty_method == "mc_dropout":
                self.uncertainty_head = MCDropoutUncertaintyHead(
                    classifier_head=self.classifier,
                    n_mc_samples=unc_cfg["mc_samples"],
                    num_classes=self.num_classes,
                )
            elif self.uncertainty_method == "evidential":
                self.uncertainty_head = EvidentialClassifierHead(
                    in_dim=embed_dim,
                    num_classes=self.num_classes,
                )

        # ---- Log model summary ----
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*50}\n"
            f"MSNet initialized:\n"
            f"  Encoder:      {self.encoder_type}\n"
            f"  Fusion:       cross_modal_attention (dim={embed_dim})\n"
            f"  Task:         {self.task} ({self.num_classes} classes)\n"
            f"  Uncertainty:  {self.uncertainty_method if self.use_uncertainty else 'disabled'}\n"
            f"  Parameters:   {total_params:,} total, {trainable:,} trainable\n"
            f"{'='*50}"
        )

    # ----------------------------------------------------------
    # Encoder Builders
    # ----------------------------------------------------------

    def _build_swin_encoders(self, model_cfg: dict):
        """Build shared-weight Swin Transformer encoders."""
        swin_cfg = model_cfg["swin"]
        out_dim  = model_cfg["fusion"]["embed_dim"]

        # SHARED backbone: all modalities use same weights + separate adapters
        shared_encoder = SwinTransformerEncoder3D(
            img_size=tuple(swin_cfg["img_size"]),
            in_channels=swin_cfg["in_channels"],
            feature_size=swin_cfg["feature_size"],
            depths=tuple(swin_cfg["depths"]),
            num_heads=tuple(swin_cfg["num_heads"]),
            window_size=swin_cfg["window_size"],
            drop_rate=swin_cfg["drop_rate"],
            attn_drop_rate=swin_cfg["attn_drop_rate"],
            dropout_path_rate=swin_cfg["dropout_path_rate"],
            out_dim=out_dim,
            use_checkpoint=swin_cfg["use_checkpoint"],
        )

        # Each modality gets its OWN adapter + shared backbone
        self.mri_encoders = nn.ModuleList([shared_encoder] * self.n_modalities)
        # NOTE: ModuleList with same object = shared params.
        #       Adapters are separate in each SwinEncoder instance.

    def _build_cnn_encoders(self, model_cfg: dict):
        """Build 3D ResNet CNN encoders."""
        cnn_cfg = model_cfg["cnn"]
        out_dim = model_cfg["fusion"]["embed_dim"]

        shared_encoder = ResNet3DEncoder(
            in_channels=cnn_cfg["in_channels"],
            out_dim=out_dim,
            dropout=cnn_cfg["dropout"],
        )
        self.mri_encoders = nn.ModuleList([shared_encoder] * self.n_modalities)

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    def forward(
        self,
        volumes: torch.Tensor,
        clinical: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            volumes: [B, n_modalities, 1, D, H, W] — MRI sequences
            clinical: [B, n_clinical_features] — clinical feature vector
            modality_mask: [B, n_modalities] — 1=present, 0=missing/dropped
            return_uncertainty: If True, run MC Dropout inference
            return_features: If True, return intermediate features

        Returns:
            Dict containing:
              'logits'          : [B, n_classes] — raw class logits
              'probabilities'   : [B, n_classes] — softmax probabilities
              'pred_class'      : [B] — predicted class indices
              'fused_features'  : [B, embed_dim] — (if return_features)
              'modality_features': [B, n_mod, embed_dim] — (if return_features)
              'attn_weights'    : list of attention maps
              'uncertainty'     : dict of uncertainty estimates (if requested)
              'clinical_gate'   : [B, n_feat] — clinical importance weights
        """
        B = volumes.shape[0]
        output = {}

        # ============================================================
        # 1. Encode each MRI modality
        # ============================================================
        modality_features = []
        modality_hidden_states = []

        for i, modality_name in enumerate(MODALITIES):
            vol = volumes[:, i, ...]  # [B, 1, D, H, W]

            # Per-modality availability mask
            mod_mask = (modality_mask[:, i] if modality_mask is not None
                        else None)

            # Encode
            feats, hidden = self.mri_encoders[i](vol, mod_mask)
            # feats: [B, encoder_out_dim]
            # hidden: last encoder hidden state

            # Project to fusion embed_dim
            feats = self.mri_proj[i](feats)  # [B, embed_dim]

            modality_features.append(feats)
            modality_hidden_states.append(hidden)

        # Stack: [B, n_modalities, embed_dim]
        mri_token_seq = torch.stack(modality_features, dim=1)

        # ============================================================
        # 2. Encode Clinical Features
        # ============================================================
        clinical_emb, clinical_gate = self.clinical_encoder(clinical)
        # clinical_emb: [B, clinical_out_dim]
        # clinical_gate: [B, n_features] — importance weights

        clinical_emb = self.clinical_proj(clinical_emb)  # [B, embed_dim]
        output["clinical_gate"] = clinical_gate

        # ============================================================
        # 3. Cross-Modal Attention Fusion
        # ============================================================
        fused, attn_weights = self.fusion(
            mri_features=mri_token_seq,         # [B, 4, embed_dim]
            clinical_features=clinical_emb,     # [B, embed_dim]
            modality_mask=modality_mask,         # [B, 4]
        )
        # fused: [B, embed_dim]
        output["attn_weights"] = attn_weights

        # ============================================================
        # 4. Classification
        # ============================================================
        logits = self.classifier(fused)              # [B, n_classes]
        probs  = torch.softmax(logits, dim=-1)       # [B, n_classes]
        preds  = torch.argmax(probs, dim=-1)         # [B]

        output["logits"]       = logits
        output["probabilities"] = probs
        output["pred_class"]   = preds

        # ============================================================
        # 5. Uncertainty Estimation (inference only)
        # ============================================================
        if return_uncertainty and self.use_uncertainty:
            with torch.no_grad():
                if self.uncertainty_method == "mc_dropout":
                    unc_output = self.uncertainty_head(fused)
                elif self.uncertainty_method == "evidential":
                    unc_output = self.uncertainty_head(fused)
            output["uncertainty"] = unc_output

        # ============================================================
        # 6. Optional Intermediate Features (for Grad-CAM, transfer)
        # ============================================================
        if return_features:
            output["fused_features"]      = fused
            output["modality_features"]   = mri_token_seq  # [B, 4, embed]
            output["modality_hiddens"]    = modality_hidden_states

        return output

    # ----------------------------------------------------------
    # Convenience Methods
    # ----------------------------------------------------------

    def predict(
        self,
        volumes: torch.Tensor,
        clinical: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        uncertainty: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-mode prediction with optional uncertainty.

        Puts model in eval mode, disables grad, runs forward pass.

        Args:
            volumes: [B, 4, 1, D, H, W]
            clinical: [B, n_features]
            modality_mask: [B, 4]
            uncertainty: Whether to estimate uncertainty

        Returns:
            Prediction dict (see forward())
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            result = self.forward(
                volumes=volumes,
                clinical=clinical,
                modality_mask=modality_mask,
                return_uncertainty=uncertainty,
                return_features=True,
            )

        if was_training:
            self.train()

        return result

    def freeze_encoder(self):
        """Freeze all MRI encoder weights (for fine-tuning head only)."""
        for enc in self.mri_encoders:
            for p in enc.parameters():
                p.requires_grad = False
        logger.info("[MSNet] MRI encoders frozen.")

    def unfreeze_encoder(self):
        """Unfreeze all MRI encoder weights."""
        for enc in self.mri_encoders:
            for p in enc.parameters():
                p.requires_grad = True
        logger.info("[MSNet] MRI encoders unfrozen.")

    def get_num_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
