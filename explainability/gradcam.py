"""
MSNet — 3D Grad-CAM Explainability Engine
==========================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
adapted for 3D volumetric MRI data.

For each MRI modality and each predicted class:
  1. Forward pass to get classification logits
  2. Compute gradients of target class score w.r.t. target conv layer
  3. Global average pool gradients → per-channel importance weights
  4. Weighted sum of activation maps → 3D saliency map
  5. ReLU + normalize → [0, 1] range
  6. Upsample to input resolution
  7. Overlay on original MRI slice

Produces:
  - Per-modality 3D saliency volumes
  - Axial/coronal/sagittal 2D projections
  - Lesion-highlighted MRI slices (publication-ready)

Reference:
  Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization", ICCV 2017.

  Zhou et al. (2016) — "Learning Deep Features for Discriminative
  Localization", CVPR 2016.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

MODALITIES = ["T1", "T2", "FLAIR", "GD_T1"]


# ============================================================
# Hook-Based Activation & Gradient Capture
# ============================================================

class GradCAMHook:
    """
    Registers forward and backward hooks on a target layer to
    capture feature activations and their gradients.

    Args:
        layer: The nn.Module layer to hook
    """

    def __init__(self, layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._gradient_handle = None

        self._fwd_hook = layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Store forward activations."""
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a tensor")
        self.activations = output.detach()
        if output.requires_grad:
            self._gradient_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        """Store gradients with respect to the hooked layer's output."""
        self.gradients = gradient.detach()

    def remove(self):
        """Remove hooks to free memory."""
        self._fwd_hook.remove()
        if self._gradient_handle is not None:
            self._gradient_handle.remove()


# ============================================================
# 3D Grad-CAM
# ============================================================

class GradCAM3D:
    """
    3D Grad-CAM for volumetric MRI feature visualization.

    Produces a 3D saliency map showing which voxels most influenced
    the model's classification decision for a given MRI modality.
    These maps directly correspond to lesion locations in MS.

    Args:
        model: Trained MSNet model
        target_layer_name: Name of the target CNN/Transformer layer
            For Swin: typically 'mri_encoders[i].swin.encoder.layers[-1]'
            For ResNet3D: typically 'mri_encoders[i].stage4[-1].conv3'
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_layer_name: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name

    def _get_target_layer(self, modality_idx: int) -> nn.Module:
        """
        Retrieve the target layer for Grad-CAM hook.

        Automatically selects the last conv block of the CNN encoder
        or the last Swin encoder block.
        """
        if self.target_layer_name is not None:
            layer_name = self.target_layer_name.format(
                modality_idx=modality_idx
            )
            return self.model.get_submodule(layer_name)

        encoders = self.model.mri_encoders
        if not 0 <= modality_idx < len(encoders):
            raise ValueError(f"Invalid modality index: {modality_idx}")
        encoder = encoders[modality_idx]
        encoder_type = self.model.encoder_type

        if encoder_type == "resnet3d":
            target = encoder.stage4[-1].conv3
        elif encoder_type == "swin_transformer":
            target = encoder.swin.swinViT.layers[-1].blocks[-1].norm2
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        return target

    def compute_gradcam(
        self,
        volumes: torch.Tensor,
        clinical: torch.Tensor,
        modality_mask: torch.Tensor,
        target_class: Optional[int] = None,
        modality_idx: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute 3D Grad-CAM saliency map for one MRI modality.

        Args:
            volumes: [1, n_modalities, 1, D, H, W] — single sample
            clinical: [1, n_features]
            modality_mask: [1, n_modalities]
            target_class: Class to explain (None = predicted class)
            modality_idx: Index of modality to explain (0=T1, 1=T2, ...)

        Returns:
            (cam_3d, target_class):
              cam_3d: [D, H, W] normalized saliency map [0, 1]
              target_class: The class that was explained
        """
        if volumes.ndim != 6 or volumes.shape[0] != 1:
            raise ValueError(
                "Grad-CAM expects one input shaped [1, modalities, 1, D, H, W]"
            )
        if clinical.ndim != 2 or clinical.shape[0] != 1:
            raise ValueError("Grad-CAM expects clinical input shaped [1, features]")
        if modality_mask.ndim != 2 or modality_mask.shape != volumes.shape[:2]:
            raise ValueError("Modality mask must match [batch, modalities]")
        if not 0 <= modality_idx < volumes.shape[1]:
            raise ValueError(f"Invalid modality index: {modality_idx}")

        target_layer = self._get_target_layer(modality_idx)
        was_training = self.model.training
        self.model.eval()
        volumes = volumes.to(self.device)
        clinical = clinical.to(self.device)
        modality_mask = modality_mask.to(self.device)

        hook = GradCAMHook(target_layer)

        try:
            with torch.enable_grad():
                output = self.model(
                    volumes=volumes,
                    clinical=clinical,
                    modality_mask=modality_mask,
                    return_features=False,
                )
                logits = output["logits"]
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("Model logits must have shape [1, classes]")
                if target_class is None:
                    target_index = int(logits.argmax(dim=-1).item())
                else:
                    target_index = int(target_class)
                if not 0 <= target_index < logits.shape[1]:
                    raise ValueError(f"Invalid target class: {target_index}")

                self.model.zero_grad(set_to_none=True)
                logits[0, target_index].backward()

            activations = hook.activations
            gradients = hook.gradients
            if activations is None or gradients is None:
                raise RuntimeError(
                    "Grad-CAM could not capture layer activations and gradients"
                )

            if activations.ndim == 3:
                token_count = activations.shape[1]
                side = round(token_count ** (1 / 3))
                if side ** 3 != token_count:
                    raise ValueError(
                        "Swin target layer tokens are not a cubic 3D grid; "
                        "choose a spatial target layer for Grad-CAM"
                    )
                activations = activations.reshape(
                    1, side, side, side, activations.shape[-1]
                ).permute(0, 4, 1, 2, 3)
                gradients = gradients.reshape(
                    1, side, side, side, gradients.shape[-1]
                ).permute(0, 4, 1, 2, 3)
            elif activations.ndim == 5 and self.model.encoder_type == "swin_transformer":
                activations = activations.permute(0, 4, 1, 2, 3)
                gradients = gradients.permute(0, 4, 1, 2, 3)
            elif activations.ndim != 5:
                raise ValueError(
                    f"Expected 3D feature map or token tensor, got "
                    f"activation shape {tuple(activations.shape)}"
                )

            weights = gradients.mean(dim=(0, 2, 3, 4))
            cam = F.relu(
                (weights[None, :, None, None, None] * activations).sum(dim=1)
            )
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=volumes.shape[-3:],
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            cam = cam.detach().cpu().numpy().astype(np.float32, copy=False)

        finally:
            hook.remove()
            self.model.train(was_training)

        return cam, target_index

    def compute_all_modalities(
        self,
        volumes: torch.Tensor,
        clinical: torch.Tensor,
        modality_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Grad-CAM for all present MRI modalities.

        Args:
            volumes: [1, 4, 1, D, H, W]
            clinical: [1, n_features]
            modality_mask: [1, 4]
            target_class: Class to explain

        Returns:
            Dict: {modality_name: cam_3d [D, H, W]}
        """
        cams = {}
        for i, modality in enumerate(MODALITIES):
            if i >= modality_mask.shape[1]:
                break
            if modality_mask[0, i].item() < 0.5:
                logger.debug(f"[GradCAM] Skipping absent modality: {modality}")
                continue
            cam, cls = self.compute_gradcam(
                volumes, clinical, modality_mask,
                target_class=target_class,
                modality_idx=i,
            )
            cams[modality] = cam
            logger.info(
                f"[GradCAM] {modality} → class {cls}, "
                f"max_activation={cam.max():.4f}"
            )

        if not cams:
            raise ValueError("Grad-CAM requires at least one present MRI modality")
        return cams


# ============================================================
# Visualization
# ============================================================

class GradCAMVisualizer:
    """
    Renders and saves 3D Grad-CAM saliency maps as visualizations.

    Produces:
      - Axial slices (most relevant for MS lesion visualization)
      - Coronal and sagittal projections
      - Multi-modality comparison panels

    Args:
        output_dir: Directory to save visualizations
        colormap: Matplotlib colormap name ('jet' or 'hot')
        alpha: Overlay transparency (0=transparent, 1=opaque)
    """

    def __init__(
        self,
        output_dir: str = "outputs/gradcam",
        colormap: str = "jet",
        alpha: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colormap = colormap
        self.alpha = alpha

    def save_axial_slices(
        self,
        volume: np.ndarray,
        cam: np.ndarray,
        subject_id: str,
        modality: str,
        target_class: int,
        class_names: List[str],
        n_slices: int = 5,
    ):
        """
        Save axial slice visualizations with CAM overlay.

        Selects n_slices evenly-spaced axial slices from the volume,
        overlays the Grad-CAM heatmap, and saves a composite figure.

        Args:
            volume: [D, H, W] — original MRI volume (normalized)
            cam: [D, H, W] — Grad-CAM saliency map [0, 1]
            subject_id: Subject identifier for filename
            modality: MRI modality name
            target_class: Predicted class index
            class_names: Human-readable class names
            n_slices: Number of axial slices to show
        """
        D = volume.shape[0]
        slice_indices = np.linspace(D * 0.2, D * 0.8, n_slices, dtype=int)

        fig, axes = plt.subplots(
            2, n_slices,
            figsize=(4 * n_slices, 8),
            facecolor="black",
        )
        fig.suptitle(
            f"Grad-CAM | Subject: {subject_id} | {modality} | "
            f"Predicted: {class_names[target_class]}",
            color="white", fontsize=14, fontweight="bold",
        )

        for j, sl_idx in enumerate(slice_indices):
            img_slice = volume[sl_idx, :, :]   # [H, W]
            cam_slice = cam[sl_idx, :, :]      # [H, W]

            # Normalize image slice for display
            img_disp = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            # ---- Row 0: Original MRI slice ----
            axes[0, j].imshow(img_disp, cmap="gray", aspect="equal")
            axes[0, j].set_title(f"Axial z={sl_idx}", color="white", fontsize=9)
            axes[0, j].axis("off")

            # ---- Row 1: Grad-CAM overlay ----
            axes[1, j].imshow(img_disp, cmap="gray", aspect="equal")
            axes[1, j].imshow(
                cam_slice,
                cmap=self.colormap,
                alpha=self.alpha * cam_slice,  # Adaptive alpha based on saliency
                vmin=0, vmax=1,
                aspect="equal",
            )
            axes[1, j].axis("off")

        plt.tight_layout()

        fname = self.output_dir / f"{subject_id}_{modality}_gradcam_axial.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close()
        logger.info(f"[GradCAM] Saved axial visualization → {fname}")

    def save_multimodal_panel(
        self,
        volumes: np.ndarray,           # [4, D, H, W]
        cams: Dict[str, np.ndarray],   # {modality: cam [D, H, W]}
        subject_id: str,
        slice_idx: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        target_class: int = 0,
    ):
        """
        Save a multi-modality Grad-CAM panel (all modalities side-by-side).

        Args:
            volumes: [4, D, H, W] — all MRI modalities
            cams: Dict of modality → CAM maps
            subject_id: Subject ID
            slice_idx: Axial slice to display (None = mid-brain)
            class_names: Class name list
            target_class: Target class index
        """
        D = volumes.shape[1]
        if slice_idx is None:
            slice_idx = D // 2

        n_mod = len(MODALITIES)
        fig, axes = plt.subplots(
            2, n_mod,
            figsize=(5 * n_mod, 10),
            facecolor="#0d0d0d",
        )

        cls_name = (class_names[target_class]
                    if class_names and target_class < len(class_names)
                    else str(target_class))

        fig.suptitle(
            f"Multi-Modal Grad-CAM | {subject_id} | Predicted: {cls_name}",
            color="white", fontsize=16, fontweight="bold", y=0.98,
        )

        for i, modality in enumerate(MODALITIES):
            vol_slice = volumes[i, slice_idx, :, :]  # [H, W]
            img_disp = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)

            # ---- Top row: original ----
            axes[0, i].imshow(img_disp, cmap="gray")
            axes[0, i].set_title(modality, color="cyan", fontsize=12, fontweight="bold")
            axes[0, i].axis("off")

            # ---- Bottom row: Grad-CAM overlay ----
            axes[1, i].imshow(img_disp, cmap="gray")
            if modality in cams:
                cam_slice = cams[modality][slice_idx, :, :]
                axes[1, i].imshow(
                    cam_slice,
                    cmap=self.colormap,
                    alpha=self.alpha,
                    vmin=0, vmax=1,
                )
            else:
                axes[1, i].set_facecolor("#1a1a1a")
                axes[1, i].text(
                    0.5, 0.5, "N/A", color="white",
                    ha="center", va="center", transform=axes[1, i].transAxes,
                )
            axes[1, i].axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=self.colormap, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1, :], fraction=0.02, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        cbar.set_label("Saliency", color="white")

        plt.tight_layout()
        fname = self.output_dir / f"{subject_id}_multimodal_gradcam.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()
        logger.info(f"[GradCAM] Saved multi-modal panel → {fname}")

    def save_modality_saliency_chart(
        self,
        cams: Dict[str, np.ndarray],
        subject_id: str,
    ) -> Path:
        """Save an SVG bar chart of mean normalized saliency per modality."""
        if not cams:
            raise ValueError("Cannot plot modality saliency without Grad-CAM maps")

        modalities = list(cams)
        mean_saliency = []
        for modality in modalities:
            cam = np.asarray(cams[modality])
            if cam.ndim != 3 or cam.size == 0 or not np.isfinite(cam).all():
                raise ValueError(
                    f"Grad-CAM map for {modality} must be a finite 3D volume"
                )
            if cam.min() < 0 or cam.max() > 1:
                raise ValueError(
                    f"Grad-CAM map for {modality} must be normalized to [0, 1]"
                )
            mean_saliency.append(float(cam.mean()))

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        bars = ax.bar(modalities, mean_saliency, color="#2878B5")
        ax.set_title(f"Mean Grad-CAM Saliency by Modality\nSubject: {subject_id}")
        ax.set_xlabel("MRI modality")
        ax.set_ylabel("Mean normalized saliency")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.text(
            0.5,
            0.01,
            "Maps are normalized independently; values describe saliency spread, "
            "not relative modality importance.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))

        path = self.output_dir / f"{subject_id}_modality_saliency.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[GradCAM] Saved modality saliency chart → {path}")
        logger.info(
            f"[GradCAM] Saved modality saliency chart preview → "
            f"{path.with_suffix('.png')}"
        )
        return path
