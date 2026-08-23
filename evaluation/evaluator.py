"""
MSNet — Full Test Set Evaluator
================================
Loads a trained checkpoint, runs inference on the test set,
computes all publication-quality metrics with 95% bootstrap CIs,
and generates Grad-CAM visualizations.

Output directory:
  outputs/evaluation/
    ├── metrics.json            # All metrics + CIs
    ├── eval_report.txt         # Formatted report
    ├── confusion_matrix.png    # Confusion matrix figure
    ├── roc_curves.png          # Per-class ROC curves
    ├── pr_curves.png           # Precision-recall curves
    └── gradcam/                # Grad-CAM visualizations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from models.msnet import MSNet
from evaluation.metrics import (
    compute_all_metrics,
    bootstrap_confidence_intervals,
    generate_evaluation_report,
)
from explainability.gradcam import GradCAM3D, GradCAMVisualizer

logger = logging.getLogger(__name__)


# ============================================================
# Evaluator
# ============================================================

class MSNetEvaluator:
    """
    Full evaluation pipeline for trained MSNet model.

    Args:
        model: Trained MSNet
        config: Full config dict
        device: Computation device
        output_dir: Directory for saving outputs
    """

    def __init__(
        self,
        model: MSNet,
        config: dict,
        device: torch.device,
        output_dir: str = "outputs/evaluation/",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = config["data"]["num_classes"]
        self.class_names = config["data"]["class_names"]
        self.task = config["data"]["task"]

        # Grad-CAM infrastructure
        self.gradcam = GradCAM3D(model, device)
        self.gradcam_vis = GradCAMVisualizer(
            output_dir=str(self.output_dir / "gradcam"),
            colormap=config["explainability"]["gradcam"].get("colormap", "jet"),
            alpha=config["explainability"]["gradcam"].get("alpha", 0.5),
        )

    # ----------------------------------------------------------
    # Inference over Dataset
    # ----------------------------------------------------------

    @torch.no_grad()
    def run_inference(
        self, loader: DataLoader
    ) -> Dict[str, List]:
        """
        Run inference on an entire DataLoader.

        Returns dict with:
          - 'labels': true class labels
          - 'preds': predicted class labels
          - 'probs': softmax probabilities
          - 'subject_ids': subject identifiers
          - 'uncertainties': epistemic uncertainty estimates
        """
        self.model.eval()
        labels, preds, probs = [], [], []
        subject_ids = []
        uncertainties = []

        for batch in tqdm(loader, desc="[Evaluation] Inference"):
            volumes = batch["volumes"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            mod_mask = batch["modality_mask"].to(self.device)
            label = batch["label"]

            output = self.model.predict(
                volumes=volumes,
                clinical=clinical,
                modality_mask=mod_mask,
                uncertainty=self.config["model"]["uncertainty"]["enabled"],
            )

            preds.extend(output["pred_class"].cpu().tolist())
            labels.extend(label.tolist())
            probs.extend(output["probabilities"].cpu().numpy().tolist())
            subject_ids.extend(batch.get("subject_id", ["unknown"] * volumes.shape[0]))

            if "uncertainty" in output:
                unc = output["uncertainty"]
                uncertainties.extend(
                    unc.get("epistemic_uncertainty", torch.zeros(volumes.shape[0]))
                    .cpu().tolist()
                )

        return {
            "labels": labels,
            "preds": preds,
            "probs": probs,
            "subject_ids": subject_ids,
            "uncertainties": uncertainties,
        }

    # ----------------------------------------------------------
    # Metric Computation
    # ----------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        save_gradcam: bool = True,
        n_gradcam_samples: int = 5,
        bootstrap_n: int = 1000,
    ) -> Dict:
        """
        Full evaluation pipeline.

        Args:
            loader: Test DataLoader
            save_gradcam: Whether to generate Grad-CAM visualizations
            n_gradcam_samples: Number of subjects to visualize
            bootstrap_n: Number of bootstrap replicates

        Returns:
            Complete results dict
        """
        logger.info("[Evaluator] Running test set inference...")
        results = self.run_inference(loader)

        labels = results["labels"]
        preds  = results["preds"]
        probs  = results["probs"]

        # ---- Point Estimates ----
        metrics = compute_all_metrics(labels, preds, self.num_classes, probs)
        logger.info(f"\n[Evaluator] Point Estimates: {metrics}")

        # ---- Bootstrap CIs ----
        logger.info(f"[Evaluator] Computing {bootstrap_n} bootstrap CIs...")
        ci = bootstrap_confidence_intervals(
            labels, preds, self.num_classes, probs, n_bootstraps=bootstrap_n
        )

        # ---- Generate Text Report ----
        report = generate_evaluation_report(
            labels=labels,
            preds=preds,
            num_classes=self.num_classes,
            class_names=self.class_names,
            probs=probs,
            n_bootstraps=bootstrap_n,
            output_path=str(self.output_dir / "eval_report.txt"),
        )

        # ---- Save JSON Metrics ----
        full_results = {
            "metrics": metrics,
            "confidence_intervals": ci,
            "n_subjects": len(labels),
            "class_names": self.class_names,
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        # ---- Visualizations ----
        self._plot_confusion_matrix(labels, preds)
        self._plot_roc_curves(labels, probs)
        self._plot_pr_curves(labels, probs)

        # ---- Grad-CAM ----
        if save_gradcam and self.config["explainability"]["gradcam"]["enabled"]:
            logger.info(f"[Evaluator] Generating Grad-CAM for {n_gradcam_samples} subjects")
            self._generate_gradcam_samples(loader, n_gradcam_samples, preds, labels)

        return full_results

    # ----------------------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------------------

    def _plot_confusion_matrix(self, labels: List[int], preds: List[int]):
        """Save a styled confusion matrix figure."""
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f0f0f")
        fig.suptitle("Confusion Matrix", color="white", fontsize=16, fontweight="bold")

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_normalized],
            ["Counts", "Normalized"],
            [".0f", ".2f"],
        ):
            sns.heatmap(
                data,
                annot=True, fmt=fmt,
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cmap="Blues" if title == "Counts" else "YlOrRd",
                ax=ax,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title(title, color="white", fontsize=12)
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("True", color="white")
            ax.tick_params(colors="white")
            ax.set_facecolor("#1a1a1a")

        fig.patch.set_facecolor("#0f0f0f")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info("[Evaluator] Confusion matrix saved.")

    # ----------------------------------------------------------
    # ROC Curves
    # ----------------------------------------------------------

    def _plot_roc_curves(self, labels: List[int], probs: List[List[float]]):
        """Save per-class OvR ROC curves."""
        labels_arr = np.array(labels)
        probs_arr  = np.array(probs)
        n_classes  = self.num_classes

        labels_bin = label_binarize(labels_arr, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        colors = plt.cm.Set1(np.linspace(0, 0.9, n_classes))

        for c, (cls, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(labels_bin[:, c], probs_arr[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=color, lw=2, alpha=0.9,
                label=f"{cls} (AUC = {roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", color="white", fontsize=12)
        ax.set_title("One-vs-Rest ROC Curves", color="white", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", facecolor="#2a2a2a", edgecolor="white",
                  labelcolor="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")

        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves.png", dpi=150,
                    bbox_inches="tight", facecolor="#0f0f0f")
        plt.close()
        logger.info("[Evaluator] ROC curves saved.")

    # ----------------------------------------------------------
    # PR Curves
    # ----------------------------------------------------------

    def _plot_pr_curves(self, labels: List[int], probs: List[List[float]]):
        """Save per-class precision-recall curves."""
        labels_arr = np.array(labels)
        probs_arr  = np.array(probs)
        labels_bin = label_binarize(labels_arr, classes=list(range(self.num_classes)))

        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        colors = plt.cm.Set2(np.linspace(0, 0.9, self.num_classes))

        for c, (cls, color) in enumerate(zip(self.class_names, colors)):
            prec, rec, _ = precision_recall_curve(labels_bin[:, c], probs_arr[:, c])
            ap = average_precision_score(labels_bin[:, c], probs_arr[:, c])
            ax.plot(
                rec, prec,
                color=color, lw=2, alpha=0.9,
                label=f"{cls} (AP = {ap:.3f})",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Sensitivity)", color="white", fontsize=12)
        ax.set_ylabel("Precision", color="white", fontsize=12)
        ax.set_title("Precision-Recall Curves", color="white", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", facecolor="#2a2a2a", edgecolor="white",
                  labelcolor="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")

        plt.tight_layout()
        plt.savefig(self.output_dir / "pr_curves.png", dpi=150,
                    bbox_inches="tight", facecolor="#0f0f0f")
        plt.close()
        logger.info("[Evaluator] PR curves saved.")

    # ----------------------------------------------------------
    # Grad-CAM Sample Generation
    # ----------------------------------------------------------

    def _generate_gradcam_samples(
        self,
        loader: DataLoader,
        n_samples: int,
        preds: List[int],
        labels: List[int],
    ):
        """Generate Grad-CAM for a subset of test samples."""
        count = 0
        for batch in loader:
            if count >= n_samples:
                break

            volumes = batch["volumes"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            mod_mask = batch["modality_mask"].to(self.device)

            # Process one sample at a time
            for b in range(min(volumes.shape[0], n_samples - count)):
                vol_b   = volumes[b:b+1]
                clin_b  = clinical[b:b+1]
                mask_b  = mod_mask[b:b+1]
                sid     = str(batch.get("subject_id", [f"subject_{count}"])[b])
                pred_b  = preds[count] if count < len(preds) else 0

                try:
                    # Compute Grad-CAM for all modalities
                    cams = self.gradcam.compute_all_modalities(
                        vol_b, clin_b, mask_b, target_class=pred_b
                    )

                    # Volume for visualization: [4, D, H, W]
                    vol_np = vol_b[0, :, 0, :, :, :].cpu().numpy()  # [4, D, H, W]

                    # Multi-modal panel
                    self.gradcam_vis.save_multimodal_panel(
                        volumes=vol_np,
                        cams=cams,
                        subject_id=sid,
                        target_class=pred_b,
                        class_names=self.class_names,
                    )

                    # Per-modality axial slices
                    for modality, cam in cams.items():
                        mod_idx = ["T1", "T2", "FLAIR", "GD_T1"].index(modality)
                        vol_mod = vol_b[0, mod_idx, 0, :, :, :].cpu().numpy()
                        self.gradcam_vis.save_axial_slices(
                            volume=vol_mod,
                            cam=cam,
                            subject_id=sid,
                            modality=modality,
                            target_class=pred_b,
                            class_names=self.class_names,
                        )
                    count += 1

                except Exception as e:
                    logger.error(f"[Evaluator] Grad-CAM failed for {sid}: {e}")
                    count += 1

        logger.info(f"[Evaluator] Grad-CAM saved for {count} subjects.")
