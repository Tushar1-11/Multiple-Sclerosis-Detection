"""
MSNet — Clinical Inference Engine
===================================
Single-subject inference pipeline for deployment.

Features:
  - Loads a trained MSNet checkpoint
  - Preprocesses raw NIfTI MRI files on-the-fly
  - Accepts clinical metadata as JSON or CLI args
  - Returns structured prediction JSON with:
      * Predicted MS class
      * Per-class probabilities
      * Epistemic uncertainty (MC Dropout)
      * Grad-CAM saliency paths
      * Clinical feature importance

Usage:
  python infer.py \
    --t1 /path/to/t1.nii.gz \
    --t2 /path/to/t2.nii.gz \
    --flair /path/to/flair.nii.gz \
    --gd_t1 /path/to/gd_t1.nii.gz \
    --clinical "age=45,gender=1,edss=3.0,disease_duration=5,igg_index=0.8,oligoclonal_bands=1,t_cell_count=650" \
    --checkpoint outputs/best_model.pth \
    --config configs/default_config.yaml \
    --output_dir outputs/patient_001/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import nibabel as nib
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.msnet import MSNet
from data.preprocessing.mri_preprocessor import MRIPreprocessingPipeline
from data.dataset import build_val_transforms
from explainability.gradcam import GradCAM3D, GradCAMVisualizer
from utils.logger import setup_logging
from utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

MODALITIES = ["T1", "T2", "FLAIR", "GD_T1"]
CLASS_NAMES = ["HC", "RRMS", "SPMS", "PPMS"]


# ============================================================
# Inference Predictor
# ============================================================

class MSNetPredictor:
    """
    Real-time inference engine for clinical deployment.

    Handles:
      - Light on-the-fly NIfTI preprocessing
      - Missing modality graceful handling
      - MC Dropout uncertainty estimation
      - Grad-CAM saliency map generation
      - Structured JSON report generation

    Args:
        config: Config dict
        checkpoint_path: Path to trained .pth checkpoint
        device: 'cuda' or 'cpu'
        output_dir: Directory for saving report + Grad-CAMs
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        device: str = "cuda",
        output_dir: str = "outputs/inference/",
    ):
        self.config = config
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = config["data"]["num_classes"]
        self.class_names = config["data"]["class_names"]
        self.clinical_features = config["data"]["clinical_features"]
        self.spatial_size = tuple(config["data"]["spatial_size"])

        # ---- Load Model ----
        logger.info(f"[Predictor] Loading model from: {checkpoint_path}")
        self.model = MSNet(config)
        load_checkpoint(self.model, checkpoint_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        # ---- MONAI Transform (val/test mode) ----
        self.transform = build_val_transforms(
            spatial_size=self.spatial_size,
            voxel_spacing=tuple(config["data"]["voxel_spacing"]),
        )

        # ---- Grad-CAM ----
        if config["explainability"]["gradcam"]["enabled"]:
            self.gradcam = GradCAM3D(self.model, self.device)
            self.gradcam_vis = GradCAMVisualizer(
                output_dir=str(self.output_dir / "gradcam"),
                colormap=config["explainability"]["gradcam"].get("colormap", "jet"),
                alpha=config["explainability"]["gradcam"].get("alpha", 0.5),
            )
        else:
            self.gradcam = None

        logger.info(f"[Predictor] Ready. Device={self.device}")

    def preprocess_mri(
        self,
        modality_paths: Dict[str, Optional[str]],
    ) -> torch.Tensor:
        """
        Preprocess NIfTI volumes using MONAI transforms.

        Args:
            modality_paths: Dict {modality: path or None}

        Returns:
            Tensor [1, n_modalities, 1, D, H, W]  (batch=1)
        """
        dummy_shape = (1, *self.spatial_size)  # [1, D, H, W]
        volumes = []
        modality_mask = []

        # Build MONAI-compatible data dict
        data_dict = {
            mod: path for mod, path in modality_paths.items()
            if path and Path(path).exists()
        }

        transformed = self.transform(data_dict)

        for mod in MODALITIES:
            if mod in transformed and transformed[mod] is not None:
                vol = transformed[mod]
                if vol.dim() == 3:
                    vol = vol.unsqueeze(0)
                volumes.append(vol.float())
                modality_mask.append(1.0)
            else:
                volumes.append(torch.zeros(dummy_shape, dtype=torch.float32))
                modality_mask.append(0.0)
                logger.warning(f"[Predictor] Missing modality: {mod} -> zero-filled")

        # Stack: [n_modalities, 1, D, H, W] → [1, n_modalities, 1, D, H, W]
        volume_stack = torch.stack(volumes, dim=0).unsqueeze(0)  # [1, 4, 1, D, H, W]
        mod_mask = torch.tensor(modality_mask, dtype=torch.float32).unsqueeze(0)  # [1, 4]

        return volume_stack, mod_mask

    def parse_clinical_features(
        self,
        clinical_str: str,
    ) -> torch.Tensor:
        """
        Parse clinical feature string into normalized tensor.

        Args:
            clinical_str: Comma-separated key=value pairs
                e.g., "age=45,gender=1,edss=3.0,..."

        Returns:
            Tensor [1, n_features]
        """
        # Default values for missing clinical features
        defaults = {
            "age": 35.0,
            "gender": 0.5,
            "edss": 0.0,
            "disease_duration": 0.0,
            "igg_index": 0.5,
            "oligoclonal_bands": 0.0,
            "t_cell_count": 800.0,
        }

        # Parse provided values
        for pair in clinical_str.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                defaults[k.strip()] = float(v.strip())

        # Build feature vector in config order
        features = [
            defaults.get(feat, 0.0)
            for feat in self.clinical_features
        ]

        logger.info(f"[Predictor] Clinical features: {dict(zip(self.clinical_features, features))}")
        return torch.tensor([features], dtype=torch.float32)

    @torch.no_grad()
    def predict(
        self,
        modality_paths: Dict[str, Optional[str]],
        clinical_str: str,
        subject_id: str = "patient",
        generate_gradcam: bool = True,
    ) -> Dict:
        """
        Full inference pipeline for a single subject.

        Args:
            modality_paths: Dict {modality: NIfTI path or None}
            clinical_str: Clinical feature string
            subject_id: Subject identifier
            generate_gradcam: Whether to generate Grad-CAM visualizations

        Returns:
            Structured prediction dict
        """
        start_time = time.time()
        logger.info(f"\n[Predictor] Processing: {subject_id}")

        # ---- Preprocess MRI ----
        volumes, mod_mask = self.preprocess_mri(modality_paths)
        volumes  = volumes.to(self.device)
        mod_mask = mod_mask.to(self.device)

        # ---- Parse Clinical Features ----
        clinical = self.parse_clinical_features(clinical_str).to(self.device)

        # ---- Model Forward Pass ----
        output = self.model.predict(
            volumes=volumes,
            clinical=clinical,
            modality_mask=mod_mask,
            uncertainty=True,
        )

        pred_class  = output["pred_class"].item()
        probs       = output["probabilities"][0].cpu().numpy()
        clinical_gate = output["clinical_gate"][0].cpu().numpy()

        # ---- Build Result ----
        result = {
            "subject_id": subject_id,
            "predicted_class": int(pred_class),
            "predicted_label": self.class_names[pred_class],
            "class_probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(self.num_classes)
            },
            "confidence": float(probs[pred_class]),
            "modality_availability": {
                mod: bool(mod_mask[0, i].item() > 0.5)
                for i, mod in enumerate(MODALITIES)
            },
            "clinical_feature_importance": {
                feat: float(clinical_gate[i])
                for i, feat in enumerate(self.clinical_features)
            },
            "inference_time_ms": round((time.time() - start_time) * 1000, 2),
        }

        # ---- Uncertainty Estimation ----
        if "uncertainty" in output:
            unc = output["uncertainty"]
            result["uncertainty"] = {
                "epistemic_uncertainty": float(
                    unc.get("epistemic_uncertainty", torch.tensor([0.0]))[0].item()
                ),
                "predictive_entropy": float(
                    unc.get("predictive_entropy", torch.tensor([0.0]))[0].item()
                ),
                "confidence_note": (
                    "Low confidence — consider clinical review"
                    if unc.get("epistemic_uncertainty", torch.tensor([0.0]))[0].item() > 0.1
                    else "High confidence prediction"
                ),
            }

        # ---- Grad-CAM ----
        gradcam_paths = {}
        if generate_gradcam and self.gradcam:
            logger.info("[Predictor] Generating Grad-CAM visualizations...")
            cams = self.gradcam.compute_all_modalities(
                volumes, clinical, mod_mask, target_class=pred_class
            )

            vol_np = volumes[0, :, 0, :, :, :].cpu().numpy()  # [4, D, H, W]

            # Multi-modal panel
            panel_path = self.output_dir / "gradcam" / f"{subject_id}_multimodal.png"
            self.gradcam_vis.save_multimodal_panel(
                volumes=vol_np,
                cams=cams,
                subject_id=subject_id,
                target_class=pred_class,
                class_names=self.class_names,
            )
            gradcam_paths["multimodal_panel"] = str(panel_path)

            # Per-modality axial views
            for mod, cam in cams.items():
                mod_idx = MODALITIES.index(mod)
                vol_mod = vol_np[mod_idx]
                self.gradcam_vis.save_axial_slices(
                    volume=vol_mod,
                    cam=cam,
                    subject_id=subject_id,
                    modality=mod,
                    target_class=pred_class,
                    class_names=self.class_names,
                )
                gradcam_paths[f"{mod}_axial"] = str(
                    self.output_dir / "gradcam" /
                    f"{subject_id}_{mod}_gradcam_axial.png"
                )

        result["gradcam_paths"] = gradcam_paths

        # ---- Save Report ----
        report_path = self.output_dir / f"{subject_id}_prediction.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"[Predictor] Report saved -> {report_path}")

        # ---- Print Summary ----
        logger.info(
            f"\n{'='*50}\n"
            f"Subject: {subject_id}\n"
            f"Prediction: {result['predicted_label']} "
            f"(confidence: {result['confidence']:.1%})\n"
            f"Probabilities:\n" +
            "\n".join(
                f"  {k}: {v:.1%}"
                for k, v in result["class_probabilities"].items()
            ) +
            f"\n{'='*50}"
        )

        return result


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MSNet Inference — Single Subject MS Prediction"
    )
    parser.add_argument("--t1",      default=None,  help="T1-weighted NIfTI path")
    parser.add_argument("--t2",      default=None,  help="T2-weighted NIfTI path")
    parser.add_argument("--flair",   default=None,  help="FLAIR NIfTI path")
    parser.add_argument("--gd_t1",   default=None,  help="Gadolinium-enhanced T1 NIfTI path")
    parser.add_argument(
        "--clinical",
        default="age=40,gender=0,edss=1.5,disease_duration=2,igg_index=0.7,oligoclonal_bands=1,t_cell_count=700",
        help="Clinical features as 'key=val,key=val,...'",
    )
    parser.add_argument(
        "--subject_id", default="patient_001",
        help="Subject identifier",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", default="configs/default_config.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--output_dir", default="outputs/inference/",
        help="Directory to save predictions and Grad-CAMs",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--no_gradcam", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging("INFO")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Build predictor
    predictor = MSNetPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Modality paths dict
    modality_paths = {
        "T1":    args.t1,
        "T2":    args.t2,
        "FLAIR": args.flair,
        "GD_T1": args.gd_t1,
    }

    # Run prediction
    result = predictor.predict(
        modality_paths=modality_paths,
        clinical_str=args.clinical,
        subject_id=args.subject_id,
        generate_gradcam=not args.no_gradcam,
    )

    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
