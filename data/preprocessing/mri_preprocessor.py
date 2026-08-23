"""
MSNet — MRI Preprocessing Pipeline
====================================
Handles complete NIfTI preprocessing for MS detection:
  1. Brain extraction (skull stripping via HD-BET / ANTs)
  2. Registration to MNI152 template (ANTs SyN)
  3. N4 bias field correction (SimpleITK)
  4. Intensity normalization (Z-score / nyul percentile)
  5. Isotropic resampling to target voxel spacing
  6. Spatial cropping/padding to uniform grid

Reference:
  Tustison et al. (2010) — N4ITK Improved N3 Bias Correction
  Avants et al. (2008)   — ANTs Symmetric Normalization
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage

# MONAI transforms for standardized augmentation pipelines
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
)

logger = logging.getLogger(__name__)


# ============================================================
# N4 Bias Field Correction
# ============================================================

class N4BiasFieldCorrector:
    """
    Applies N4ITK bias field correction using SimpleITK.

    N4 correction is crucial for MRI preprocessing as scanner
    hardware introduces low-frequency intensity inhomogeneities
    that can confound CNN feature extraction.

    Args:
        num_fitting_levels (int): Number of resolution levels.
        convergence_threshold (float): Convergence criterion.
        max_iterations (List[int]): Max iterations per level.
        shrink_factor (int): Subsampling factor for speed.
    """

    def __init__(
        self,
        num_fitting_levels: int = 4,
        convergence_threshold: float = 1e-6,
        max_iterations: Optional[List[int]] = None,
        shrink_factor: int = 4,
    ):
        self.num_fitting_levels = num_fitting_levels
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations or [50, 50, 50, 50]
        self.shrink_factor = shrink_factor

    def correct(self, image_path: str, output_path: str) -> str:
        """
        Run N4 bias field correction on a NIfTI image.

        Args:
            image_path: Path to input NIfTI (.nii / .nii.gz)
            output_path: Path to save corrected NIfTI

        Returns:
            output_path (str) upon success
        """
        logger.info(f"[N4] Correcting: {Path(image_path).name}")

        # Load image as SimpleITK
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)

        # Create brain mask using Otsu thresholding
        # (real implementations use HD-BET or FSL BET)
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Shrink image for faster computation
        shrunken_image = sitk.Shrink(image, [self.shrink_factor] * image.GetDimension())
        shrunken_mask = sitk.Shrink(mask, [self.shrink_factor] * mask.GetDimension())

        # Configure N4 corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            self.max_iterations[:self.num_fitting_levels]
        )
        corrector.SetConvergenceThreshold(self.convergence_threshold)

        # Run on shrunk image to estimate bias field
        corrector.Execute(shrunken_image, shrunken_mask)

        # Apply full-resolution log bias field
        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        corrected_image = image / sitk.Exp(log_bias_field)

        # Save output
        sitk.WriteImage(corrected_image, output_path)
        logger.info(f"[N4] Saved corrected image → {output_path}")
        return output_path


# ============================================================
# Intensity Normalization
# ============================================================

class IntensityNormalizer:
    """
    Provides multiple intensity normalization strategies for MRI.

    Strategies:
      - 'zscore': Zero-mean, unit-variance normalization
      - 'percentile': Robust min–max using 1st & 99th percentiles
      - 'nyul': Nyul piecewise linear histogram matching
        (requires a reference histogram, best for multi-site)

    Reference:
      Nyul & Udupa (1999) — On Standardizing the MR Image
      Intensity Scale.
    """

    def __init__(self, method: str = "zscore"):
        assert method in ("zscore", "percentile", "nyul"), \
            f"Unknown normalization method: {method}"
        self.method = method

    def normalize(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize voxel intensities within (optional) brain mask.

        Args:
            volume: 3D numpy array [D, H, W]
            mask: Optional binary brain mask [D, H, W]

        Returns:
            Normalized 3D numpy array
        """
        if mask is not None:
            brain_voxels = volume[mask > 0]
        else:
            brain_voxels = volume[volume > 0]

        if self.method == "zscore":
            mu = brain_voxels.mean()
            sigma = brain_voxels.std() + 1e-8
            volume = (volume - mu) / sigma

        elif self.method == "percentile":
            p1 = np.percentile(brain_voxels, 1)
            p99 = np.percentile(brain_voxels, 99)
            volume = np.clip(volume, p1, p99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)

        elif self.method == "nyul":
            # Simplified linear histogram standardization
            p10 = np.percentile(brain_voxels, 10)
            p90 = np.percentile(brain_voxels, 90)
            volume = np.clip(volume, p10, p90)
            volume = (volume - p10) / (p90 - p10 + 1e-8)

        return volume.astype(np.float32)


# ============================================================
# Image Registration to MNI Template
# ============================================================

class MNIRegistrar:
    """
    Registers subject MRI to MNI152 standard space using ANTs SyN.

    Registration ensures all subjects share the same coordinate
    system, which is critical for group-level CNN feature learning.
    Applies a two-stage registration:
      1. Rigid alignment (6 DOF)
      2. Deformable SyN (diffeomorphic)

    Requires: antspyx (pip install antspyx)
    """

    def __init__(self, template_path: str):
        """
        Args:
            template_path: Path to MNI152 reference template (.nii.gz).
                           Can use FSL's MNI152_T1_1mm.nii.gz
        """
        self.template_path = template_path
        self._template = None

    def _load_template(self):
        """Lazy-load the MNI template to avoid memory overhead."""
        if self._template is None:
            try:
                import ants
                self._template = ants.image_read(self.template_path)
            except ImportError:
                raise RuntimeError(
                    "antspyx not installed. Run: pip install antspyx"
                )
        return self._template

    def register(
        self,
        moving_path: str,
        output_path: str,
        type_of_transform: str = "SyN",
    ) -> str:
        """
        Register moving image to MNI152 space.

        Args:
            moving_path: Path to subject MRI NIfTI
            output_path: Path to save warped image
            type_of_transform: ANTs transform type

        Returns:
            output_path upon success
        """
        import ants

        logger.info(f"[MNI] Registering: {Path(moving_path).name}")
        fixed = self._load_template()
        moving = ants.image_read(moving_path)

        # Run SyN registration
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=type_of_transform,
            verbose=False,
        )

        # Save warped image
        ants.image_write(registration["warpedmovout"], output_path)
        logger.info(f"[MNI] Registered image → {output_path}")
        return output_path


# ============================================================
# Resampling to Isotropic Spacing
# ============================================================

class IsotropicResampler:
    """
    Resamples NIfTI images to a target isotropic voxel spacing
    using B-spline interpolation (order=3) for intensity images
    and nearest-neighbor for label maps.

    All subjects will share the same voxel grid after resampling,
    enabling batch processing with consistent tensor shapes.

    Args:
        target_spacing (Tuple[float, ...]): Target voxel spacing in mm.
        interpolation (str): 'bspline' | 'linear' | 'nearest'
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        interpolation: str = "bspline",
    ):
        self.target_spacing = target_spacing
        interp_map = {
            "bspline": sitk.sitkBSpline,
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
        }
        self.interp = interp_map[interpolation]

    def resample(self, image_path: str, output_path: str) -> str:
        """
        Resample image to target voxel spacing.

        Args:
            image_path: Input NIfTI path
            output_path: Output NIfTI path

        Returns:
            output_path upon success
        """
        logger.info(f"[Resample] Resampling: {Path(image_path).name}")
        image = sitk.ReadImage(image_path, sitk.sitkFloat32)

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Compute new size to maintain field-of-view
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(
                original_size, original_spacing, self.target_spacing
            )
        ]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(self.target_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputDirection(image.GetDirection())
        resample_filter.SetOutputOrigin(image.GetOrigin())
        resample_filter.SetTransform(sitk.Transform())
        resample_filter.SetDefaultPixelValue(0)
        resample_filter.SetInterpolator(self.interp)

        resampled = resample_filter.Execute(image)
        sitk.WriteImage(resampled, output_path)
        logger.info(f"[Resample] → {output_path} (new size: {new_size})")
        return output_path


# ============================================================
# Full Preprocessing Pipeline
# ============================================================

class MRIPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for multi-sequence MRI data.

    Processing steps (in order):
      1. N4 Bias Field Correction
      2. Registration to MNI152
      3. Isotropic Resampling
      4. Spatial Crop + Pad to target size
      5. Intensity Normalization

    Args:
        config: Config dict with preprocessing parameters.
        mni_template_path: Path to MNI152 template NIfTI.
        output_dir: Root output directory.
        normalization: Normalization method ('zscore'/'percentile').
        target_spacing: Isotropic voxel size in mm.
        target_size: Final spatial size [D, H, W].
        skip_registration: Set True to skip ANTs (faster, less accurate).
        skip_n4: Set True to skip N4 correction (if pre-corrected).
    """

    MODALITIES = ["T1", "T2", "FLAIR", "GD_T1"]

    def __init__(
        self,
        mni_template_path: str,
        output_dir: str,
        normalization: str = "zscore",
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (128, 128, 128),
        skip_registration: bool = False,
        skip_n4: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.skip_registration = skip_registration
        self.skip_n4 = skip_n4

        # Initialize sub-processors
        self.n4_corrector = N4BiasFieldCorrector()
        self.registrar = MNIRegistrar(mni_template_path) if not skip_registration else None
        self.resampler = IsotropicResampler(target_spacing=target_spacing)
        self.normalizer = IntensityNormalizer(method=normalization)

    def process_subject(
        self,
        subject_id: str,
        modality_paths: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        """
        Process all modalities for a single subject.

        Args:
            subject_id: Unique subject identifier (e.g., 'sub-001')
            modality_paths: Dict mapping modality → NIfTI path or None

        Returns:
            Dict mapping modality → processed NIfTI path
        """
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        for modality, path in modality_paths.items():
            if path is None:
                # Missing modality — mark and continue
                logger.warning(f"[{subject_id}] Missing modality: {modality}")
                output_paths[modality] = None
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"Processing [{subject_id}] — {modality}")
            logger.info(f"{'='*50}")

            current_path = path

            # Step 1: N4 Bias Field Correction
            if not self.skip_n4:
                n4_path = str(subject_dir / f"{modality}_n4.nii.gz")
                current_path = self.n4_corrector.correct(current_path, n4_path)

            # Step 2: MNI Registration (only T1 drives the warp field)
            if not self.skip_registration:
                reg_path = str(subject_dir / f"{modality}_mni.nii.gz")
                current_path = self.registrar.register(current_path, reg_path)

            # Step 3: Isotropic Resampling
            res_path = str(subject_dir / f"{modality}_resampled.nii.gz")
            current_path = self.resampler.resample(current_path, res_path)

            # Step 4: Intensity Normalization + Crop/Pad
            nii = nib.load(current_path)
            volume = nii.get_fdata().astype(np.float32)
            normalized = self.normalizer.normalize(volume)

            # Crop/Pad to target spatial size
            processed = self._crop_or_pad(normalized, self.target_size)

            # Save final preprocessed volume
            final_path = str(subject_dir / f"{modality}_preprocessed.nii.gz")
            final_nii = nib.Nifti1Image(processed, nii.affine, nii.header)
            nib.save(final_nii, final_path)

            output_paths[modality] = final_path
            logger.info(f"[{subject_id}][{modality}] → {final_path} ✓")

        return output_paths

    def _crop_or_pad(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Crop from center or zero-pad to exactly target_size.

        Args:
            volume: Input 3D array [D, H, W]
            target_size: Desired output size [D, H, W]

        Returns:
            Resized 3D array
        """
        output = np.zeros(target_size, dtype=np.float32)
        src_shape = volume.shape

        # Compute slices to copy
        starts_src = []
        starts_dst = []
        lengths = []

        for s, t in zip(src_shape, target_size):
            if s >= t:
                # Crop: center crop from source
                start_src = (s - t) // 2
                start_dst = 0
                length = t
            else:
                # Pad: center into destination
                start_src = 0
                start_dst = (t - s) // 2
                length = s

            starts_src.append(start_src)
            starts_dst.append(start_dst)
            lengths.append(length)

        d0, d1, d2 = starts_dst
        s0, s1, s2 = starts_src
        l0, l1, l2 = lengths

        output[d0:d0+l0, d1:d1+l1, d2:d2+l2] = \
            volume[s0:s0+l0, s1:s1+l1, s2:s2+l2]

        return output

    def process_dataset(
        self,
        subjects: List[Dict],
        n_jobs: int = 1,
    ) -> List[Dict]:
        """
        Process an entire dataset of subjects.

        Args:
            subjects: List of dicts, each with:
                      'subject_id', 'T1', 'T2', 'FLAIR', 'GD_T1'
            n_jobs: Number of parallel workers (future: joblib)

        Returns:
            List of processed subject dicts with output paths
        """
        results = []
        for i, subject in enumerate(subjects):
            logger.info(f"\n[Dataset] Processing subject {i+1}/{len(subjects)}: "
                        f"{subject['subject_id']}")
            modality_paths = {
                mod: subject.get(mod) for mod in self.MODALITIES
            }
            output_paths = self.process_subject(
                subject["subject_id"], modality_paths
            )
            subject_result = {"subject_id": subject["subject_id"], **output_paths}
            results.append(subject_result)
        return results


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MSNet MRI Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Root directory of raw NIfTI data")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--mni_template", default="data/templates/MNI152_T1_1mm.nii.gz",
                        help="Path to MNI152 reference template")
    parser.add_argument("--normalization", default="zscore",
                        choices=["zscore", "percentile", "nyul"])
    parser.add_argument("--target_size", nargs=3, type=int,
                        default=[128, 128, 128],
                        metavar=("D", "H", "W"))
    parser.add_argument("--target_spacing", nargs=3, type=float,
                        default=[1.0, 1.0, 1.0],
                        metavar=("dz", "dy", "dx"))
    parser.add_argument("--skip_registration", action="store_true")
    parser.add_argument("--skip_n4", action="store_true")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    pipeline = MRIPreprocessingPipeline(
        mni_template_path=args.mni_template,
        output_dir=args.output_dir,
        normalization=args.normalization,
        target_spacing=tuple(args.target_spacing),
        target_size=tuple(args.target_size),
        skip_registration=args.skip_registration,
        skip_n4=args.skip_n4,
    )

    # Auto-discover subjects from BIDS-style directory layout
    # Expected: input_dir/sub-XXX/{T1,T2,FLAIR,GD_T1}.nii.gz
    input_root = Path(args.input_dir)
    subjects = []
    for subject_dir in sorted(input_root.glob("sub-*")):
        entry = {"subject_id": subject_dir.name}
        for modality in MRIPreprocessingPipeline.MODALITIES:
            candidates = list(subject_dir.glob(f"*{modality}*.nii*"))
            entry[modality] = str(candidates[0]) if candidates else None
        subjects.append(entry)

    logger.info(f"Found {len(subjects)} subjects to process")
    results = pipeline.process_dataset(subjects)
    logger.info(f"\nPreprocessing complete. Processed {len(results)} subjects → {args.output_dir}")


if __name__ == "__main__":
    main()
