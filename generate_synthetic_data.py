"""
MSNet — Synthetic Dataset Generator
=====================================
Generates fake NIfTI MRI volumes and a metadata CSV so that
the full train.py pipeline can be validated on any machine
without real patient data.

Run:
    python generate_synthetic_data.py
"""

import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
N_SUBJECTS   = 40          # total subjects (train+val+test)
SPATIAL_SIZE = (64, 64, 64)  # small volumes → fast loading
CLASSES      = {0: "HC", 1: "RRMS", 2: "SPMS", 3: "PPMS"}
MODALITIES   = ["T1", "T2", "FLAIR", "GD_T1"]
OUT_DIR      = Path("data/synthetic")
CSV_PATH     = Path("data/metadata.csv")

GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def make_synthetic_volume(label: int, spatial_size=(64, 64, 64)) -> np.ndarray:
    """
    Create a synthetic MRI-like 3D volume.
    Each class gets slightly different intensity statistics
    so the model has a signal to learn.
    """
    base_intensity = 0.3 + label * 0.15          # class-specific mean
    noise          = np.random.randn(*spatial_size) * 0.1
    vol            = np.random.rand(*spatial_size) * base_intensity + noise

    # Add a small 'lesion-like' bright region for MS classes
    if label > 0:
        cx, cy, cz = [np.random.randint(16, s - 16) for s in spatial_size]
        r = np.random.randint(3, 8)
        x, y, z = np.ogrid[:spatial_size[0], :spatial_size[1], :spatial_size[2]]
        mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
        vol[mask] += 0.4 + label * 0.1

    return vol.astype(np.float32)


def save_nifti(data: np.ndarray, path: Path):
    """Save a 3D numpy array as a NIfTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)  # identity: 1mm isotropic
    img    = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def make_clinical(label: int) -> dict:
    """Generate realistic-ish clinical features per class."""
    rng = np.random
    age = rng.randint(20, 60) if label == 0 else rng.randint(25, 65)
    return {
        "age":              age,
        "gender":           rng.randint(0, 2),
        "edss":             round(rng.uniform(0.0, 1.0) if label == 0 else rng.uniform(1.5, 7.0), 1),
        "disease_duration": round(rng.uniform(0.0, 0.5) if label == 0 else rng.uniform(0.5, 20.0), 1),
        "igg_index":        round(rng.uniform(0.3, 0.7) if label == 0 else rng.uniform(0.6, 2.5), 2),
        "oligoclonal_bands":0 if label == 0 else rng.randint(0, 2),
        "t_cell_count":     rng.randint(400, 900) if label == 0 else rng.randint(200, 700),
    }


def main():
    print(f"\n{BOLD}{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}{CYAN}  MSNet — Synthetic Data Generator{RESET}")
    print(f"{BOLD}{CYAN}{'='*55}{RESET}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    labels_cycle = []
    per_class = N_SUBJECTS // len(CLASSES)
    for lbl in CLASSES:
        labels_cycle.extend([lbl] * per_class)
    # fill remainder
    while len(labels_cycle) < N_SUBJECTS:
        labels_cycle.append(0)
    np.random.shuffle(labels_cycle)

    print(f"  Generating {N_SUBJECTS} synthetic subjects …")
    for i, label in enumerate(labels_cycle):
        subj_id = f"sub-{i+1:03d}"
        subj_dir = OUT_DIR / subj_id

        row = {"subject_id": subj_id, "label": label}

        # Generate one NIfTI per modality
        for mod in MODALITIES:
            vol  = make_synthetic_volume(label, SPATIAL_SIZE)
            path = subj_dir / f"{mod}.nii.gz"
            save_nifti(vol, path)
            row[mod] = str(path)

        row.update(make_clinical(label))
        records.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {GREEN}[{i+1:2d}/{N_SUBJECTS}]{RESET}  subjects created")

    # Save CSV
    df = pd.DataFrame(records)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    print(f"\n  {GREEN}[Done]{RESET}  {N_SUBJECTS} subjects written to: {OUT_DIR}")
    print(f"  {GREEN}[Done]{RESET}  Metadata CSV written to:      {CSV_PATH}")
    print(f"\n  Class distribution:")
    for lbl, name in CLASSES.items():
        n = (df["label"] == lbl).sum()
        print(f"    {name:>6s} (label={lbl}): {n} subjects")
    print()


if __name__ == "__main__":
    np.random.seed(42)
    main()
