"""
MSNet — Utility: Seed Everything for Reproducibility
=====================================================
Sets random seeds for Python, NumPy, PyTorch (CPU + CUDA),
and configures cuDNN for deterministic operation.

Critical for:
  - Reproducible training runs
  - Exact result reporting in publications
  - Fair ablation study comparisons
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # cuDNN determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ fill operations
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception:
            pass  # Some ops don't have deterministic implementations

    logger.info(f"[Seed] All random seeds set to {seed}")
