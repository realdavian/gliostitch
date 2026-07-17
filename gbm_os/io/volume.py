from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_volume(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume via nibabel and return a float32 numpy array [H, W, D]."""
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required for volume loading. "
            "Install it with: pip install gbm-os[load]"
        ) from e

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)
