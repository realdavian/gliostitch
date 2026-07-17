from __future__ import annotations

import numpy as np


def foreground_zscore(
    volume: np.ndarray,
    intensity_prenormalised: bool = False,
    brain_threshold: float = 0.0,
) -> np.ndarray:
    """Z-score a volume over foreground voxels only.

    Passthrough (returns unchanged array) when intensity_prenormalised is True.
    Foreground is defined as voxels > brain_threshold (configurable to avoid
    background fill differences across datasets).
    """
    if intensity_prenormalised:
        return volume
    fg = volume[volume > brain_threshold]
    if fg.size == 0:
        return volume
    mu, sigma = fg.mean(), fg.std()
    if sigma < 1e-8:
        return volume
    return (volume - mu) / sigma
