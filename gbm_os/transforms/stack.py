from __future__ import annotations

import numpy as np


def stack_modalities(
    volumes: dict[str, np.ndarray | None],
    modality_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-modality volumes into [C, H, W, D] and return a [C] presence mask.

    Missing modalities (None) are zero-filled. The presence mask is 1 for present
    channels and 0 for zero-filled ones.
    """
    arrays = []
    mask = []
    ref_shape: tuple[int, ...] | None = None

    for mod in modality_order:
        vol = volumes.get(mod)
        if vol is not None:
            if ref_shape is None:
                ref_shape = vol.shape
            arrays.append(vol.astype(np.float32))
            mask.append(1)
        else:
            arrays.append(None)
            mask.append(0)

    if ref_shape is None:
        raise ValueError("All modalities are missing — cannot determine volume shape")

    filled = [
        arr if arr is not None else np.zeros(ref_shape, dtype=np.float32)
        for arr in arrays
    ]
    return np.stack(filled, axis=0), np.array(mask, dtype=np.float32)
