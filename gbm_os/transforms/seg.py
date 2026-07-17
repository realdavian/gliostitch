from __future__ import annotations

import numpy as np


# RHUH uses {0,1,2,3} with ET=3; canonical BraTS convention uses {0,1,2,4} with ET=4.
RHUH_REMAP: dict[int, int] = {3: 4}


def remap_seg(seg: np.ndarray, seg_convention: str) -> np.ndarray:
    """Return seg array with labels remapped to canonical BraTS convention.

    Only mutates RHUH labels; all others are passed through unchanged.
    """
    if seg_convention != "rhuh":
        return seg
    out = seg.copy()
    for src, dst in RHUH_REMAP.items():
        out[seg == src] = dst
    return out
