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


class SegRemapd:
    """MONAI dict-transform: remap seg labels to canonical BraTS convention.

    Apply AFTER LoadImaged in your Compose pipeline. Reads seg_convention
    from the data dict (set automatically by MonaiDataset) and remaps only
    RHUH data (label 3 → 4); all other conventions pass through unchanged.

    Example::

        transforms = Compose([
            LoadImaged(keys=["t1ce", "seg"]),
            EnsureChannelFirstd(keys=["t1ce", "seg"]),
            SegRemapd(seg_key="seg"),
        ])
    """

    def __init__(
        self,
        seg_key: str = "seg",
        convention_key: str = "seg_convention",
    ) -> None:
        self.seg_key = seg_key
        self.convention_key = convention_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        seg = d.get(self.seg_key)
        convention = d.get(self.convention_key)
        if seg is None or convention is None:
            return d

        import torch

        remapped = remap_seg(np.asarray(seg, dtype=np.int32), str(convention))
        t = torch.from_numpy(remapped)

        try:
            from monai.data import MetaTensor
            if isinstance(seg, MetaTensor):
                d[self.seg_key] = MetaTensor(t, affine=seg.affine, meta=seg.meta)
                return d
        except ImportError:
            pass

        d[self.seg_key] = t
        return d
