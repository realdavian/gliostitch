from __future__ import annotations

import numpy as np


# Canonical BraTS convention is {0,1,2,4}: 1 = necrosis/non-enhancing,
# 2 = edema, 4 = enhancing tumour.
#
# RHUH uses {0,1,2,3} with only the enhancing label moved, so 3 -> 4 suffices.
#
# LUMIERE ships DeepBraTumIA output, also {0,1,2,3}, but under a different
# ordering: 1 = enhancing, 2 = necrosis, 3 = edema. That is a permutation, not a
# substitution, and it is why the two cannot share a table — applying the RHUH
# map to LUMIERE would leave enhancing tumour labelled 1 (necrosis) and edema
# labelled 4 (enhancing).
RHUH_REMAP: dict[int, int] = {3: 4}
LUMIERE_REMAP: dict[int, int] = {1: 4, 2: 1, 3: 2}

_REMAPS: dict[str, dict[int, int]] = {
    "rhuh": RHUH_REMAP,
    "lumiere": LUMIERE_REMAP,
}


def remap_seg(seg: np.ndarray, seg_convention: str) -> np.ndarray:
    """Return seg array with labels remapped to canonical BraTS convention.

    Conventions without an entry — brats_legacy, and anything unrecognised —
    pass through unchanged.
    """
    table = _REMAPS.get(seg_convention)
    if table is None:
        return seg
    out = seg.copy()
    # Masks are taken against the ORIGINAL array, so a permutation such as
    # LUMIERE's cannot overwrite a label it has yet to read.
    for src, dst in table.items():
        out[seg == src] = dst
    return out


class SegRemapd:
    """MONAI dict-transform: remap seg labels to canonical BraTS convention.

    Apply AFTER LoadImaged in your Compose pipeline. Reads seg_convention
    from the data dict (set automatically by MonaiDataset) and remaps RHUH
    (3 → 4) and LUMIERE (1 → 4, 2 → 1, 3 → 2); all other conventions pass
    through unchanged.

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
