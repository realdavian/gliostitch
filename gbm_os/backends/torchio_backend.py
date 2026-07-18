from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gbm_os.config import CohortConfig
from gbm_os.transforms.seg import remap_seg

logger = logging.getLogger(__name__)


def _load_seg_remapped(path: str, seg_convention: str | None):
    """Load a seg NIfTI, remap labels to BraTS convention, return a tio.LabelMap."""
    import nibabel as nib
    import torch
    import torchio as tio

    img = nib.load(path)
    arr = np.asarray(img.dataobj, dtype=np.int32)
    if seg_convention:
        arr = remap_seg(arr, seg_convention)
    return tio.LabelMap(tensor=torch.from_numpy(arr[np.newaxis]), affine=np.array(img.affine))


class TorchioDataset:
    """Wraps a CohortView as a torchio SubjectsDataset."""

    def __init__(self, view, config: CohortConfig, include_seg: bool | None = None) -> None:
        try:
            import torchio as tio
        except ImportError as e:
            raise ImportError(
                "torchio is required for the torchio backend. "
                "Install with: pip install gbm-os[torchio]"
            ) from e

        self._include_seg = include_seg if include_seg is not None else config.include_seg
        self._config = config
        subjects = []
        for spec in view:
            data: dict[str, Any] = {}
            for mod in config.modalities:
                p = spec.paths.get(mod)
                if p is not None and spec.present.get(mod, False):
                    data[mod] = tio.ScalarImage(p)
            if self._include_seg and spec.paths.get("seg") and spec.present.get("seg", False):
                data["seg"] = _load_seg_remapped(spec.paths["seg"], spec.seg_convention)
            data["dataset"] = spec.dataset
            data["patient_id"] = spec.patient_id
            data["age"] = spec.age
            data["os_days"] = spec.os_days
            data["os_class"] = spec.os_class
            subjects.append(tio.Subject(**data))

        self._ds = tio.SubjectsDataset(subjects)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]
