from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gbm_os.config import CohortConfig
from gbm_os.io.volume import load_volume
from gbm_os.transforms.stack import stack_modalities

logger = logging.getLogger(__name__)


class TorchDataset:
    """Nibabel + numpy backend; no framework dependency beyond numpy."""

    def __init__(self, view, config: CohortConfig, include_seg: bool | None = None) -> None:
        self._specs = list(view)
        self._config = config
        self._include_seg = include_seg if include_seg is not None else config.include_seg

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec = self._specs[idx]

        volumes: dict[str, np.ndarray | None] = {}
        for mod in self._config.modalities:
            p = spec.paths.get(mod)
            volumes[mod] = load_volume(p) if p and spec.present.get(mod) else None

        image, modality_mask = stack_modalities(volumes, self._config.modalities)

        item: dict[str, Any] = {
            "image": image,               # [C, H, W, D]
            "modality_mask": modality_mask,
            "dataset": spec.dataset,
            "patient_id": spec.patient_id,
            "session_index": spec.session_index,
            "age": spec.age,
            "os_days": spec.os_days,
            "os_event": spec.os_event,
            "os_class": spec.os_class,
        }

        if self._include_seg and spec.paths.get("seg"):
            from gbm_os.transforms.seg import remap_seg
            seg_vol = load_volume(spec.paths["seg"]).astype(np.int32)
            if spec.seg_convention:
                seg_vol = remap_seg(seg_vol, spec.seg_convention)
            item["seg"] = seg_vol[np.newaxis]  # [1, H, W, D]

        return item
