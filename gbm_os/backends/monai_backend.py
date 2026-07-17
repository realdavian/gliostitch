from __future__ import annotations

import logging
from typing import Any, Optional

from gbm_os.config import CohortConfig

logger = logging.getLogger(__name__)


class MonaiDataset:
    """Wraps a CohortView as a MONAI-compatible dataset.

    Emits dicts with:
      image   [C, H, W, D]  float32  (modalities stacked in config.modalities order)
      modality_mask  [C]  float32  (1 = present, 0 = zero-filled)
      seg    [1, H, W, D]  (if include_seg=True)
      + all SampleSpec scalars as flat keys
    """

    def __init__(
        self,
        view,  # CohortView
        config: CohortConfig,
        transforms=None,
        include_seg: Optional[bool] = None,
        remap_seg: bool = False,
    ) -> None:
        try:
            from monai.data import Dataset
            from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
        except ImportError as e:
            raise ImportError(
                "monai is required for the MONAI backend. "
                "Install with: pip install gbm-os[monai]"
            ) from e

        self._view = view
        self._config = config
        self._transforms = transforms
        self._include_seg = include_seg if include_seg is not None else config.include_seg
        self._remap_seg = remap_seg
        self._dataset = self._build(view)

    def _build(self, view):
        from monai.data import Dataset as MonaiDS

        data_list = []
        for spec in view:
            item: dict[str, Any] = {}
            # image modalities
            for mod in self._config.modalities:
                p = spec.paths.get(mod)
                if p is not None and spec.present.get(mod, False):
                    item[mod] = p
            if self._include_seg and spec.paths.get("seg"):
                item["seg"] = spec.paths["seg"]

            # clinical scalars
            item.update({
                "dataset": spec.dataset,
                "patient_id": spec.patient_id,
                "session_index": spec.session_index,
                "age": spec.age,
                "os_days": spec.os_days,
                "os_event": spec.os_event,
                "os_class": spec.os_class,
                "who_grade": spec.who_grade,
                "eor": spec.eor,
                "intensity_prenormalised": spec.intensity_prenormalised,
                "seg_convention": spec.seg_convention,
                "_modalities": self._config.modalities,
                "_present": spec.present,
            })
            data_list.append(item)

        return MonaiDS(data=data_list, transform=self._transforms)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]
