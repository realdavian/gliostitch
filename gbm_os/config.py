from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator


class CohortConfig(BaseModel):
    data_roots: dict[str, Path]
    os_thresholds: tuple[int, int] = (300, 450)
    # partition_map: e.g. {"external": {"upenn_gbm"}}
    partition_map: dict[str, set[str]] = {}
    # priority: ordered list of datasets; determines dedup winner within same partition
    priority: list[str] = []
    modalities: list[str] = ["t1", "t1ce", "t2", "flair"]
    include_seg: bool = False
    resolve_duplicates: str = "flag"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("resolve_duplicates")
    @classmethod
    def _valid_dedup(cls, v: str) -> str:
        if v not in ("drop", "flag", "keep"):
            raise ValueError(f"resolve_duplicates must be drop/flag/keep, got {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _coerce_external_shorthand(cls, data: Any) -> Any:
        if isinstance(data, dict) and "external" in data and "partition_map" not in data:
            data = dict(data)
            data["partition_map"] = {"external": data.pop("external")}
        return data

    def dataset_partition(self, dataset: str) -> str:
        """Return partition name for a dataset, or 'train' if not in any partition."""
        for part, datasets in self.partition_map.items():
            if dataset in datasets:
                return part
        return "train"

    def dataset_rank(self, dataset: str) -> int:
        """Lower = higher priority (wins dedup). external loses; priority list orders rest."""
        if self.dataset_partition(dataset) == "external":
            return 10_000
        try:
            return self.priority.index(dataset)
        except ValueError:
            return 5_000
