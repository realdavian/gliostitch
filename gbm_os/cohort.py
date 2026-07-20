from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pandas as pd

from gbm_os.config import CohortConfig
from gbm_os.manifest import load_manifest
from gbm_os.provenance import SelectionTrace
from gbm_os.sample import SampleSpec
from gbm_os.selection import (
    SelectionCriteria,
    apply_criteria,
    apply_duplicate_policy,
)

logger = logging.getLogger(__name__)

_SCALAR_NA = (float("nan"),)


def _na(val: Any) -> Optional[Any]:
    """Convert pandas NA/NaN scalars to None."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _row_to_spec(row: pd.Series, data_roots: dict[str, Path]) -> SampleSpec:
    dataset = row["dataset"]
    root = data_roots.get(dataset)

    paths: dict[str, Optional[str]] = {}
    for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
        rel = _na(row.get(f"{mod}_path"))
        if rel is None or root is None:
            paths[mod] = None
        else:
            paths[mod] = str(root / rel)

    present: dict[str, bool] = {
        "t1": bool(row["has_t1"]),
        "t1ce": bool(row["has_t1ce"]),
        "t2": bool(row["has_t2"]),
        "flair": bool(row["has_flair"]),
        "seg": bool(row["has_seg"]),
    }

    os_class_raw = _na(row.get("os_class"))
    os_class = int(os_class_raw) if os_class_raw is not None else None

    return SampleSpec(
        paths=paths,
        present=present,
        dataset=dataset,
        patient_id=row["patient_id"],
        session_index=int(row["session_index"]),
        global_session_key=row["global_session_key"],
        is_baseline=bool(row["is_baseline"]),
        is_longitudinal=bool(row["is_longitudinal"]),
        is_structural_complete=bool(row["is_structural_complete"]),
        intensity_prenormalised=bool(row["intensity_prenormalised"]),
        seg_convention=_na(row.get("seg_convention")),
        age=_na(row.get("age")),
        os_days=_na(row.get("os_days")),
        os_event=_na(row.get("os_event")),
        os_class=os_class,
        who_grade=_na(row.get("who_grade")),
        eor=_na(row.get("eor")),
        idh_status=_na(row.get("idh_status")),
        mgmt_methylation=_na(row.get("mgmt_methylation")),
        duplicate_group_id=_na(row.get("duplicate_group_id")),
        dedup_confidence=str(row.get("dedup_confidence", "unique")),
        is_duplicate_flagged=bool(row.get("_dup_flagged", False)),
    )


class CohortView:
    """An ordered, immutable selection of SampleSpecs derived from the manifest."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_roots: dict[str, Path],
        config: CohortConfig,
        trace: Optional[SelectionTrace] = None,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._data_roots = data_roots
        self._config = config
        self._trace = trace if trace is not None else SelectionTrace(n_input=len(df))

    # ------------------------------------------------------------------ #
    # Provenance                                                           #
    # ------------------------------------------------------------------ #

    def provenance(self) -> SelectionTrace:
        """How this view was derived: each criterion, and what it removed."""
        return self._trace

    def exclusions(self) -> pd.DataFrame:
        """Every row excluded on the way to this view, tagged with the reason.

        Reasons partition the drops: each row is attributed to the first
        criterion that removed it, so len(exclusions) + len(view) == n_input.
        """
        return self._trace.exclusions()

    # ------------------------------------------------------------------ #
    # Core access                                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[SampleSpec]:
        for _, row in self._df.iterrows():
            yield _row_to_spec(row, self._data_roots)

    def to_records(self) -> list[SampleSpec]:
        return list(self)

    def to_frame(self) -> pd.DataFrame:
        """Return a pandas DataFrame view of the selected cohort (no paths resolved)."""
        return self._df.drop(columns=["_dup_flagged"], errors="ignore").copy()

    # ------------------------------------------------------------------ #
    # Chaining                                                             #
    # ------------------------------------------------------------------ #

    def select(
        self,
        datasets: Optional[list[str]] = None,
        partition: Optional[str] = None,
        baseline_only: bool = False,
        require_complete: bool = False,
        modalities: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        where: Optional[Callable] = None,
        resolve_duplicates: Optional[str] = None,
    ) -> "CohortView":
        """Further narrow this view. Returns a new CohortView."""
        criteria = SelectionCriteria(
            datasets=datasets,
            partition=partition,
            baseline_only=baseline_only,
            require_complete=require_complete,
            modalities=modalities,
            filters=filters or {},
            where=where,
            resolve_duplicates=resolve_duplicates,
        )
        trace = SelectionTrace()
        selected = apply_criteria(self._df, criteria, self._config, trace)
        policy = resolve_duplicates or self._config.resolve_duplicates
        selected = apply_duplicate_policy(selected, policy, self._config, trace)
        return CohortView(selected, self._data_roots, self._config, trace)

    # ------------------------------------------------------------------ #
    # Optional: CV splitting                                               #
    # ------------------------------------------------------------------ #

    def split(self, k: int = 5, seed: int = 42) -> "FoldCollection":
        from gbm_os.splits import stratified_group_kfold

        fold_assignments = stratified_group_kfold(self._df, k=k, seed=seed)
        return FoldCollection(self._df, fold_assignments, self._data_roots, self._config)

    # ------------------------------------------------------------------ #
    # Optional: loading backends                                           #
    # ------------------------------------------------------------------ #

    def to_monai(self, transforms=None, **kwargs):
        from gbm_os.backends.monai_backend import MonaiDataset

        return MonaiDataset(self, transforms=transforms, config=self._config, **kwargs)

    def to_torchio(self, **kwargs):
        from gbm_os.backends.torchio_backend import TorchioDataset

        return TorchioDataset(self, config=self._config, **kwargs)

    def to_torch(self, **kwargs):
        from gbm_os.backends.torch_backend import TorchDataset

        return TorchDataset(self, config=self._config, **kwargs)


class FoldCollection:
    """Holds fold assignments for a CohortView; exposes .fold(i, split) → CohortView."""

    def __init__(
        self,
        df: pd.DataFrame,
        fold_assignments: pd.Series,
        data_roots: dict[str, Path],
        config: CohortConfig,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._folds = fold_assignments.reset_index(drop=True)  # int series same index
        self._data_roots = data_roots
        self._config = config

    @property
    def k(self) -> int:
        return int(self._folds.max()) + 1

    def fold(self, i: int, split: str = "train") -> CohortView:
        if split == "val":
            mask = self._folds == i
        elif split == "train":
            mask = self._folds != i
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        sub = self._df[mask].reset_index(drop=True)
        return CohortView(sub, self._data_roots, self._config)

    def assignments(self) -> pd.Series:
        return self._folds.copy()


class Cohort:
    """Entry point: load manifest and provide selection."""

    def __init__(self, df: pd.DataFrame, data_roots: dict[str, Path], config: CohortConfig) -> None:
        self._df = df
        self._data_roots = data_roots
        self._config = config
        logger.info("Cohort ready: %d rows", len(df))

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        data_roots: dict[str, Any],
        config: Optional[CohortConfig] = None,
    ) -> "Cohort":
        if config is None:
            config = CohortConfig(data_roots={k: Path(v) for k, v in data_roots.items()})
        else:
            # data_roots passed directly take precedence for path resolution
            pass
        roots = {k: Path(v) for k, v in data_roots.items()}
        df = load_manifest(path, config)
        return cls(df, roots, config)

    def select(
        self,
        datasets: Optional[list[str]] = None,
        partition: Optional[str] = None,
        baseline_only: bool = False,
        require_complete: bool = False,
        modalities: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        where: Optional[Callable] = None,
        resolve_duplicates: Optional[str] = None,
    ) -> CohortView:
        """Select a cohort view. All criteria are optional and composable."""
        criteria = SelectionCriteria(
            datasets=datasets,
            partition=partition,
            baseline_only=baseline_only,
            require_complete=require_complete,
            modalities=modalities,
            filters=filters or {},
            where=where,
            resolve_duplicates=resolve_duplicates,
        )
        # Start from a clean base frame with flag column preset
        base = self._df.copy()
        base["_dup_flagged"] = False

        trace = SelectionTrace()
        selected = apply_criteria(base, criteria, self._config, trace)
        policy = resolve_duplicates or self._config.resolve_duplicates
        selected = apply_duplicate_policy(selected, policy, self._config, trace)
        return CohortView(selected, self._data_roots, self._config, trace)
