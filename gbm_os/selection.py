from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from gbm_os.config import CohortConfig

logger = logging.getLogger(__name__)


@dataclass
class SelectionCriteria:
    """Composable, all-optional selection knobs passed to Cohort.select()."""

    datasets: Optional[list[str]] = None
    partition: Optional[str] = None
    baseline_only: bool = False
    require_complete: bool = False
    # require these specific modalities to be present (independent of require_complete)
    modalities: Optional[list[str]] = None
    # equality / set / bool / None filters on any manifest column
    filters: dict[str, Any] = field(default_factory=dict)
    # arbitrary row predicate (row is a pandas Series)
    where: Optional[Callable[[pd.Series], bool]] = None
    # overrides config default when not None
    resolve_duplicates: Optional[str] = None


def apply_criteria(
    df: pd.DataFrame,
    criteria: SelectionCriteria,
    config: CohortConfig,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if criteria.datasets is not None:
        mask &= df["dataset"].isin(criteria.datasets)

    if criteria.partition is not None:
        # Use dataset_partition() so the implicit "train" partition (any dataset
        # not listed in partition_map) is handled correctly.
        unique_datasets = df["dataset"].unique()
        partition_datasets = {
            d for d in unique_datasets
            if config.dataset_partition(d) == criteria.partition
        }
        mask &= df["dataset"].isin(partition_datasets)

    if criteria.baseline_only:
        mask &= df["is_baseline"]

    if criteria.require_complete:
        mask &= df["is_structural_complete"]

    if criteria.modalities is not None:
        for mod in criteria.modalities:
            col = f"has_{mod}"
            if col in df.columns:
                mask &= df[col].astype(bool)
            else:
                logger.warning("Modality column %r not found in manifest", col)

    for key, val in criteria.filters.items():
        if key not in df.columns:
            logger.warning("Filter key %r not in manifest columns — skipped", key)
            continue
        col = df[key]
        if isinstance(val, (list, set, tuple)):
            mask &= col.isin(list(val))
        elif val is None:
            mask &= col.isna()
        elif isinstance(val, bool):
            mask &= col == val
        else:
            mask &= col == val

    if criteria.where is not None:
        mask &= df.apply(criteria.where, axis=1)

    selected = df[mask].reset_index(drop=True)
    logger.debug("apply_criteria: %d / %d rows selected", len(selected), len(df))
    return selected


def apply_duplicate_policy(
    df: pd.DataFrame,
    policy: str,
    config: CohortConfig,
) -> pd.DataFrame:
    """Apply the duplicate resolution policy to the already-selected frame."""
    if policy == "keep":
        out = df.copy()
        out["_dup_flagged"] = False
        return out

    if policy == "flag":
        out = df.copy()
        out["_dup_flagged"] = out["duplicate_group_id"].notna()
        return out

    if policy == "drop":
        groups = df.loc[df["duplicate_group_id"].notna(), "duplicate_group_id"].unique()
        drop_idx: list[int] = []
        for gid in groups:
            members = df[df["duplicate_group_id"] == gid]
            if len(members) <= 1:
                continue
            ranked = members.assign(
                _rank=members["dataset"].map(config.dataset_rank)
            ).sort_values("_rank")
            drop_idx.extend(ranked.index[1:].tolist())

        out = df.drop(index=drop_idx).reset_index(drop=True)
        out["_dup_flagged"] = False
        logger.debug(
            "resolve_duplicates=drop: removed %d rows from %d groups",
            len(drop_idx),
            len(groups),
        )
        return out

    raise ValueError(f"resolve_duplicates must be drop/flag/keep, got {policy!r}")
