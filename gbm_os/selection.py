from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from gbm_os.config import CohortConfig
from gbm_os.provenance import SelectionTrace

logger = logging.getLogger(__name__)


def _nullsafe_row(row: pd.Series) -> pd.Series:
    """A copy of `row` with every missing value replaced by None.

    Lets a `where=` predicate use plain Python null checks (`is None`,
    `or`, `not`) instead of having to know that pandas spells missing as NaN.
    """
    return row.where(pd.notna(row), None)


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
    trace: Optional[SelectionTrace] = None,
) -> pd.DataFrame:
    """Apply each criterion in turn, optionally recording what each one removed.

    Criteria are applied sequentially rather than as one combined mask so that
    every dropped row can be attributed to the first criterion that excluded it.
    The selected set is identical either way; only the accounting differs.
    """
    if trace is not None:
        trace.n_input = len(df)

    current = df

    def step(criterion: str, detail: str, mask: pd.Series) -> None:
        nonlocal current
        after = current[mask.reindex(current.index, fill_value=False)]
        if trace is not None:
            trace.record(criterion, detail, current, after)
        current = after

    if criteria.datasets is not None:
        step("datasets", f"dataset in {sorted(criteria.datasets)}",
             current["dataset"].isin(criteria.datasets))

    if criteria.partition is not None:
        # Use dataset_partition() so the implicit "train" partition (any dataset
        # not listed in partition_map) is handled correctly.
        partition_datasets = {
            d for d in current["dataset"].unique()
            if config.dataset_partition(d) == criteria.partition
        }
        step("partition", f"partition == {criteria.partition!r}",
             current["dataset"].isin(partition_datasets))

    if criteria.baseline_only:
        step("baseline_only", "session_index == 0", current["is_baseline"])

    if criteria.require_complete:
        step("require_complete", "t1 & t1ce & t2 & flair present",
             current["is_structural_complete"])

    if criteria.modalities is not None:
        for mod in criteria.modalities:
            col = f"has_{mod}"
            if col not in current.columns:
                logger.warning("Modality column %r not found in manifest", col)
                continue
            step(f"modality:{mod}", f"{col} is True", current[col].astype(bool))

    for key, val in criteria.filters.items():
        if key not in current.columns:
            logger.warning("Filter key %r not in manifest columns — skipped", key)
            continue
        col = current[key]
        if isinstance(val, (list, set, tuple)):
            # A None inside the collection admits nulls alongside the concrete
            # values — e.g. who_grade in [4, None] keeps grade-IV cases and
            # cohorts that record no grade at all (UPENN is GBM by construction).
            values = list(val)
            concrete = [v for v in values if v is not None]
            mask = col.isin(concrete)
            if any(v is None for v in values):
                mask |= col.isna()
            detail = f"{key} in {values}"
        elif val is None:
            mask, detail = col.isna(), f"{key} is null"
        else:
            mask, detail = col == val, f"{key} == {val!r}"
        step(f"filter:{key}", detail, mask)

    if criteria.where is not None:
        # Nulls are normalised to None before the row reaches the predicate.
        # pandas represents a missing value as NaN, so the natural Python idiom
        # `r["os_days"] is not None` is silently always true against a raw row —
        # the filter looks applied and does nothing.
        step("where", "custom predicate",
             current.apply(lambda r: bool(criteria.where(_nullsafe_row(r))),
                           axis=1))

    selected = current.reset_index(drop=True)
    logger.debug("apply_criteria: %d / %d rows selected", len(selected), len(df))
    return selected


def apply_duplicate_policy(
    df: pd.DataFrame,
    policy: str,
    config: CohortConfig,
    trace: Optional[SelectionTrace] = None,
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

        kept = df.drop(index=drop_idx)
        if trace is not None:
            trace.record("resolve_duplicates",
                         "keep highest-priority member per duplicate_group_id",
                         df, kept)
        out = kept.reset_index(drop=True)
        out["_dup_flagged"] = False
        logger.debug(
            "resolve_duplicates=drop: removed %d rows from %d groups",
            len(drop_idx),
            len(groups),
        )
        return out

    raise ValueError(f"resolve_duplicates must be drop/flag/keep, got {policy!r}")
