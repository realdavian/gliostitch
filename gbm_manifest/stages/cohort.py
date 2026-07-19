"""CohortSelector: derives the OS modelling cohort from the manifest.

This is a DERIVED artifact (selected.csv / exclusions.csv), NOT the manifest itself.
All filtering logic lives here, not in the manifest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import CohortConfig
from ..core.schema import EORCategory, derive_os_class
from ..infra.io import write_csv

log = logging.getLogger(__name__)

# Priority for duplicate resolution (higher index = keep preference)
_PRIORITY = {
    "brats2020": 3,
    "ucsf_pdgm": 2,
    "rhuh_gbm": 1,
    "upenn_gbm": 0,   # external test set candidate — lowest priority
}


class CohortSelector:
    def __init__(self, output_dir: Path, cfg: CohortConfig) -> None:
        self.output_dir = output_dir / "cohort"
        self.cfg = cfg

    def run(self, manifest: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        sel_path = self.output_dir / "selected.csv"
        exc_path = self.output_dir / "exclusions.csv"

        if not force and sel_path.exists():
            log.info("cohort: loading cached %s", sel_path)
            return pd.read_csv(sel_path)

        df = manifest.copy()
        exclusions = []

        def _exclude(mask: pd.Series, reason: str) -> None:
            rows = df[mask].copy()
            rows["exclusion_reason"] = reason
            exclusions.append(rows)

        # Step 1: baseline only
        not_baseline = df["session_index"] != 0
        _exclude(not_baseline, "not_baseline")
        df = df[~not_baseline].copy()

        # Step 2: structural complete
        not_complete = ~(df["has_t1"] & df["has_t1ce"] & df["has_t2"] & df["has_flair"])
        _exclude(not_complete, "incomplete_modalities")
        df = df[~not_complete].copy()

        # Step 3: grade IV only
        not_grade4 = df["who_grade"].isna() | (df["who_grade"] != 4)
        # UPENN is GBM-only (implicit grade IV), so don't exclude on grade for UPENN
        not_grade4_strict = not_grade4 & (df["dataset"] != "upenn_gbm")
        _exclude(not_grade4_strict, "not_grade_iv")
        df = df[~not_grade4_strict].copy()

        # Step 4: GTR only
        not_gtr = df["eor"] != EORCategory.GTR.value
        _exclude(not_gtr, "not_gtr")
        df = df[~not_gtr].copy()

        # Step 5: has OS data
        no_os = df["os_days"].isna() | df["os_event"].isna()
        _exclude(no_os, "no_os_data")
        df = df[~no_os].copy()

        # Step 6: derive os_class
        df["os_class"] = df["os_days"].apply(
            lambda d: derive_os_class(d, self.cfg.os_short_max_days,
                                      self.cfg.os_mid_max_days)
        )

        # Step 7: resolve duplicates (keep highest-priority dataset)
        df = self._resolve_duplicates(df, exclusions)

        log.info("cohort: selected %d sessions", len(df))

        write_csv(df, sel_path)
        if exclusions:
            exc_df = pd.concat(exclusions, ignore_index=True)
            write_csv(exc_df, exc_path)
            log.info("cohort: excluded %d sessions", len(exc_df))

        return df

    def _resolve_duplicates(self, df: pd.DataFrame,
                            exclusions: list[pd.DataFrame]) -> pd.DataFrame:
        if "duplicate_group_id" not in df.columns:
            return df
        duped = df[df["duplicate_group_id"].notna()].copy()
        if duped.empty:
            return df

        keep_keys: set[str] = set()
        for group_id, group in duped.groupby("duplicate_group_id"):
            # Keep the row from the highest-priority dataset
            group_sorted = group.copy()
            group_sorted["_pri"] = group_sorted["dataset"].map(
                lambda d: _PRIORITY.get(d, -1)
            )
            group_sorted = group_sorted.sort_values("_pri", ascending=False)
            keep_keys.add(group_sorted.iloc[0]["global_session_key"])

        # Exclude duplicate losers
        dup_losers = duped[~duped["global_session_key"].isin(keep_keys)]
        if not dup_losers.empty:
            dup_losers = dup_losers.copy()
            dup_losers["exclusion_reason"] = "duplicate_lower_priority"
            exclusions.append(dup_losers)

        df = df[
            df["duplicate_group_id"].isna() |
            df["global_session_key"].isin(keep_keys)
        ].copy()
        return df
