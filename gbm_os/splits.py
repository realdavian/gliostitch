from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stratified_group_kfold(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    strat_cols: tuple[str, ...] = ("os_class", "dataset"),
    group_col: tuple[str, ...] = ("dataset", "patient_id"),
) -> pd.Series:
    """Return a Series of fold indices (0..k-1), same index as df.

    - Patient-grouped: all rows for a patient land in the same fold.
    - Stratified: patients distributed across folds to balance strat_cols strata.
    - Deterministic: same seed → identical assignments.
    """
    rng = np.random.default_rng(seed)

    # Build a patient-level table: one row per unique patient group.
    group_key = list(group_col)
    strat_key = list(strat_cols)
    # Only fetch strat cols that aren't already part of the group index
    extra_strat = [c for c in strat_key if c not in group_key]

    if extra_strat:
        patient_df = (
            df.groupby(group_key, sort=False)[extra_strat]
            .first()
            .reset_index()
        )
    else:
        patient_df = df[group_key].drop_duplicates().reset_index(drop=True)

    # Encode strata as a single string label (using all strat_key cols available)
    avail_strat = [c for c in strat_key if c in patient_df.columns]
    patient_df["_stratum"] = patient_df[avail_strat].astype(str).agg("|".join, axis=1)

    fold_col = np.full(len(patient_df), -1, dtype=int)

    for stratum, idx in patient_df.groupby("_stratum").groups.items():
        idx_arr = np.array(idx)
        rng_stratum = np.random.default_rng(
            seed ^ hash(stratum) & 0xFFFF_FFFF
        )
        rng_stratum.shuffle(idx_arr)
        for offset, patient_idx in enumerate(idx_arr):
            fold_col[patient_idx] = offset % k

    patient_df["_fold"] = fold_col

    # Warn if any patient remains unassigned (shouldn't happen)
    unassigned = (fold_col == -1).sum()
    if unassigned:
        logger.warning("%d patients have no fold assignment", unassigned)

    # Map fold back to original df rows
    merge_key = group_key
    fold_map = patient_df.set_index(merge_key)["_fold"]
    row_folds = (
        df[merge_key]
        .apply(lambda r: fold_map.get(tuple(r), -1), axis=1)
    )
    row_folds.index = df.index

    logger.info(
        "stratified_group_kfold: k=%d, seed=%d, %d patients → fold distribution: %s",
        k,
        seed,
        len(patient_df),
        pd.Series(row_folds).value_counts().sort_index().to_dict(),
    )
    return row_folds
