from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gbm_os.config import CohortConfig

logger = logging.getLogger(__name__)

# Dataset-level fact: RHUH volumes are pre-z-scored; others are not.
_INTENSITY_PRENORM_DATASETS: frozenset[str] = frozenset({"rhuh_gbm"})


def _derive_os_class(os_days: pd.Series, short_max: int, mid_max: int) -> pd.Series:
    """Classify os_days into 0=short, 1=mid, 2=long; NaN stays NaN."""
    result = pd.cut(
        os_days,
        bins=[-np.inf, short_max, mid_max, np.inf],
        labels=[0, 1, 2],
        right=False,
    )
    return result.astype("Int64")


def load_manifest(path: str | Path, config: CohortConfig) -> pd.DataFrame:
    """Load master_manifest.csv and append runtime-derived columns."""
    df = pd.read_csv(path)
    logger.info(
        "Loaded manifest: %d rows from %d datasets",
        len(df),
        df["dataset"].nunique(),
    )

    short_max, mid_max = config.os_thresholds

    # --- derived columns (DERIVED_NOT_STORED per Phase 1 contract) ---
    df["os_class"] = _derive_os_class(df["os_days"], short_max, mid_max)
    df["is_baseline"] = df["session_index"] == 0

    counts = df.groupby(["dataset", "patient_id"])["session_index"].transform("count")
    df["is_longitudinal"] = counts > 1

    df["is_structural_complete"] = (
        df["has_t1"] & df["has_t1ce"] & df["has_t2"] & df["has_flair"]
    )

    # Convenience boolean for clinical filters
    df["has_os"] = df["os_days"].notna()

    # Dataset-level imaging fact
    df["intensity_prenormalised"] = df["dataset"].isin(_INTENSITY_PRENORM_DATASETS)

    return df
