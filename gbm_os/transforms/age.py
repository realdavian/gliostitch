from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AgeNormalizer:
    """Leakage-safe fold-aware age normalizer.

    Usage:
        norm = AgeNormalizer()
        norm.fit(train_view)          # computes mean/std from train ages
        train_ages = norm.transform(train_view)
        val_ages   = norm.transform(val_view)   # uses train stats
    """

    def __init__(self) -> None:
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, view) -> "AgeNormalizer":
        """Fit on a CohortView; stats are computed over non-null ages only."""
        from gbm_os.cohort import CohortView

        df = view.to_frame()
        ages = df["age"].dropna()
        if len(ages) == 0:
            logger.warning("AgeNormalizer.fit: no non-null ages in training view")
            self._mean = 0.0
            self._std = 1.0
        else:
            self._mean = float(ages.mean())
            self._std = float(ages.std())
            if self._std < 1e-8:
                self._std = 1.0
        logger.debug("AgeNormalizer fitted: mean=%.2f std=%.2f", self._mean, self._std)
        return self

    def transform(self, view) -> pd.Series:
        """Return normalized ages (NaN preserved) using train-fit statistics."""
        if self._mean is None:
            raise RuntimeError("Call fit() before transform()")
        df = view.to_frame()
        return (df["age"] - self._mean) / self._std

    def fit_transform(self, view) -> pd.Series:
        return self.fit(view).transform(view)
