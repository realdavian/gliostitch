"""`where=` predicates receive Python nulls, not NaN.

pandas spells a missing value as NaN, so `r["os_days"] is not None` against a
raw row is always true — the documented idiom looks like a filter and silently
does nothing. Rows are normalised before reaching the predicate.
"""
from __future__ import annotations

import pandas as pd
import pytest


class TestNullNormalisation:
    def test_is_not_none_actually_filters(self, cohort):
        """The idiom from spec 02 §5 must behave as written."""
        view = cohort.select(where=lambda r: r["os_days"] is not None)
        df = view.to_frame()
        assert df["os_days"].notna().all()
        assert 0 < len(df) < 1661

    def test_is_none_selects_the_complement(self, cohort):
        with_os = len(cohort.select(where=lambda r: r["os_days"] is not None))
        without = len(cohort.select(where=lambda r: r["os_days"] is None))
        assert with_os + without == 1661

    def test_matches_pandas_notna(self, cohort):
        via_python = len(cohort.select(where=lambda r: r["os_days"] is not None))
        via_pandas = len(cohort.select(where=lambda r: pd.notna(r["os_days"])))
        assert via_python == via_pandas

    def test_spec_example_composes(self, cohort):
        """The full documented example: null check AND a numeric comparison."""
        view = cohort.select(
            where=lambda r: r["os_days"] is not None and r["age"] is not None
            and r["age"] >= 18
        )
        df = view.to_frame()
        assert df["os_days"].notna().all()
        assert (df["age"] >= 18).all()

    def test_present_values_are_untouched(self, cohort):
        """Normalisation must not disturb non-null values."""
        view = cohort.select(where=lambda r: r["dataset"] == "rhuh_gbm")
        assert len(view) == 120


class TestSummaryReporting:
    def test_report_returns_text(self, cohort):
        from gbm_os import cohort_summary

        summary = cohort_summary(cohort.select(datasets=["rhuh_gbm"]))
        assert isinstance(summary.report(), str)
        assert summary.report() == str(summary)

    def test_library_does_not_print(self):
        from pathlib import Path

        repo = Path(__file__).parents[2]
        source = (repo / "gbm_os" / "summary.py").read_text()
        assert "\n        print(" not in source
