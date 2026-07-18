"""Tests for manifest schema validation in gbm_os.manifest._validate_manifest."""
from __future__ import annotations

import pandas as pd
import pytest

from gbm_manifest.core.schema import MANIFEST_COLUMNS
from gbm_os.manifest import _validate_manifest


class TestValidateManifest:
    def _make_df(self, drop: list[str] | None = None) -> pd.DataFrame:
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
        if drop:
            df = df.drop(columns=drop)
        return df

    def test_valid_manifest_passes(self):
        _validate_manifest(self._make_df())

    def test_missing_one_column_raises(self):
        df = self._make_df(drop=["os_days"])
        with pytest.raises(ValueError, match="os_days"):
            _validate_manifest(df)

    def test_missing_multiple_columns_raises(self):
        df = self._make_df(drop=["os_days", "patient_id", "dataset"])
        with pytest.raises(ValueError) as exc:
            _validate_manifest(df)
        msg = str(exc.value)
        assert "os_days" in msg
        assert "patient_id" in msg
        assert "dataset" in msg

    def test_error_mentions_count(self):
        df = self._make_df(drop=["os_days", "age"])
        with pytest.raises(ValueError, match="2 required"):
            _validate_manifest(df)

    def test_extra_columns_are_allowed(self):
        df = self._make_df()
        df["extra_derived_column"] = 0
        _validate_manifest(df)  # should not raise

    def test_real_manifest_passes(self, cohort):
        """The actual master_manifest.csv on disk satisfies the contract."""
        from pathlib import Path
        from tests.conftest import MANIFEST_PATH
        df = pd.read_csv(MANIFEST_PATH)
        _validate_manifest(df)
