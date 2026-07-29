"""P1 — manifest loading + runtime derivations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gbm_os.config import CohortConfig
from gbm_os.manifest import load_manifest

MANIFEST = Path(__file__).parents[2] / "output" / "master_manifest.csv"


@pytest.fixture(scope="module")
def df():
    if not MANIFEST.exists():
        pytest.skip(
            f"{MANIFEST} not found — run `gliostitch build` to generate it. "
            "Tests that exercise adapter and pipeline logic run without it."
        )
    config = CohortConfig(data_roots={"brats2020": Path("/tmp")})
    return load_manifest(MANIFEST, config)


class TestManifestLoad:
    def test_row_count(self, df):
        assert len(df) == 2109

    def test_datasets(self, df):
        assert set(df["dataset"].unique()) == {
            "brats2020", "lumiere", "rhuh_gbm", "ucsf_pdgm", "upenn_gbm"
        }

    def test_derived_columns_present(self, df):
        for col in ("os_class", "is_baseline", "is_longitudinal",
                    "is_structural_complete", "has_os", "intensity_prenormalised"):
            assert col in df.columns, f"missing derived column: {col}"

    def test_derived_not_stored_originally(self, df):
        raw = pd.read_csv(MANIFEST)
        for col in ("os_class", "is_baseline", "is_longitudinal",
                    "is_structural_complete", "has_os"):
            assert col not in raw.columns, f"column should not be in raw CSV: {col}"


class TestOsClassBoundaries:
    """Unit-tests os_class derivation at the documented threshold boundaries."""

    @pytest.mark.parametrize("os_days, expected_class", [
        (299.0, 0),   # short: < 300
        (300.0, 1),   # mid:  300 <= x < 450
        (449.0, 1),   # mid:  300 <= x < 450
        (450.0, 2),   # long: >= 450
        (0.0,   0),
        (600.0, 2),
    ])
    def test_boundary(self, df, os_days, expected_class):
        from gbm_os.manifest import derive_os_class
        import pandas as pd
        result = derive_os_class(pd.Series([os_days]), short_max=300, mid_max=450)
        assert int(result.iloc[0]) == expected_class

    @pytest.mark.parametrize("missing", [float("nan"), None])
    def test_missing_survival_is_never_a_band(self, missing):
        """A null survival time must yield a null class, not 'long'.

        The removed scalar implementation returned 2 for NaN, so a session with
        no outcome was indistinguishable from a long survivor.
        """
        import numpy as np
        import pandas as pd
        from gbm_os.manifest import derive_os_class

        result = derive_os_class(pd.Series([missing], dtype="float64"),
                                 short_max=300, mid_max=450)
        assert pd.isna(result.iloc[0])

    def test_single_definition_of_os_class(self):
        """The pipeline must not carry a second implementation."""
        import gbm_manifest.core.schema as schema

        assert not hasattr(schema, "derive_os_class")
        assert not hasattr(schema, "is_baseline")
        assert not hasattr(schema, "is_structural_complete")

    def test_null_os_days_gives_null_class(self, df):
        null_rows = df[df["os_days"].isna()]
        assert null_rows["os_class"].isna().all()

    def test_nonnull_os_days_gives_valid_class(self, df):
        nonnull = df[df["os_days"].notna()]
        assert nonnull["os_class"].isin([0, 1, 2]).all()


class TestDerivedFlags:
    def test_is_baseline(self, df):
        assert (df["is_baseline"] == (df["session_index"] == 0)).all()

    def test_is_longitudinal_rhuh(self, df):
        rhuh = df[df["dataset"] == "rhuh_gbm"]
        # RHUH has 3 sessions per patient — all longitudinal
        assert rhuh["is_longitudinal"].all()

    def test_is_longitudinal_brats(self, df):
        brats = df[df["dataset"] == "brats2020"]
        # BraTS has only session 0 per patient — none longitudinal
        assert not brats["is_longitudinal"].any()

    def test_is_structural_complete(self, df):
        expected = df["has_t1"] & df["has_t1ce"] & df["has_t2"] & df["has_flair"]
        assert (df["is_structural_complete"] == expected).all()

    def test_intensity_prenormalised_rhuh_only(self, df):
        assert df[df["dataset"] == "rhuh_gbm"]["intensity_prenormalised"].all()
        assert not df[df["dataset"] != "rhuh_gbm"]["intensity_prenormalised"].any()
