"""Adapter join contracts, exercised against the synthetic tree.

These encode spec rules that the production data cannot cheaply guard:
a clinical CSV keyed per-session must not collapse onto a per-patient key,
and a fact known on disk must survive into the manifest.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _row(df: pd.DataFrame, dataset: str, patient_id: str, session_index: int) -> pd.Series:
    m = df[(df["dataset"] == dataset)
           & (df["patient_id"] == patient_id)
           & (df["session_index"] == session_index)]
    assert len(m) == 1, f"expected exactly one row for {patient_id} tp{session_index}, got {len(m)}"
    return m.iloc[0]


# ── UPENN: per-session clinical rows must not collapse (C-1) ──────────────── #

class TestUPENNSessionKeying:
    def test_every_clinical_row_survives(self, synthetic_adapters):
        """3 CSV rows must yield 3 records — collapsing on a stripped id loses one."""
        records = synthetic_adapters["upenn_gbm"].load_clinical()
        assert len(records) == 3

    def test_baseline_keeps_its_own_eor(self, synthetic_manifest):
        """_11 says GTR; _21 says 'Not Applicable'. Baseline must keep GTR."""
        row = _row(synthetic_manifest, "upenn_gbm", "UPENN-GBM-00001", 0)
        assert row["eor"] == "GTR"

    def test_baseline_keeps_its_own_age(self, synthetic_manifest):
        """_11 scan age is 50.00; the _21 follow-up scan age is 51.50."""
        row = _row(synthetic_manifest, "upenn_gbm", "UPENN-GBM-00001", 0)
        assert row["age"] == pytest.approx(50.00)

    def test_followup_keeps_its_own_age(self, synthetic_manifest):
        row = _row(synthetic_manifest, "upenn_gbm", "UPENN-GBM-00001", 1)
        assert row["age"] == pytest.approx(51.50)

    def test_single_session_patient_unaffected(self, synthetic_manifest):
        row = _row(synthetic_manifest, "upenn_gbm", "UPENN-GBM-00002", 0)
        assert row["eor"] == "non_GTR"
        assert row["age"] == pytest.approx(60.00)


# ── BraTS: grade is known for OS-less subjects (H-1) ──────────────────────── #

class TestBraTSGradeCoverage:
    def test_grade_known_without_survival_row(self, synthetic_manifest):
        """003 is absent from survival_info but graded HGG in name_mapping."""
        row = _row(synthetic_manifest, "brats2020", "BraTS20_Training_003", 0)
        assert row["who_grade"] == 4

    def test_lgg_grade_preserved(self, synthetic_manifest):
        """004 is LGG — it must be distinguishable from an unknown grade."""
        row = _row(synthetic_manifest, "brats2020", "BraTS20_Training_004", 0)
        assert row["who_grade"] == 2

    def test_os_less_subjects_still_flagged(self, synthetic_manifest):
        """Restoring grade must not fake a survival row."""
        for pid in ("BraTS20_Training_003", "BraTS20_Training_004"):
            row = _row(synthetic_manifest, "brats2020", pid, 0)
            assert not row["clinical_row_found"]
            assert pd.isna(row["os_days"])

    def test_no_grade_is_lost(self, synthetic_manifest):
        b = synthetic_manifest[synthetic_manifest["dataset"] == "brats2020"]
        assert b["who_grade"].notna().all(), "every BraTS subject is graded on disk"


# ── UCSF / RHUH: id and session handling ──────────────────────────────────── #

class TestIdentityHandling:
    def test_ucsf_zero_pad_join(self, synthetic_manifest):
        """4-digit directory id must join to the 3-digit CSV id."""
        row = _row(synthetic_manifest, "ucsf_pdgm", "UCSF-PDGM-0004", 0)
        assert row["clinical_row_found"]
        assert row["age"] == pytest.approx(45.5)
        assert row["eor"] == "GTR"

    def test_rhuh_longitudinal_sessions(self, synthetic_manifest):
        r = synthetic_manifest[synthetic_manifest["dataset"] == "rhuh_gbm"]
        assert sorted(r["session_index"].tolist()) == [0, 1]

    def test_rhuh_censor_flag_inverted(self, synthetic_manifest):
        """right_censored 'no' means the event WAS observed."""
        row = _row(synthetic_manifest, "rhuh_gbm", "RHUH-0001", 0)
        assert row["os_event"] == 1


# ── left-scan: nothing is dropped ─────────────────────────────────────────── #

class TestLeftScan:
    def test_all_sessions_present(self, synthetic_manifest):
        counts = synthetic_manifest.groupby("dataset").size().to_dict()
        assert counts == {
            "brats2020": 4, "rhuh_gbm": 2, "ucsf_pdgm": 1, "upenn_gbm": 3,
        }

    def test_no_nan_strings(self, synthetic_manifest):
        for col in synthetic_manifest.columns:
            assert (synthetic_manifest[col].astype(str) == "NaN").sum() == 0
