"""P6 — cohort_summary: structure, correctness, and per-dataset counts."""
from __future__ import annotations

import pytest

from gbm_os import cohort_summary
from gbm_os.summary import CohortSummary


class TestSummaryStructure:
    def test_returns_cohort_summary(self, cohort):
        s = cohort_summary(cohort.select())
        assert isinstance(s, CohortSummary)

    def test_spatial_none_by_default(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.spatial is None

    def test_clinical_has_total_row(self, cohort):
        s = cohort_summary(cohort.select())
        assert "TOTAL" in s.clinical.index

    def test_clinical_has_all_datasets(self, cohort):
        s = cohort_summary(cohort.select())
        datasets = set(s.clinical.index) - {"TOTAL"}
        assert datasets == {"brats2020", "lumiere", "rhuh_gbm", "ucsf_pdgm", "upenn_gbm"}

    def test_imaging_has_total_row(self, cohort):
        s = cohort_summary(cohort.select())
        assert "TOTAL" in s.imaging.index

    def test_distributions_has_expected_keys(self, cohort):
        s = cohort_summary(cohort.select())
        assert set(s.distributions.keys()) == {"eor", "idh_status", "mgmt_methylation", "who_grade"}

    def test_str_is_non_empty(self, cohort):
        s = cohort_summary(cohort.select())
        text = str(s)
        assert len(text) > 100
        assert "COHORT SUMMARY" in text


class TestSessionCounts:
    def test_total_sessions_match_view(self, cohort):
        view = cohort.select()
        s = cohort_summary(view)
        assert s.clinical.loc["TOTAL", "n_sessions"] == len(view)

    def test_brats_session_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["brats2020", "n_sessions"] == 369

    def test_rhuh_session_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["rhuh_gbm", "n_sessions"] == 120

    def test_ucsf_session_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["ucsf_pdgm", "n_sessions"] == 501

    def test_upenn_session_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["upenn_gbm", "n_sessions"] == 671

    def test_rhuh_longitudinal_count(self, cohort):
        """RHUH has 40 patients with 3 sessions each — 40 longitudinal."""
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["rhuh_gbm", "n_longitudinal"] == 40

    def test_brats_no_longitudinal(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["brats2020", "n_longitudinal"] == 0


class TestPatientCounts:
    def test_rhuh_patient_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["rhuh_gbm", "n_patients"] == 40

    def test_brats_patient_count(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["brats2020", "n_patients"] == 369

    def test_filtered_view_reduces_count(self, cohort):
        full = cohort_summary(cohort.select())
        filtered = cohort_summary(cohort.select(baseline_only=True, require_complete=True))
        assert filtered.clinical.loc["TOTAL", "n_sessions"] < full.clinical.loc["TOTAL", "n_sessions"]


class TestImagingStats:
    def test_brats_100pct_complete(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.imaging.loc["brats2020", "pct_complete"] == 100.0

    def test_rhuh_100pct_complete(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.imaging.loc["rhuh_gbm", "pct_complete"] == 100.0

    def test_upenn_partial_complete(self, cohort):
        """UPENN has many sessions missing modalities."""
        s = cohort_summary(cohort.select())
        assert s.imaging.loc["upenn_gbm", "pct_complete"] < 100.0

    def test_modality_columns_present(self, cohort):
        s = cohort_summary(cohort.select())
        for col in ["pct_t1", "pct_t1ce", "pct_t2", "pct_flair", "pct_seg", "pct_complete"]:
            assert col in s.imaging.columns, f"missing column: {col}"


class TestClinicalStats:
    def test_rhuh_zero_age_missing(self, cohort):
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["rhuh_gbm", "age_pct_missing"] == 0.0

    def test_brats_age_missing_nonzero(self, cohort):
        """BraTS has 133 patients with no clinical row → missing age."""
        s = cohort_summary(cohort.select())
        assert s.clinical.loc["brats2020", "age_pct_missing"] > 0.0

    def test_event_rate_in_range(self, cohort):
        """A rate must be a percentage, or absent where vital status is.

        LUMIERE records a survival time but no event indicator, so its event
        rate is undefined rather than zero — reporting 0% would assert every
        patient was censored, which is the opposite of what the source says.
        """
        import pandas as pd

        s = cohort_summary(cohort.select())
        for dataset in ["brats2020", "lumiere", "rhuh_gbm", "ucsf_pdgm", "upenn_gbm"]:
            rate = s.clinical.loc[dataset, "event_rate_pct"]
            assert pd.isna(rate) or 0.0 <= rate <= 100.0

    def test_event_rate_is_undefined_only_where_vital_status_is(self, cohort):
        import pandas as pd

        s = cohort_summary(cohort.select())
        undefined = {d for d in s.clinical.index
                     if d != "TOTAL" and pd.isna(s.clinical.loc[d, "event_rate_pct"])}
        assert undefined == {"lumiere"}


class TestDistributions:
    def test_eor_distribution_columns(self, cohort):
        s = cohort_summary(cohort.select())
        dist = s.distributions["eor"]
        assert "TOTAL" in dist.columns
        assert "brats2020" in dist.columns

    def test_eor_gtr_present(self, cohort):
        s = cohort_summary(cohort.select())
        assert "GTR" in s.distributions["eor"].index

    def test_who_grade_values(self, cohort):
        s = cohort_summary(cohort.select())
        idx = set(s.distributions["who_grade"].index)
        assert "4.0" in idx or 4.0 in idx
