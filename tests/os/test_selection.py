"""P2 — flexible, composable selection + GBM-OS regression."""
from __future__ import annotations


class TestEmptySelection:
    def test_all_rows_returned(self, cohort):
        view = cohort.select()
        assert len(view) == 2109


class TestBaselineFilter:
    def test_baseline_only(self, cohort):
        view = cohort.select(baseline_only=True)
        df = view.to_frame()
        assert (df["session_index"] == 0).all()
        assert len(df) == 1590  # 369+40+501+611+69

    def test_non_baseline_excluded(self, cohort):
        view = cohort.select(baseline_only=True)
        df = view.to_frame()
        assert (df["is_baseline"]).all()


class TestCompletenessFilter:
    def test_require_complete(self, cohort):
        view = cohort.select(require_complete=True)
        df = view.to_frame()
        assert df["is_structural_complete"].all()

    def test_complete_count_higher_than_baseline_complete(self, cohort):
        """More sessions are complete without the baseline constraint."""
        all_complete = len(cohort.select(require_complete=True))
        baseline_complete = len(cohort.select(baseline_only=True, require_complete=True))
        assert all_complete >= baseline_complete

    def test_flip_completeness_changes_count(self, cohort):
        """Removing require_complete from a selection increases or keeps count."""
        with_complete = len(cohort.select(
            baseline_only=True, require_complete=True,
            filters={"eor": "GTR", "has_os": True},
        ))
        without_complete = len(cohort.select(
            baseline_only=True, require_complete=False,
            filters={"eor": "GTR", "has_os": True},
        ))
        assert without_complete >= with_complete


class TestDatasetFilter:
    def test_single_dataset(self, cohort):
        view = cohort.select(datasets=["rhuh_gbm"])
        df = view.to_frame()
        assert set(df["dataset"].unique()) == {"rhuh_gbm"}
        assert len(df) == 120

    def test_multi_dataset(self, cohort):
        view = cohort.select(datasets=["rhuh_gbm", "brats2020"])
        df = view.to_frame()
        assert set(df["dataset"].unique()) == {"rhuh_gbm", "brats2020"}


class TestClinicalFilters:
    def test_eor_filter(self, cohort):
        view = cohort.select(filters={"eor": "GTR"})
        df = view.to_frame()
        assert (df["eor"] == "GTR").all()

    def test_has_os_filter(self, cohort):
        view = cohort.select(filters={"has_os": True})
        df = view.to_frame()
        assert df["os_days"].notna().all()

    def test_list_filter(self, cohort):
        view = cohort.select(filters={"eor": ["GTR", "STR"]})
        df = view.to_frame()
        assert df["eor"].isin(["GTR", "STR"]).all()


class TestWherePredicate:
    def test_where_applied(self, cohort):
        view = cohort.select(where=lambda r: r["session_index"] == 0)
        df = view.to_frame()
        assert (df["session_index"] == 0).all()


class TestChaining:
    def test_chain_narrows(self, cohort):
        full = cohort.select()
        baseline = full.select(baseline_only=True)
        complete = baseline.select(require_complete=True)
        assert len(complete) <= len(baseline) <= len(full)

    def test_chain_equivalent_to_single_call(self, cohort):
        chained = (
            cohort.select(baseline_only=True)
            .select(require_complete=True)
            .select(filters={"eor": "GTR"})
        )
        single = cohort.select(
            baseline_only=True,
            require_complete=True,
            filters={"eor": "GTR"},
        )
        assert len(chained) == len(single)


class TestGBMOSRegression:
    """P2 regression: GBM-OS study cohort.

    Criteria: preop ∩ baseline ∩ complete ∩ GTR ∩ grade-IV[UCSF] ∩ has-OS
    Config: external=UPENN, resolve_duplicates="drop"
    Expected count: 531 (502 from the original four, plus 29 LUMIERE)

    Was 488 before the UPENN session-keying fix. clinical_info.csv holds one row
    per session, and 41 patients have both a _11 baseline and a _21 follow-up;
    keying on the suffix-stripped patient_id let the follow-up row overwrite the
    baseline, which carried an EOR of "Not Applicable" into 41 baseline sessions.
    Restoring the per-session key returns 14 GTR cases to the external arm.
    """

    def _gbm_os_view(self, cohort):
        return cohort.select(
            baseline_only=True,
            require_complete=True,
            filters={"eor": "GTR", "has_os": True,
                     "acquisition_context": "preop"},
            where=lambda r: r["dataset"] != "ucsf_pdgm" or r["who_grade"] == 4,
            resolve_duplicates="drop",
        )

    def test_cohort_count(self, cohort):
        view = self._gbm_os_view(cohort)
        assert len(view) == 531

    def test_external_arm_without_has_os(self, cohort):
        """Omitting has-OS from the cohort definition gives an external arm of
        the external arm is exactly 131 (132 pass criteria, 1 dup removed)."""
        view = cohort.select(
            baseline_only=True,
            require_complete=True,
            filters={"eor": "GTR", "acquisition_context": "preop"},
            where=lambda r: r["dataset"] != "ucsf_pdgm" or r["who_grade"] == 4,
            resolve_duplicates="drop",
        )
        df = view.to_frame()
        assert (df["dataset"] == "upenn_gbm").sum() == 131

    def test_deterministic(self, cohort):
        """Same criteria → same count every time."""
        v1 = self._gbm_os_view(cohort)
        v2 = self._gbm_os_view(cohort)
        assert len(v1) == len(v2)

    def test_all_baseline(self, cohort):
        df = self._gbm_os_view(cohort).to_frame()
        assert df["is_baseline"].all()

    def test_all_complete(self, cohort):
        df = self._gbm_os_view(cohort).to_frame()
        assert df["is_structural_complete"].all()

    def test_all_gtr(self, cohort):
        df = self._gbm_os_view(cohort).to_frame()
        assert (df["eor"] == "GTR").all()

    def test_all_have_os(self, cohort):
        df = self._gbm_os_view(cohort).to_frame()
        assert df["os_days"].notna().all()

    def test_ucsf_grade4_only(self, cohort):
        df = self._gbm_os_view(cohort).to_frame()
        ucsf = df[df["dataset"] == "ucsf_pdgm"]
        assert (ucsf["who_grade"] == 4).all()

    def test_completeness_flip_increases_count(self, cohort):
        with_complete = len(self._gbm_os_view(cohort))
        without = len(cohort.select(
            baseline_only=True,
            require_complete=False,
            filters={"eor": "GTR", "has_os": True},
            where=lambda r: r["dataset"] != "ucsf_pdgm" or r["who_grade"] == 4,
            resolve_duplicates="drop",
        ))
        assert without >= with_complete
