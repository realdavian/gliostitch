"""Study definitions own the research decisions, and reproduce their targets."""
from __future__ import annotations

from pathlib import Path

import pytest

from gbm_os.studies import GBM_OS_STUDY, M6_RECONSTRUCTION, STUDIES, get_study


class TestRegistry:
    def test_lookup(self):
        assert get_study("gbm-os") is GBM_OS_STUDY

    def test_unknown_study_lists_options(self):
        with pytest.raises(KeyError, match="gbm-os"):
            get_study("nope")

    def test_definitions_are_declarative(self):
        """No predicates: a study must be printable, diffable and versioned."""
        for study in STUDIES.values():
            for value in study.filters.values():
                assert not callable(value)
            assert study.version
            assert study.description

    def test_describe_names_every_criterion(self):
        text = GBM_OS_STUDY.describe()
        for key in GBM_OS_STUDY.filters:
            assert key in text


@pytest.fixture(scope="module")
def view(cohort):
    return GBM_OS_STUDY.apply(cohort)


class TestGBMOSStudy:
    def test_selected_count(self, view):
        assert len(view) == 502

    def test_external_arm(self, view):
        df = view.to_frame()
        assert (df["dataset"] == "upenn_gbm").sum() == 125

    def test_accounts_for_every_session(self, view):
        assert len(view) + len(view.exclusions()) == 1661

    def test_all_selected_are_gtr_baseline_complete(self, view):
        df = view.to_frame()
        assert (df["eor"] == "GTR").all()
        assert (df["session_index"] == 0).all()
        assert df["is_structural_complete"].all()

    def test_grade_rule_admits_null_but_not_low_grade(self, view):
        """UPENN records no grade; BraTS LGG and UCSF II/III are excluded."""
        grades = view.to_frame()["who_grade"]
        assert set(grades.dropna().unique()) == {4}
        assert grades.isna().any(), "UPENN rows should survive on a null grade"


class TestM6Reconstruction:
    def test_reproduces_documented_external_arm(self, cohort):
        """Spec 01 M6 omits has-OS and documents a 131-session external arm."""
        df = M6_RECONSTRUCTION.apply(cohort).to_frame()
        assert (df["dataset"] == "upenn_gbm").sum() == 131

    def test_differs_from_study_only_by_has_os(self, cohort):
        study = set(GBM_OS_STUDY.apply(cohort).to_frame()["global_session_key"])
        m6 = set(M6_RECONSTRUCTION.apply(cohort).to_frame()["global_session_key"])
        assert study < m6
        extra = cohort.select().to_frame().set_index("global_session_key").loc[
            sorted(m6 - study)]
        assert extra["os_days"].isna().all()


class TestPhase1Delegates:
    def test_cohort_stage_owns_no_criteria(self):
        """The pipeline stage must not re-encode eligibility rules."""
        source = Path("gbm_manifest/stages/cohort.py").read_text()
        for leaked in ("GTR", "who_grade", "session_index", "has_t1", "_PRIORITY"):
            assert leaked not in source, f"{leaked!r} leaked back into the pipeline"
