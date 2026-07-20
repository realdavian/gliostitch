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
        repo = Path(__file__).parents[2]
        source = (repo / "gbm_manifest" / "stages" / "cohort.py").read_text()
        for leaked in ("GTR", "who_grade", "session_index", "has_t1", "_PRIORITY"):
            assert leaked not in source, f"{leaked!r} leaked back into the pipeline"


class TestCensoringIsDownstream:
    """The manifest records censored outcomes; using them is a modelling choice.

    Dropping censored rows inside the study would bake a modelling assumption
    into the cohort and make the alternative unreachable.
    """

    def test_study_retains_censored_cases(self, view):
        df = view.to_frame()
        assert (df["os_event"] == 0).sum() > 0

    def test_the_one_brats_censored_case_survives(self, view):
        """BraTS20_Training_084 — GTR, grade IV, 361 days, censored.

        The only censored subject among BraTS's 236 survival rows, and the
        reason training is 377 rather than the originally recorded 376.
        """
        df = view.to_frame()
        row = df[df["patient_id"] == "BraTS20_Training_084"]
        assert len(row) == 1
        assert row.iloc[0]["os_event"] == 0
        assert row.iloc[0]["os_days"] == 361.0

    def test_deceased_only_is_reachable_downstream(self, view):
        deceased = view.select(filters={"os_event": 1})
        df = deceased.to_frame()
        assert (df["os_event"] == 1).all()
        assert (df["dataset"] != "upenn_gbm").sum() == 265

    def test_downstream_filter_is_traced(self, view):
        deceased = view.select(filters={"os_event": 1})
        reasons = set(deceased.exclusions()["exclusion_reason"].unique())
        assert "filter:os_event" in reasons


class TestCanonicalStudy:
    """gbm-os is the study cohort; gbm-os-m6 exists only to explain an old number."""

    def test_canonical_points_at_gbm_os(self):
        from gbm_os.studies import CANONICAL

        assert get_study(CANONICAL) is GBM_OS_STUDY

    def test_pipeline_config_uses_the_canonical_study(self):
        """config/pipeline.yaml must not ship pointing at a reconciliation study."""
        import yaml

        from gbm_os.studies import CANONICAL

        repo = Path(__file__).parents[2]
        cfg = yaml.safe_load((repo / "config" / "pipeline.yaml").read_text())
        assert cfg["cohort"]["study"] == CANONICAL

    def test_reconciliation_study_is_labelled_as_such(self):
        assert "RECONCILIATION ONLY" in M6_RECONSTRUCTION.description

    def test_the_six_extra_cases_have_no_survival_label(self, cohort):
        """The whole basis of the decision: they cannot be trained on."""
        study = set(GBM_OS_STUDY.apply(cohort).to_frame()["global_session_key"])
        m6 = set(M6_RECONSTRUCTION.apply(cohort).to_frame()["global_session_key"])
        extra = sorted(m6 - study)

        assert len(extra) == 6
        rows = cohort.select().to_frame().set_index("global_session_key").loc[extra]
        assert rows["os_days"].isna().all()
        assert (rows["dataset"] == "upenn_gbm").all()

    def test_training_arm_is_unaffected_by_the_choice(self, cohort):
        """The criterion only ever removes UPENN, so training is 377 either way."""
        def train(study):
            df = study.apply(cohort).to_frame()
            return (df["dataset"] != "upenn_gbm").sum()

        assert train(GBM_OS_STUDY) == train(M6_RECONSTRUCTION) == 377


class TestStudyPoliciesTravelWithIt:
    """Applying a study must carry its policies, not just its criteria.

    A Cohort built without a config has an empty partition map and priority
    list. Applying a study to it used to produce a view where
    select(partition="external") returned nothing — silently, with no error,
    which is exactly the path the README documents.
    """

    def _bare_cohort(self):
        from gbm_os import Cohort
        from tests.conftest import DATA_ROOTS, MANIFEST_PATH

        return Cohort.from_manifest(MANIFEST_PATH, data_roots=DATA_ROOTS)

    def test_partition_works_on_a_config_less_cohort(self, cohort):
        view = GBM_OS_STUDY.apply(self._bare_cohort())
        assert len(view.select(partition="external")) == 125
        assert len(view.select(partition="train")) == 377

    def test_apply_matches_load(self, cohort):
        """The two documented entry points must agree."""
        from tests.conftest import DATA_ROOTS, MANIFEST_PATH

        via_apply = GBM_OS_STUDY.apply(self._bare_cohort())
        via_load = GBM_OS_STUDY.load(MANIFEST_PATH, DATA_ROOTS)

        assert len(via_apply) == len(via_load)
        for part in ("train", "external"):
            assert len(via_apply.select(partition=part)) == \
                   len(via_load.select(partition=part))

    def test_priority_is_carried(self, cohort):
        view = GBM_OS_STUDY.apply(self._bare_cohort())
        assert tuple(view._config.priority) == GBM_OS_STUDY.priority

    def test_thresholds_are_redderived_when_they_differ(self, cohort):
        """A study banding survival differently must not report another
        study's os_class."""
        import dataclasses

        wide = dataclasses.replace(GBM_OS_STUDY, name="wide", os_thresholds=(100, 200))
        default_view = GBM_OS_STUDY.apply(self._bare_cohort())
        wide_view = wide.apply(self._bare_cohort())

        assert len(default_view) == len(wide_view)          # same eligibility
        d = default_view.to_frame()["os_class"].value_counts().to_dict()
        w = wide_view.to_frame()["os_class"].value_counts().to_dict()
        assert d != w, "os_class should differ under different thresholds"
