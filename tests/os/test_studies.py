"""Study definitions own the research decisions, and reproduce their targets."""
from __future__ import annotations

from pathlib import Path

import pytest

from gbm_os.studies import (
    GBM_OS_ANY_EOR_STUDY,
    GBM_OS_STUDY,
    STUDIES,
    get_study,
)


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
        assert len(view) == 531

    def test_external_arm(self, view):
        df = view.to_frame()
        assert (df["dataset"] == "upenn_gbm").sum() == 125

    def test_accounts_for_every_session(self, view):
        assert len(view) + len(view.exclusions()) == 2109

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


@pytest.fixture(scope="module")
def any_eor_view(cohort):
    return GBM_OS_ANY_EOR_STUDY.apply(cohort)


class TestGBMOSAnyEorStudy:
    """gbm-os-any-eor drops the one criterion that is a post-baseline event."""

    def test_selected_count(self, any_eor_view):
        assert len(any_eor_view) == 925

    def test_partitions(self, any_eor_view):
        assert len(any_eor_view.select(partition="train")) == 721
        assert len(any_eor_view.select(partition="external")) == 204

    def test_eor_is_not_an_eligibility_criterion(self):
        assert "eor" not in GBM_OS_ANY_EOR_STUDY.filters

    def test_admits_every_extent_of_resection(self, any_eor_view):
        """Including the ones gbm-os drops — biopsy-only above all."""
        present = set(any_eor_view.to_frame()["eor"].dropna().unique())
        assert {"GTR", "STR", "biopsy", "non_GTR", "unknown"} <= present

    def test_gbm_os_is_the_gtr_stratum_of_any_eor(self, cohort, any_eor_view):
        """The two studies must nest, so the stricter one stays a sub-analysis.

        This is what makes 'primary on 925, sensitivity on the GTR stratum'
        a single cohort rather than two incomparable ones.
        """
        strict = set(GBM_OS_STUDY.apply(cohort).to_frame()["global_session_key"])
        loose = set(any_eor_view.to_frame()["global_session_key"])
        assert strict < loose

        gtr = any_eor_view.select(filters={"eor": "GTR"})
        assert set(gtr.to_frame()["global_session_key"]) == strict

    def test_grade_rule_is_unchanged_from_gbm_os(self, any_eor_view):
        """Dropping EOR must not quietly widen the disease under study."""
        grades = any_eor_view.to_frame()["who_grade"]
        assert set(grades.dropna().unique()) == {4}

    def test_still_baseline_and_structurally_complete(self, any_eor_view):
        df = any_eor_view.to_frame()
        assert (df["session_index"] == 0).all()
        assert df["is_structural_complete"].all()

    def test_accounts_for_every_session(self, any_eor_view):
        assert len(any_eor_view) + len(any_eor_view.exclusions()) == 2109


class TestPreoperativeIsAsserted:
    """'Preoperative' must be a recorded fact, not an inference from ordering.

    For the original four datasets session_index == 0 and preoperative agreed,
    so this suite was a tautology. LUMIERE ended that: 24 of its local patients
    have no preoperative study on disk, so their earliest session is a
    follow-up sitting at session_index == 0. Without the criterion those enter
    the cohort as baselines, and nothing else would catch it.
    """

    def test_study_requires_it_explicitly(self):
        assert GBM_OS_ANY_EOR_STUDY.filters["acquisition_context"] == "preop"

    def test_no_postoperative_session_is_ever_selected(self, any_eor_view):
        assert (any_eor_view.to_frame()["acquisition_context"] == "preop").all()

    def test_rhuh_follow_ups_are_postoperative(self, cohort):
        """RHUH sessions 1 and 2 follow the resection."""
        df = cohort.select().to_frame()
        rhuh = df[df["dataset"] == "rhuh_gbm"]
        assert (rhuh[rhuh["session_index"] > 0]["acquisition_context"] == "postop").all()
        assert (rhuh[rhuh["session_index"] == 0]["acquisition_context"] == "preop").all()

    def test_upenn_second_timepoint_is_postoperative(self, cohort):
        """UPENN _11 is presurgical, _21 is a follow-up."""
        df = cohort.select().to_frame()
        upenn = df[df["dataset"] == "upenn_gbm"]
        assert (upenn[upenn["session_index"] == 1]["acquisition_context"] == "postop").all()

    def test_unknown_is_confined_to_unrated_lumiere_sessions(self, cohort):
        """UNKNOWN is allowed, but only where the source really is silent.

        LUMIERE leaves some follow-up studies unrated, and one whose patient has
        no rated preoperative study to order it against cannot be placed. Every
        other dataset documents its own timing, so a stray UNKNOWN there means
        an adapter stopped deciding and started guessing.
        """
        df = cohort.select().to_frame()
        unknown = df[df["acquisition_context"] == "unknown"]
        assert set(unknown["dataset"].unique()) <= {"lumiere"}
        assert set(df["acquisition_context"].unique()) <= {"preop", "postop", "unknown"}

    def test_unknown_never_reaches_a_cohort(self, any_eor_view, view):
        """Unplaceable sessions must fail closed, not sneak in as baselines."""
        for v in (any_eor_view, view):
            assert (v.to_frame()["acquisition_context"] == "preop").all()

    def test_criterion_excludes_postoperative_baselines(self, cohort):
        """Dropping the criterion readmits exactly the sessions it exists for.

        Pinning the difference rather than the totals: whatever the cohort size
        becomes, removing the criterion must let in postoperative or unplaceable
        studies and nothing else.
        """
        from dataclasses import replace

        without = replace(GBM_OS_ANY_EOR_STUDY, filters={
            k: v for k, v in GBM_OS_ANY_EOR_STUDY.filters.items()
            if k != "acquisition_context"
        })
        strict = set(GBM_OS_ANY_EOR_STUDY.apply(cohort).to_frame()["global_session_key"])
        loose = without.apply(cohort).to_frame().set_index("global_session_key")

        readmitted = loose.loc[sorted(set(loose.index) - strict)]
        assert len(readmitted) > 0, "LUMIERE should make this criterion load-bearing"
        assert (readmitted["acquisition_context"] != "preop").all()

    def test_criterion_still_costs_the_original_four_datasets_nothing(self, cohort):
        """It must exclude only LUMIERE sessions, never re-cut the old cohorts."""
        from dataclasses import replace

        without = replace(GBM_OS_ANY_EOR_STUDY, filters={
            k: v for k, v in GBM_OS_ANY_EOR_STUDY.filters.items()
            if k != "acquisition_context"
        })
        strict = set(GBM_OS_ANY_EOR_STUDY.apply(cohort).to_frame()["global_session_key"])
        loose = without.apply(cohort).to_frame().set_index("global_session_key")
        readmitted = loose.loc[sorted(set(loose.index) - strict)]
        assert set(readmitted["dataset"].unique()) == {"lumiere"}


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
    """gbm-os is what the pipeline emits unless told otherwise."""

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

    def test_the_has_os_criterion_removes_only_unlabelled_sessions(self, cohort):
        """External is 125, not the 131 an early spec recorded.

        The six extra UPENN sessions meet every imaging and surgical criterion
        but carry no survival annotation, so there is nothing to train on or
        evaluate against. A named study once existed to reproduce that number;
        building the variant here keeps the reconciliation tested without
        shipping a third study nobody would legitimately run.
        """
        from dataclasses import replace

        no_has_os = replace(GBM_OS_STUDY, filters={
            k: v for k, v in GBM_OS_STUDY.filters.items() if k != "has_os"
        })
        strict = set(GBM_OS_STUDY.apply(cohort).to_frame()["global_session_key"])
        loose = no_has_os.apply(cohort).to_frame().set_index("global_session_key")
        extra = sorted(set(loose.index) - strict)

        assert extra, "the criterion must still remove something"
        rows = cohort.select().to_frame().set_index("global_session_key").loc[extra]
        assert rows["os_days"].isna().all()
        assert (rows["dataset"] == "upenn_gbm").sum() == 6
        assert (loose["dataset"] == "upenn_gbm").sum() == 131


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
        assert len(view.select(partition="train")) == 406

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
