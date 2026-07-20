"""A selection must be able to account for everything it dropped.

A cohort count is not defensible without the flow that produced it, so the
exclusion reasons must partition the drops exactly — every row that leaves the
facts superset is attributed to the criterion that removed it.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def study_view(cohort):
    return cohort.select(
        baseline_only=True,
        require_complete=True,
        filters={"eor": "GTR", "has_os": True},
        where=lambda r: r["dataset"] != "ucsf_pdgm" or r["who_grade"] == 4,
        resolve_duplicates="drop",
    )


class TestTraceArithmetic:
    def test_selected_plus_excluded_equals_input(self, study_view):
        trace = study_view.provenance()
        assert len(study_view) + len(study_view.exclusions()) == trace.n_input

    def test_input_is_the_full_manifest(self, study_view):
        assert study_view.provenance().n_input == 1661

    def test_steps_chain_without_gaps(self, study_view):
        """Each step must start where the previous one ended."""
        steps = study_view.provenance().steps
        assert steps[0].n_before == study_view.provenance().n_input
        for prev, nxt in zip(steps, steps[1:]):
            assert nxt.n_before == prev.n_after
        assert steps[-1].n_after == len(study_view)

    def test_drops_sum_to_exclusions(self, study_view):
        trace = study_view.provenance()
        assert sum(s.n_dropped for s in trace.steps) == len(trace.exclusions())


class TestExclusionReasons:
    def test_every_excluded_row_has_a_reason(self, study_view):
        exc = study_view.exclusions()
        assert exc["exclusion_reason"].notna().all()

    def test_reasons_name_real_criteria(self, study_view):
        reasons = set(study_view.exclusions()["exclusion_reason"].unique())
        assert reasons <= {
            "baseline_only", "require_complete", "filter:eor", "filter:has_os",
            "where", "resolve_duplicates",
        }
        assert "baseline_only" in reasons

    def test_no_row_excluded_twice(self, study_view):
        exc = study_view.exclusions()
        assert exc["global_session_key"].is_unique

    def test_duplicate_drop_is_attributed(self, study_view):
        exc = study_view.exclusions()
        dropped = exc[exc["exclusion_reason"] == "resolve_duplicates"]
        assert len(dropped) == 1
        assert dropped.iloc[0]["dataset"] == "upenn_gbm"


class TestTraceReporting:
    def test_frame_has_one_row_per_step(self, study_view):
        trace = study_view.provenance()
        assert len(trace.to_frame()) == len(trace.steps)

    def test_str_renders_a_flow(self, study_view):
        text = str(study_view.provenance())
        assert "input" in text and "selected" in text

    def test_empty_selection_has_empty_trace(self, cohort):
        view = cohort.select()
        assert view.provenance().steps == []
        assert view.exclusions().empty
        assert len(view) == 1661
