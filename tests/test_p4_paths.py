"""P4 — path resolution: every SampleSpec path that is marked present must exist on disk."""
from __future__ import annotations

import os

import pytest


class TestPathResolution:
    def test_all_present_paths_exist(self, cohort):
        """100% of present-modality paths must resolve to existing files."""
        view = cohort.select(baseline_only=True)
        missing: list[str] = []
        checked = 0

        for spec in view:
            for mod, present in spec.present.items():
                if not present:
                    continue
                path = spec.paths.get(mod)
                if path is None:
                    missing.append(
                        f"{spec.global_session_key}/{mod}: present=True but path=None"
                    )
                    continue
                if not os.path.exists(path):
                    missing.append(f"{spec.global_session_key}/{mod}: {path}")
                checked += 1

        assert checked > 0, "No paths were checked"
        assert not missing, f"{len(missing)} paths missing:\n" + "\n".join(missing[:20])

    def test_absent_modalities_have_none_path(self, cohort):
        """Modalities marked absent must have None as resolved path."""
        view = cohort.select(baseline_only=True)
        for spec in view:
            for mod, present in spec.present.items():
                if not present:
                    assert spec.paths.get(mod) is None, (
                        f"{spec.global_session_key}/{mod}: present=False but path is not None"
                    )

    def test_all_datasets_have_resolved_paths(self, cohort):
        """At least one path resolves per dataset."""
        datasets_seen: set[str] = set()
        view = cohort.select(baseline_only=True, require_complete=True)
        for spec in view:
            for mod, present in spec.present.items():
                if present and spec.paths.get(mod):
                    datasets_seen.add(spec.dataset)
        assert datasets_seen == {"brats2020", "rhuh_gbm", "ucsf_pdgm", "upenn_gbm"}
