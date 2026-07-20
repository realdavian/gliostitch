"""Named study configurations.

A study is a set of research decisions — which sessions are eligible, how
survival is banded, which cohort is held out, who wins a duplicate. Those
decisions belong here, in the selection layer, not in the manifest pipeline:
the manifest records facts, and a study is one interpretation of them.

Definitions are declarative (no predicates) so they can be printed, diffed,
serialised into a methods section, and versioned when the eligibility rules
change. Applying one returns an ordinary CohortView, so everything downstream —
provenance, splitting, backends — works unchanged.

Censoring is deliberately NOT filtered here. The manifest records what is true,
including censored outcomes, and whether a model may use them is a modelling
decision made at selection time:

    view = GBM_OS_STUDY.apply(cohort)              # 502, censoring intact
    deceased = view.select(filters={"os_event": 1})  # 390, complete-case

Dropping censored rows earlier would bake a modelling assumption into the
cohort and make the alternative unreachable. See "Training is 377" in
developer-docs/03_design_decisions.md for the one BraTS case this turns on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from gbm_os.cohort import Cohort, CohortView
    from gbm_os.config import CohortConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudyDefinition:
    """A reproducible set of eligibility criteria and cohort policies."""

    name: str
    version: str
    description: str

    # Eligibility
    baseline_only: bool = True
    require_complete: bool = True
    filters: Mapping[str, Any] = field(default_factory=dict)

    # Cohort policy
    os_thresholds: tuple[int, int] = (300, 450)
    external_datasets: frozenset[str] = frozenset()
    priority: tuple[str, ...] = ()
    resolve_duplicates: str = "drop"

    def config(self, data_roots: Mapping[str, Path]) -> "CohortConfig":
        """A CohortConfig carrying this study's policies."""
        from gbm_os.config import CohortConfig

        return CohortConfig(
            data_roots={k: Path(v) for k, v in data_roots.items()},
            os_thresholds=self.os_thresholds,
            partition_map={"external": set(self.external_datasets)},
            priority=list(self.priority),
            resolve_duplicates=self.resolve_duplicates,
        )

    def apply(self, cohort: "Cohort") -> "CohortView":
        """Select this study's cohort from a loaded Cohort."""
        view = cohort.select(
            baseline_only=self.baseline_only,
            require_complete=self.require_complete,
            filters=dict(self.filters),
            resolve_duplicates=self.resolve_duplicates,
        )
        logger.info("study %s v%s: selected %d sessions",
                    self.name, self.version, len(view))
        return view

    def load(self, manifest_path: str | Path,
             data_roots: Mapping[str, Path]) -> "CohortView":
        """Load the manifest and apply this study in one step."""
        from gbm_os.cohort import Cohort

        cfg = self.config(data_roots)
        cohort = Cohort.from_manifest(manifest_path, data_roots=dict(data_roots),
                                      config=cfg)
        return self.apply(cohort)

    def describe(self) -> str:
        lines = [f"{self.name} v{self.version}", self.description, "",
                 "Eligibility:"]
        if self.baseline_only:
            lines.append("  - baseline session only (session_index == 0)")
        if self.require_complete:
            lines.append("  - all four structural modalities present")
        for key, val in self.filters.items():
            lines.append(f"  - {key} in {val}" if isinstance(val, (list, tuple, set))
                         else f"  - {key} == {val!r}")
        lines += [
            "",
            "Policy:",
            f"  - os_class thresholds: <{self.os_thresholds[0]} / "
            f"<{self.os_thresholds[1]} / >=",
            f"  - external cohort: {sorted(self.external_datasets) or 'none'}",
            f"  - duplicate policy: {self.resolve_duplicates} "
            f"(priority {list(self.priority)})",
        ]
        return "\n".join(lines)


# ── the GBM-OS survival study ─────────────────────────────────────────────── #

_PRIORITY = ("brats2020", "ucsf_pdgm", "rhuh_gbm", "upenn_gbm")

GBM_OS_STUDY = StudyDefinition(
    name="gbm-os",
    version="2",
    description=(
        "Overall-survival classification over baseline preoperative GBM MRI, "
        "held out on UPENN-GBM."
    ),
    baseline_only=True,
    require_complete=True,
    filters={
        "eor": "GTR",
        # Grade IV where grade is recorded. A null admits UPENN, which ships no
        # grade column because the cohort is GBM by construction; it excludes
        # BraTS LGG and UCSF grade II/III, which are recorded explicitly.
        "who_grade": [4, None],
        # A survival study cannot use a session with no survival label.
        # NOTE: spec 01 M6 omits this criterion and spec 02 P2 includes it —
        # see M6_RECONSTRUCTION for the M6 wording.
        "has_os": True,
    },
    os_thresholds=(300, 450),
    external_datasets=frozenset({"upenn_gbm"}),
    priority=_PRIORITY,
    resolve_duplicates="drop",
)


#: The study cohort. Everything else in this module is reconciliation.
CANONICAL = "gbm-os"


M6_RECONSTRUCTION = StudyDefinition(
    name="gbm-os-m6",
    version="2",
    description=(
        "RECONCILIATION ONLY — not the study cohort. Reproduces spec 01 M6's "
        "original wording, which omitted has-OS and therefore counted 131 "
        "external sessions. The extra six are UPENN patients with no survival "
        "annotation: they meet every imaging and surgical criterion but carry "
        "no label to train on or evaluate against. Use GBM_OS_STUDY."
    ),
    baseline_only=True,
    require_complete=True,
    filters={"eor": "GTR", "who_grade": [4, None]},
    os_thresholds=(300, 450),
    external_datasets=frozenset({"upenn_gbm"}),
    priority=_PRIORITY,
    resolve_duplicates="drop",
)


STUDIES: dict[str, StudyDefinition] = {
    GBM_OS_STUDY.name: GBM_OS_STUDY,
    M6_RECONSTRUCTION.name: M6_RECONSTRUCTION,
}


def get_study(name: str) -> StudyDefinition:
    try:
        return STUDIES[name]
    except KeyError:
        raise KeyError(
            f"unknown study {name!r}; available: {sorted(STUDIES)}"
        ) from None
