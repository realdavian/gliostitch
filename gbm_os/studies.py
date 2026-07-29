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

    view = GBM_OS_STUDY.apply(cohort)              # 531, censoring intact
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
        """Select this study's cohort from a loaded Cohort.

        The study's POLICIES travel with it, not just its criteria. A cohort
        built without a config carries an empty partition map and priority
        list, so applying a study to it used to yield a view where
        `select(partition="external")` returned nothing and duplicate
        resolution had no ordering to work from — silently, with no error.
        The view is therefore rebound to a config derived from this study,
        keeping the cohort's data roots.
        """
        from gbm_os.cohort import Cohort

        cfg = self.config(cohort._data_roots)
        df = cohort._df

        # os_class is derived at load time from whatever thresholds the cohort
        # was built with. If this study bands survival differently, recompute
        # rather than silently reporting another study's classes.
        if tuple(cohort._config.os_thresholds) != tuple(self.os_thresholds):
            from gbm_os.manifest import derive_os_class

            logger.info("study %s: re-deriving os_class for thresholds %s "
                        "(cohort was loaded with %s)",
                        self.name, self.os_thresholds,
                        tuple(cohort._config.os_thresholds))
            df = df.copy()
            df["os_class"] = derive_os_class(df["os_days"], *self.os_thresholds)

        scoped = Cohort(df, cohort._data_roots, cfg)
        view = scoped.select(
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

_PRIORITY = ("brats2020", "ucsf_pdgm", "rhuh_gbm", "lumiere", "upenn_gbm")

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
        # A survival study cannot use a session with no survival label. The
        # cohort was first specified without this criterion, which is why an
        # external arm of 131 was originally recorded; see "Training is 377" in
        # developer-docs/03_design_decisions.md.
        "has_os": True,
        # This study has always described itself as preoperative; until
        # acquisition_context existed it could only approximate that with
        # baseline_only, which asks a weaker question — "earliest session on
        # record". The two agreed for as long as every cohort's first study was
        # preoperative, and stopped agreeing the moment LUMIERE arrived: 24 of
        # its local patients have no preoperative study on disk, so their
        # earliest session is a follow-up sitting at session_index == 0. This
        # criterion is a no-op on the original four datasets and the reason
        # those postoperative scans are not in the cohort.
        "acquisition_context": "preop",
    },
    os_thresholds=(300, 450),
    external_datasets=frozenset({"upenn_gbm"}),
    priority=_PRIORITY,
    resolve_duplicates="drop",
)


GBM_OS_ANY_EOR_STUDY = StudyDefinition(
    name="gbm-os-any-eor",
    version="1",
    description=(
        "Overall-survival classification over baseline preoperative GBM MRI, "
        "held out on UPENN-GBM. Identical to gbm-os except that extent of "
        "resection is recorded, never required."
    ),
    baseline_only=True,
    require_complete=True,
    filters={
        # No `eor` criterion, and that absence is the point of this study.
        #
        # gbm-os requires eor == GTR. Extent of resection is a *treatment* — it
        # happens after the scan the model predicts from — and it is partly
        # determined by the very imaging phenotype being modelled: tumours are
        # biopsied rather than resected because of where and how they present.
        # Selecting on it therefore conditions on a descendant of the predictor
        # and leaves a cohort no one can identify prospectively. At inference
        # time nobody knows who will get a gross-total resection.
        #
        # who_grade stays: grade is a property the tumour already has when it
        # is scanned, which histopathology later confirms. Dropping it would
        # admit UCSF grade II/III glioma and change the disease under study,
        # which is a different decision from removing a treatment variable.
        "who_grade": [4, None],
        "has_os": True,
        # baseline_only gives session_index == 0, which only means "earliest on
        # record". For a preoperative study that has to be asserted, not
        # assumed: a cohort whose earliest study follows surgery would satisfy
        # baseline_only and be silently wrong.
        "acquisition_context": "preop",
    },
    os_thresholds=(300, 450),
    external_datasets=frozenset({"upenn_gbm"}),
    priority=_PRIORITY,
    resolve_duplicates="drop",
)


#: The study the pipeline emits by default.
CANONICAL = "gbm-os"


#: Two studies, differing in exactly one criterion: whether a gross-total
#: resection is required. Both are preoperative, both band survival the same
#: way, both hold out UPENN-GBM. gbm-os is the stricter of the two and nests
#: inside gbm-os-any-eor as its GTR stratum.
STUDIES: dict[str, StudyDefinition] = {
    GBM_OS_STUDY.name: GBM_OS_STUDY,
    GBM_OS_ANY_EOR_STUDY.name: GBM_OS_ANY_EOR_STUDY,
}


def get_study(name: str) -> StudyDefinition:
    try:
        return STUDIES[name]
    except KeyError:
        raise KeyError(
            f"unknown study {name!r}; available: {sorted(STUDIES)}"
        ) from None
