"""Cohort stage: emit the derived study cohort as a convenience artifact.

This stage owns no research decisions. Eligibility rules, survival thresholds,
the held-out cohort and the duplicate priority all live in gbm_os.studies —
the manifest records facts, and a study is one interpretation of them
(spec 01 §8: the selector belongs to the loader/runtime, not the manifest).

The pipeline may still *emit* selected.csv and exclusions.csv so the cohort
table is regenerable from a single command, which is what this stage does: it
loads the study definition, applies it, and writes the result. It is the only
module in gbm_manifest that depends on gbm_os, and the import is deferred so
the rest of the pipeline stays independent of the selection layer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import CohortStageConfig
from ..infra import cache
from ..infra.io import write_csv

log = logging.getLogger(__name__)


class CohortSelector:
    def __init__(self, output_dir: Path, cfg: CohortStageConfig,
                 data_roots: dict[str, Path] | None = None,
                 fingerprint: str = "") -> None:
        self.output_dir = output_dir / "cohort"
        self.cfg = cfg
        self.data_roots = data_roots or {}
        self.fingerprint = fingerprint

    def run(self, manifest_path: Path, force: bool = False) -> pd.DataFrame:
        sel_path = self.output_dir / "selected.csv"
        exc_path = self.output_dir / "exclusions.csv"

        if not force and cache.is_valid(sel_path, self.fingerprint):
            log.info("cohort: loading cached %s", sel_path)
            return pd.read_csv(sel_path)

        from gbm_os.studies import get_study

        study = get_study(self.cfg.study)
        view = study.load(manifest_path, self.data_roots)

        selected = view.to_frame()
        exclusions = view.exclusions()

        log.info("cohort: study %s v%s selected %d of %d sessions",
                 study.name, study.version, len(selected),
                 view.provenance().n_input)
        for step in view.provenance().steps:
            log.debug("cohort:   %s", step)

        write_csv(selected, sel_path)
        cache.record(sel_path, self.fingerprint)

        if not exclusions.empty:
            write_csv(exclusions, exc_path)
            log.info("cohort: excluded %d sessions across %d reasons",
                     len(exclusions), exclusions["exclusion_reason"].nunique())

        return selected
