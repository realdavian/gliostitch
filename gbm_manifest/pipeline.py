"""Orchestrator: wires config -> adapters -> stages."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import adapters as _adapters_mod  # noqa: F401 — triggers self-registration
from .adapters.base import get_adapter
from .config import PipelineConfig
from .core.schema import Dataset
from .infra.fs import register_dataset_roots
from .stages.audit import Auditor
from .stages.cohort import CohortSelector
from .stages.deduplicate import Deduplicator
from .stages.manifest import ManifestBuilder
from .stages.standardize import LabelStandardizer

log = logging.getLogger(__name__)

_DATASET_MAP = {
    "brats2020": Dataset.BRATS2020,
    "rhuh_gbm": Dataset.RHUH_GBM,
    "upenn_gbm": Dataset.UPENN_GBM,
    "ucsf_pdgm": Dataset.UCSF_PDGM,
}


class Pipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        register_dataset_roots(cfg.dataset_roots())
        self.adapters = self._load_adapters()
        self.out = cfg.output_dir

    def _load_adapters(self):
        adapters = []
        for name, ds_cfg in self.cfg.datasets.items():
            if not ds_cfg.enabled:
                continue
            ds = _DATASET_MAP[name]
            adapters.append(get_adapter(ds, Path(ds_cfg.root)))
        return adapters

    def _dataset_roots_map(self) -> dict[str, Path]:
        return {
            name: Path(ds_cfg.root)
            for name, ds_cfg in self.cfg.datasets.items()
            if ds_cfg.enabled
        }

    def run_audit(self, force: bool = False) -> dict[Dataset, pd.DataFrame]:
        auditor = Auditor(self.adapters, self.out, workers=self.cfg.workers)
        return auditor.run(force=force)

    def run_standardize(self, inventory: dict[Dataset, pd.DataFrame],
                        force: bool = False) -> dict[Dataset, pd.DataFrame]:
        standardizer = LabelStandardizer(self.adapters, self.out)
        return standardizer.run(inventory, force=force)

    def run_dedup(self, standardized: dict[Dataset, pd.DataFrame],
                  force: bool = False) -> pd.DataFrame:
        deduplicator = Deduplicator(
            dataset_roots=self._dataset_roots_map(),
            output_dir=self.out,
            cfg=self.cfg.dedup,
            workers=self.cfg.workers,
        )
        return deduplicator.run(standardized, force=force)

    def run_manifest(self, combined: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        builder = ManifestBuilder(self.out)
        return builder.run(combined, force=force)

    def run_cohort(self, manifest: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        selector = CohortSelector(self.out, self.cfg.cohort)
        return selector.run(manifest, force=force)

    def build(self, force: bool = False) -> pd.DataFrame:
        log.info("=== audit ===")
        inventory = self.run_audit(force=force)
        log.info("=== standardize ===")
        standardized = self.run_standardize(inventory, force=force)
        log.info("=== dedup ===")
        combined = self.run_dedup(standardized, force=force)
        log.info("=== manifest ===")
        manifest = self.run_manifest(combined, force=force)
        log.info("=== cohort ===")
        self.run_cohort(manifest, force=force)
        log.info("=== done ===  manifest: %s", self.out / "master_manifest.csv")
        return manifest
