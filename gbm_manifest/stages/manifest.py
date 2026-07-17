"""Manifest stage: assemble final master_manifest.csv."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..core.schema import MANIFEST_COLUMNS
from ..infra.io import write_csv

log = logging.getLogger(__name__)


class ManifestBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def run(self, combined: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        out_path = self.output_dir / "master_manifest.csv"
        if not force and out_path.exists():
            log.info("manifest: loading cached %s", out_path)
            return pd.read_csv(out_path)

        # Ensure all manifest columns exist
        for col in MANIFEST_COLUMNS:
            if col not in combined.columns:
                combined[col] = None

        # Reorder columns to canonical order
        manifest = combined[MANIFEST_COLUMNS].copy()

        # Deterministic sort
        manifest = manifest.sort_values(
            ["dataset", "patient_id", "session_index"],
            kind="stable"
        ).reset_index(drop=True)

        # Ensure clean nulls (no "NaN" strings)
        write_csv(manifest, out_path)
        log.info("manifest: wrote %d rows, %d columns to %s",
                 len(manifest), len(manifest.columns), out_path)
        return manifest
