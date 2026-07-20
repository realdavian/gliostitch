"""Standardize stage: left-join discovered sessions with clinical records."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..adapters.base import DatasetAdapter
from ..core.schema import ClinicalRecord, Dataset
from ..infra import cache
from ..infra.io import write_parquet

log = logging.getLogger(__name__)

_NULL_CLINICAL = ClinicalRecord(
    patient_id="", clinical_row_found=False
)


class LabelStandardizer:
    def __init__(self, adapters: list[DatasetAdapter], output_dir: Path,
                 fingerprint: str = "") -> None:
        self.adapters = {a.name: a for a in adapters}
        self.output_dir = output_dir / "standardized"
        self.fingerprint = fingerprint

    def run(self, inventory: dict[Dataset, pd.DataFrame],
            force: bool = False) -> dict[Dataset, pd.DataFrame]:
        results: dict[Dataset, pd.DataFrame] = {}
        for dataset, df in inventory.items():
            out_path = self.output_dir / f"{dataset.value}.parquet"
            if not force and cache.is_valid(out_path, self.fingerprint):
                log.info("standardize: loading cached %s", out_path)
                results[dataset] = pd.read_parquet(out_path)
                continue

            adapter = self.adapters[dataset]
            clinical = adapter.load_clinical()
            log.info("standardize: %s — %d sessions, %d clinical records",
                     dataset.value, len(df), len(clinical))

            rows = []
            no_match = 0
            for _, session_row in df.iterrows():
                # Build a minimal RawSession-like object for clinical_key
                class _S:
                    patient_id = session_row["patient_id"]
                    session_index = session_row["session_index"]

                key = adapter.clinical_key(_S())  # type: ignore[arg-type]
                rec = clinical.get(key)
                if rec is None:
                    no_match += 1
                    rec = ClinicalRecord(
                        patient_id=session_row["patient_id"],
                        clinical_row_found=False,
                    )

                row = dict(session_row)
                row.update({
                    "age": rec.age,
                    "os_days": rec.os_days,
                    "os_event": rec.os_event,
                    "who_grade": rec.who_grade,
                    "who_grade_raw": rec.who_grade_raw,
                    "eor": rec.eor.value if rec.eor is not None else None,
                    "eor_raw": rec.eor_raw,
                    "idh_status": rec.idh_status,
                    "mgmt_methylation": rec.mgmt_methylation,
                    "mgmt_raw": rec.mgmt_raw,
                    "codeletion_1p19q": rec.codeletion_1p19q,
                    "clinical_row_found": rec.clinical_row_found,
                })
                rows.append(row)

            if no_match:
                log.info("standardize: %s — %d sessions with no clinical match "
                         "(clinical_row_found=False)", dataset.value, no_match)

            std_df = pd.DataFrame(rows)
            write_parquet(std_df, out_path)
            cache.record(out_path, self.fingerprint)
            results[dataset] = std_df
            log.info("standardize: wrote %d rows for %s", len(std_df), dataset.value)

        return results
