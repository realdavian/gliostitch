"""Audit stage: discover sessions across all adapters, emit raw_inventory parquets."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..adapters.base import DatasetAdapter
from ..core.schema import Dataset, RawSession
from ..infra.io import write_parquet
from ..infra.parallel import thread_map

log = logging.getLogger(__name__)


def _discover_one(adapter: DatasetAdapter) -> list[dict]:
    sessions = list(adapter.discover())
    log.info("%s: discovered %d sessions", adapter.name.value, len(sessions))
    rows = []
    for s in sessions:
        d = {
            "dataset": s.dataset.value,
            "patient_id": s.patient_id,
            "session_index": s.session_index,
            "global_session_key": s.global_session_key,
            "t1_path": s.t1_path,
            "t1ce_path": s.t1ce_path,
            "t2_path": s.t2_path,
            "flair_path": s.flair_path,
            "seg_path": s.seg_path,
            "seg_convention": s.seg_convention.value if s.seg_convention else None,
            "seg_source": s.seg_source,
            "has_t1": s.t1_path is not None,
            "has_t1ce": s.t1ce_path is not None,
            "has_t2": s.t2_path is not None,
            "has_flair": s.flair_path is not None,
            "has_seg": s.seg_path is not None,
        }
        rows.append(d)
    return rows


class Auditor:
    def __init__(self, adapters: list[DatasetAdapter], output_dir: Path,
                 workers: int = 4) -> None:
        self.adapters = adapters
        self.output_dir = output_dir / "raw_inventory"
        self.workers = workers

    def run(self, force: bool = False) -> dict[Dataset, pd.DataFrame]:
        results: dict[Dataset, pd.DataFrame] = {}
        pending = []
        for adapter in self.adapters:
            out_path = self.output_dir / f"{adapter.name.value}.parquet"
            if not force and out_path.exists():
                log.info("audit: loading cached %s", out_path)
                results[adapter.name] = pd.read_parquet(out_path)
            else:
                pending.append(adapter)

        if pending:
            all_rows = thread_map(
                _discover_one, pending, workers=self.workers, desc="audit"
            )
            for adapter, rows in zip(pending, all_rows):
                df = pd.DataFrame(rows)
                write_parquet(df, self.output_dir / f"{adapter.name.value}.parquet")
                results[adapter.name] = df
                log.info("audit: wrote %d rows for %s", len(df), adapter.name.value)

        return results
