"""RHUH-GBM adapter.

Longitudinal: patient dirs contain session subdirs 0/1/2 (0 = preop baseline).
Seg convention is RHUH ({0,1,2,3}); remapped 3->4 only at load time.
right_censored 'no' -> event=1, 'yes' -> event=0.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..core.exceptions import MissingColumnError
from ..core.layout import LayoutIssue
from ..core.schema import ClinicalRecord, Dataset, RawSession, SegConvention
from .base import register_adapter
from .normalize import (event_from_censored_flag, normalize_eor_categorical,
                        normalize_grade, normalize_idh, raw_str, to_float)

log = logging.getLogger(__name__)

_CLINICAL = Path("manifests/clinical_info.csv")
_REQUIRED = {"patient_id", "os_days", "right_censored", "age", "who_grade",
             "eor_category", "idh_status"}


@register_adapter(Dataset.RHUH_GBM)
class RHUHAdapter:
    name = Dataset.RHUH_GBM

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def discover(self) -> Iterator[RawSession]:
        for subj in sorted(self.root.glob("RHUH-*")):
            if not subj.is_dir():
                continue
            pid = subj.name
            for sess in sorted(subj.glob("[0-9]*")):
                if not sess.is_dir():
                    continue
                try:
                    idx = int(sess.name)
                except ValueError:
                    continue

                def rel(tok: str, _pid: str = pid, _idx: int = idx) -> str | None:
                    fname = f"{_pid}_{_idx}_{tok}.nii.gz"
                    p = sess / fname
                    return str(Path(_pid) / str(_idx) / fname) if p.exists() else None

                yield RawSession(
                    dataset=self.name, patient_id=pid, session_index=idx,
                    t1_path=rel("t1"), t1ce_path=rel("t1ce"),
                    t2_path=rel("t2"), flair_path=rel("flair"),
                    seg_path=rel("segmentations"),
                    seg_convention=SegConvention.RHUH, seg_source="rhuh",
                )

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        df = pd.read_csv(self.root / _CLINICAL)
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise MissingColumnError(f"RHUH clinical_info.csv missing columns: {missing}")

        out: dict[str, ClinicalRecord] = {}
        for _, r in df.iterrows():
            pid = str(r["patient_id"]).strip()
            eraw = r["eor_category"]
            out[pid] = ClinicalRecord(
                patient_id=pid,
                age=to_float(r["age"]),
                os_days=to_float(r["os_days"]),
                os_event=event_from_censored_flag(r["right_censored"]),
                who_grade=normalize_grade(r["who_grade"]),
                who_grade_raw=raw_str(r["who_grade"]),
                eor=normalize_eor_categorical(eraw), eor_raw=raw_str(eraw),
                idh_status=normalize_idh(r["idh_status"]),
                mgmt_methylation=None, mgmt_raw=None,
                codeletion_1p19q=None, clinical_row_found=True,
            )
        log.debug("RHUH: loaded %d clinical records", len(out))
        return out

    def clinical_key(self, session: RawSession) -> str:
        return session.patient_id

    def check_layout(self) -> list[LayoutIssue]:
        issues: list[LayoutIssue] = []

        if not self.root.exists():
            issues.append(LayoutIssue(
                severity="error", check="root directory",
                expected=str(self.root),
                fix=f"Create or mount the RHUH-GBM data directory at {self.root}",
            ))
            return issues

        subjects = list(self.root.glob("RHUH-*"))
        if not subjects:
            issues.append(LayoutIssue(
                severity="error", check="patient directories (RHUH-*)",
                expected=str(self.root / "RHUH-*"),
                fix=f"Extract patient RHUH-* directories directly into {self.root}",
            ))

        clinical = self.root / _CLINICAL
        if not clinical.exists():
            issues.append(LayoutIssue(
                severity="error", check="manifests/clinical_info.csv",
                expected=str(clinical),
                fix=f"Ensure manifests/clinical_info.csv is present inside {self.root}",
            ))

        return issues
