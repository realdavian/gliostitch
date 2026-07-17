"""BraTS-2020 adapter.

Training split only (validation has no OS and no seg).
Survival_days is dual-encoded -> parse_survival_days yields (days, event).
133 subjects have no survival row: discover() still yields them (left-scan).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..core.exceptions import MissingColumnError
from ..core.schema import ClinicalRecord, Dataset, RawSession, SegConvention
from .base import register_adapter
from .normalize import (normalize_eor_categorical, normalize_grade,
                        parse_survival_days, raw_str, to_float)

log = logging.getLogger(__name__)

_TRAIN = Path("BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")

_SURV_REQUIRED = {"Brats20ID", "Age", "Survival_days", "Extent_of_Resection"}
_NAMES_REQUIRED = {"BraTS_2020_subject_ID", "Grade"}


@register_adapter(Dataset.BRATS2020)
class BraTS2020Adapter:
    name = Dataset.BRATS2020

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def discover(self) -> Iterator[RawSession]:
        train = self.root / _TRAIN
        for subj in sorted(train.glob("BraTS20_Training_*")):
            if not subj.is_dir():
                continue
            sid = subj.name

            def rel(tok: str, _sid: str = sid) -> str | None:
                fname = f"{_sid}_{tok}.nii"
                return str(_TRAIN / _sid / fname) if (subj / fname).exists() else None

            yield RawSession(
                dataset=self.name, patient_id=sid, session_index=0,
                t1_path=rel("t1"), t1ce_path=rel("t1ce"),
                t2_path=rel("t2"), flair_path=rel("flair"),
                seg_path=rel("seg"),
                seg_convention=SegConvention.BRATS_LEGACY, seg_source="brats",
            )

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        surv = pd.read_csv(self.root / _TRAIN / "survival_info.csv")
        names = pd.read_csv(self.root / _TRAIN / "name_mapping.csv")

        missing_s = _SURV_REQUIRED - set(surv.columns)
        if missing_s:
            raise MissingColumnError(f"survival_info.csv missing columns: {missing_s}")
        missing_n = _NAMES_REQUIRED - set(names.columns)
        if missing_n:
            raise MissingColumnError(f"name_mapping.csv missing columns: {missing_n}")

        grade_by_id = {
            str(r["BraTS_2020_subject_ID"]).strip(): r["Grade"]
            for _, r in names.iterrows()
        }
        out: dict[str, ClinicalRecord] = {}
        for _, r in surv.iterrows():
            sid = str(r["Brats20ID"]).strip()
            days, event = parse_survival_days(r["Survival_days"])
            graw = grade_by_id.get(sid)
            out[sid] = ClinicalRecord(
                patient_id=sid,
                age=to_float(r["Age"]),
                os_days=days, os_event=event,
                who_grade=normalize_grade(graw), who_grade_raw=raw_str(graw),
                eor=normalize_eor_categorical(r["Extent_of_Resection"]),
                eor_raw=raw_str(r["Extent_of_Resection"]),
                idh_status=None, mgmt_methylation=None, mgmt_raw=None,
                codeletion_1p19q=None, clinical_row_found=True,
            )
        log.debug("BraTS2020: loaded %d clinical records", len(out))
        return out

    def clinical_key(self, session: RawSession) -> str:
        return session.patient_id
