"""UCSF-PDGM adapter.

Flat per-patient dir '<id>_nifti/' with RAW modalities (_T1/_T1c/_T2/_FLAIR).
ID mismatch: directory 'UCSF-PDGM-0004' (4-digit) vs CSV 'UCSF-PDGM-004' (3-digit).
Event from literal '1-dead 0-alive' column.
MGMT 'positive'/'negative' -> methylated/unmethylated.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..core.exceptions import MissingColumnError
from ..core.schema import ClinicalRecord, Dataset, RawSession, SegConvention
from .base import register_adapter
from .normalize import (event_from_int, normalize_eor_categorical,
                        normalize_grade, normalize_idh, normalize_mgmt,
                        raw_str, to_float)

log = logging.getLogger(__name__)

_V5 = Path("UCSF-PDGM-v5")
_MODS = {"t1": "T1", "t1ce": "T1c", "t2": "T2", "flair": "FLAIR",
         "seg": "tumor_segmentation"}
_NUM = re.compile(r"(\d+)")

_REQUIRED = {"ID", "OS", "1-dead 0-alive", "Age at MRI", "WHO CNS Grade",
             "EOR", "IDH", "MGMT status"}


@register_adapter(Dataset.UCSF_PDGM)
class UCSFAdapter:
    name = Dataset.UCSF_PDGM

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def discover(self) -> Iterator[RawSession]:
        for subj in sorted((self.root / _V5).glob("*_nifti")):
            if not subj.is_dir():
                continue
            pid = subj.name[: -len("_nifti")]  # UCSF-PDGM-0004

            def rel(tok: str, _pid: str = pid) -> str | None:
                fname = f"{_pid}_{tok}.nii.gz"
                return str(_V5 / subj.name / fname) if (subj / fname).exists() else None

            yield RawSession(
                dataset=self.name, patient_id=pid, session_index=0,
                t1_path=rel(_MODS["t1"]), t1ce_path=rel(_MODS["t1ce"]),
                t2_path=rel(_MODS["t2"]), flair_path=rel(_MODS["flair"]),
                seg_path=rel(_MODS["seg"]),
                seg_convention=SegConvention.BRATS_LEGACY, seg_source="tumor",
            )

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        df = pd.read_csv(self.root / "UCSF-PDGM-metadata_v5.csv")
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise MissingColumnError(f"UCSF metadata_v5.csv missing columns: {missing}")

        out: dict[str, ClinicalRecord] = {}
        for _, r in df.iterrows():
            cid = str(r["ID"]).strip()  # UCSF-PDGM-004 (3-digit in CSV)
            mraw = r["MGMT status"]
            out[cid] = ClinicalRecord(
                patient_id=cid,
                age=to_float(r["Age at MRI"]),
                os_days=to_float(r["OS"]),
                os_event=event_from_int(r["1-dead 0-alive"]),
                who_grade=normalize_grade(r["WHO CNS Grade"]),
                who_grade_raw=raw_str(r["WHO CNS Grade"]),
                eor=normalize_eor_categorical(r["EOR"]), eor_raw=raw_str(r["EOR"]),
                idh_status=normalize_idh(r["IDH"]),
                mgmt_methylation=normalize_mgmt(mraw), mgmt_raw=raw_str(mraw),
                codeletion_1p19q=None, clinical_row_found=True,
            )
        log.debug("UCSF: loaded %d clinical records", len(out))
        return out

    def clinical_key(self, session: RawSession) -> str:
        """dir 'UCSF-PDGM-0004' -> csv key 'UCSF-PDGM-004' (3-digit)."""
        m = _NUM.search(session.patient_id)
        return f"UCSF-PDGM-{int(m.group()):03d}" if m else session.patient_id
