"""UPENN-GBM adapter.

Discovers both _11 (session_index=0, presurgical) and _21 (session_index=1) sessions.
Seg files are flat in images_segm/ and automated_segm/ (no per-session subdirectory).
T1ce token is T1GD; seg tokens _segm / _automated_approx_segm.
EOR from binary GTR_over90percent; WHO grade absent (implicit IV).
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
from .normalize import (event_from_status, normalize_eor_binary, normalize_idh,
                        normalize_mgmt, raw_str, to_float)

log = logging.getLogger(__name__)

_NIFTI = Path("imaging/NIfTI")
_STRUCT = _NIFTI / "images_structural"
_SEGM_DIR = _NIFTI / "images_segm"
_AUTO_DIR = _NIFTI / "automated_segm"
_MODS = {"t1": "T1", "t1ce": "T1GD", "t2": "T2", "flair": "FLAIR"}

_REQUIRED = {"ID", "Survival_from_surgery_days_UPDATED", "Survival_Status",
             "Age_at_scan_years", "GTR_over90percent", "IDH1", "MGMT"}

# Map session suffix to session_index
_SUFFIX_IDX = {"11": 0, "21": 1}


@register_adapter(Dataset.UPENN_GBM)
class UPENNAdapter:
    name = Dataset.UPENN_GBM

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def discover(self) -> Iterator[RawSession]:
        for subj in sorted((self.root / _STRUCT).glob("UPENN-GBM-*")):
            if not subj.is_dir():
                continue
            sid = subj.name  # e.g. UPENN-GBM-00001_11
            parts = sid.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in _SUFFIX_IDX:
                continue
            suffix = parts[1]
            session_index = _SUFFIX_IDX[suffix]
            # patient_id strips the _NN suffix
            pid = parts[0]

            def rel(tok: str, _sid: str = sid) -> str | None:
                fname = f"{_sid}_{tok}.nii.gz"
                return str(_STRUCT / _sid / fname) if (subj / fname).exists() else None

            seg_path, seg_source = self._resolve_seg(sid)
            yield RawSession(
                dataset=self.name, patient_id=pid, session_index=session_index,
                t1_path=rel(_MODS["t1"]), t1ce_path=rel(_MODS["t1ce"]),
                t2_path=rel(_MODS["t2"]), flair_path=rel(_MODS["flair"]),
                seg_path=seg_path,
                seg_convention=SegConvention.BRATS_LEGACY, seg_source=seg_source,
            )

    def _resolve_seg(self, sid: str) -> tuple[str | None, str | None]:
        # Seg dirs are flat (files directly in images_segm/, not per-session subdir)
        manual = _SEGM_DIR / f"{sid}_segm.nii.gz"
        auto = _AUTO_DIR / f"{sid}_automated_approx_segm.nii.gz"
        if (self.root / manual).exists():
            return str(manual), "manual"
        if (self.root / auto).exists():
            return str(auto), "automated"
        return None, None

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        df = pd.read_csv(self.root / "clinical_info.csv")
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise MissingColumnError(f"UPENN clinical_info.csv missing columns: {missing}")

        out: dict[str, ClinicalRecord] = {}
        for _, r in df.iterrows():
            cid = str(r["ID"]).strip()  # e.g. UPENN-GBM-00001_11
            # Strip the _NN suffix to get patient_id (same key as discover)
            parts = cid.rsplit("_", 1)
            pid = parts[0] if len(parts) == 2 and parts[1] in _SUFFIX_IDX else cid
            mraw = r["MGMT"]
            out[pid] = ClinicalRecord(
                patient_id=pid,
                age=to_float(r["Age_at_scan_years"]),
                os_days=to_float(r["Survival_from_surgery_days_UPDATED"]),
                os_event=event_from_status(r["Survival_Status"]),
                who_grade=None, who_grade_raw=None,
                eor=normalize_eor_binary(r["GTR_over90percent"]),
                eor_raw=raw_str(r["GTR_over90percent"]),
                idh_status=normalize_idh(r["IDH1"]),
                mgmt_methylation=normalize_mgmt(mraw), mgmt_raw=raw_str(mraw),
                codeletion_1p19q=None, clinical_row_found=True,
            )
        log.debug("UPENN: loaded %d clinical records", len(out))
        return out

    def clinical_key(self, session: RawSession) -> str:
        return session.patient_id

    def check_layout(self) -> list[LayoutIssue]:
        issues: list[LayoutIssue] = []

        if not self.root.exists():
            issues.append(LayoutIssue(
                severity="error", check="root directory",
                expected=str(self.root),
                fix=f"Create or mount the UPENN-GBM data directory at {self.root}",
            ))
            return issues

        struct_dir = self.root / _STRUCT
        if not struct_dir.exists():
            issues.append(LayoutIssue(
                severity="error", check="images_structural subdirectory",
                expected=str(struct_dir),
                fix=f"Extract the UPENN-GBM zip directly into {self.root} — "
                    f"imaging/NIfTI/images_structural/ must appear as a nested child",
            ))
            return issues

        clinical = self.root / "clinical_info.csv"
        if not clinical.exists():
            issues.append(LayoutIssue(
                severity="error", check="clinical_info.csv",
                expected=str(clinical),
                fix=f"Place clinical_info.csv directly inside {self.root}",
            ))

        subjects = list(struct_dir.glob("UPENN-GBM-*_11"))
        if not subjects:
            issues.append(LayoutIssue(
                severity="error", check="baseline session directories (UPENN-GBM-*_11)",
                expected=str(struct_dir / "UPENN-GBM-*_11"),
                fix=f"Extract patient session directories into {struct_dir}; "
                    f"baseline sessions end with _11",
            ))

        return issues
