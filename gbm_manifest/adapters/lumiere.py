"""LUMIERE adapter.

Longitudinal single-centre GBM cohort from Inselspital Bern (Suter et al.,
Sci Data 9:768, 2022). Patient directories hold week-offset session directories
(``week-000``, ``week-000-1``, ``week-044``), so a session's position in the
treatment course is encoded in its name rather than in an index.

Three things differ from the other adapters:

Preoperative status is read, not inferred. ``LUMIERE-ExpertRating`` labels every
rated study Pre-Op / Post-Op / RANO grade, and a patient's first two studies can
both sit in week 0 on either side of surgery — ``Patient-001`` has ``week-000-1``
(Pre-Op) and ``week-000-2`` (Post-Op). Ordering alone cannot separate those.

Survival is recorded in weeks and converted to days. There is no event
indicator, so ``os_event`` stays None: the cohort's follow-up window makes death
very likely for everyone with a recorded time, but inferring vital status is a
modelling decision and belongs in gbm_os. See ``analysis/lumiere_vital_status.py``.

Extent of resection is not a column. It is carried in the Post-Op study's RANO
rationale, where CRET (complete resection of the enhancing tumour) and PRET
(partial) are the recorded values. Every patient was resected — the publication's
inclusion criteria require it — so there is no biopsy-only arm.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from ..core.exceptions import MissingColumnError
from ..core.layout import LayoutIssue
from ..core.schema import (AcquisitionContext, ClinicalRecord, Dataset,
                           EORCategory, RawSession, SegConvention)
from .base import register_adapter
from .normalize import days_from_weeks, normalize_idh, normalize_mgmt, raw_str, to_float

log = logging.getLogger(__name__)

_IMAGES = Path("train")
_DEMOGRAPHICS = Path("LUMIERE-Demographics_Pathology.csv")
_RATINGS = Path("LUMIERE-ExpertRating-v202211.csv")

_WEEK_RE = re.compile(r"^week-(\d+)(?:-(\d+))?$")

# token in a session directory -> RawSession field. 'ct1' is the post-contrast
# T1; LUMIERE names it for the contrast agent rather than the sequence.
_MODS = {
    "t1": "t1_skull_strip.nii",
    "t1ce": "ct1_skull_strip.nii",
    "t2": "t2_skull_strip.nii",
    "flair": "flair_skull_strip.nii",
    "seg": "seg_mask.nii",
}

_REQUIRED = {"Patient", "Survival time (weeks)", "Age at surgery (years)"}

_PREOP_RATING = "Pre-Op"


def _week_sort_key(name: str, has_seg: bool) -> tuple[int, int, int]:
    """Order sessions within a patient: week, then intra-week suffix.

    The final term only breaks a tie between two studies in the same week with
    the same suffix, preferring the segmented one. Patient-060 carries two
    Pre-Op studies (week-000 and week-069) and is resolved by week alone.
    """
    m = _WEEK_RE.match(name)
    if not m:
        return (10**6, 0, 0)
    return (int(m.group(1)), int(m.group(2) or 0), 0 if has_seg else 1)


@register_adapter(Dataset.LUMIERE)
class LumiereAdapter:
    name = Dataset.LUMIERE

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._ratings: Optional[dict[tuple[str, str], str]] = None

    # ── expert ratings ────────────────────────────────────────────────── #

    def _rating_table(self) -> dict[tuple[str, str], str]:
        """(patient, session) -> RANO rating, cached."""
        if self._ratings is not None:
            return self._ratings

        table: dict[tuple[str, str], str] = {}
        path = self.root / _RATINGS
        if not path.exists():
            log.warning("LUMIERE: %s absent — every session becomes UNKNOWN", path)
            self._ratings = table
            return table

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            key = next((k for k in (reader.fieldnames or []) if k.startswith("Rating (")), None)
            if key is None:
                raise MissingColumnError(
                    f"{_RATINGS} has no 'Rating (...)' column; found {reader.fieldnames}"
                )
            for row in reader:
                table[(row["Patient"].strip(), row["Date"].strip())] = row[key].strip()

        self._ratings = table
        return table

    def _context(self, pid: str, session: str, preop_week: Optional[int]) -> AcquisitionContext:
        """Preoperative status for one session.

        An explicit Pre-Op rating wins. An explicit anything-else means the
        study follows surgery. An unrated study is postoperative only when the
        patient has a rated Pre-Op earlier in the course to place it against —
        otherwise there is nothing to justify the claim and it stays UNKNOWN.
        """
        rating = self._rating_table().get((pid, session))
        if rating == _PREOP_RATING:
            return AcquisitionContext.PREOP
        if rating:
            return AcquisitionContext.POSTOP
        if preop_week is None:
            return AcquisitionContext.UNKNOWN
        week = _week_sort_key(session, False)[0]
        return (AcquisitionContext.POSTOP if week > preop_week
                else AcquisitionContext.UNKNOWN)

    # ── discovery ─────────────────────────────────────────────────────── #

    def discover(self) -> Iterator[RawSession]:
        ratings = self._rating_table()
        images = self.root / _IMAGES

        for subj in sorted(images.glob("Patient-*")):
            if not subj.is_dir():
                continue
            pid = subj.name

            sessions = [d for d in subj.iterdir() if d.is_dir() and _WEEK_RE.match(d.name)]
            if not sessions:
                continue

            preop_weeks = [
                _week_sort_key(d.name, False)[0] for d in sessions
                if ratings.get((pid, d.name)) == _PREOP_RATING
            ]
            preop_week = min(preop_weeks) if preop_weeks else None

            ordered = sorted(
                sessions,
                key=lambda d: _week_sort_key(d.name, (d / _MODS["seg"]).exists()),
            )

            for idx, sess in enumerate(ordered):
                def rel(tok: str, _s: Path = sess) -> Optional[str]:
                    p = _s / _MODS[tok]
                    return str(p.relative_to(self.root)) if p.exists() else None

                yield RawSession(
                    dataset=self.name, patient_id=pid, session_index=idx,
                    acquisition_context=self._context(pid, sess.name, preop_week),
                    t1_path=rel("t1"), t1ce_path=rel("t1ce"),
                    t2_path=rel("t2"), flair_path=rel("flair"),
                    seg_path=rel("seg"),
                    seg_convention=SegConvention.LUMIERE, seg_source="deepbratumia",
                )

    # ── clinical ──────────────────────────────────────────────────────── #

    def _eor_by_patient(self) -> dict[str, tuple[EORCategory, Optional[str]]]:
        """Resection extent from the Post-Op study's RANO rationale.

        CRET is a complete resection of the enhancing tumour and PRET a partial
        one; the raw rationale is kept so the mapping stays auditable.
        """
        path = self.root / _RATINGS
        if not path.exists():
            return {}

        out: dict[str, tuple[EORCategory, Optional[str]]] = {}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            names = reader.fieldnames or []
            rating_key = next((k for k in names if k.startswith("Rating (")), None)
            reason_key = next((k for k in names if k.startswith("Rating rationale")), None)
            if rating_key is None or reason_key is None:
                return {}
            for row in reader:
                if not row[rating_key].strip().startswith("Post-Op"):
                    continue
                pid = row["Patient"].strip()
                if pid in out:
                    continue  # first post-operative study is the resection one
                reason = row[reason_key].strip()
                token = reason.upper()
                if token.startswith("CRET"):
                    out[pid] = (EORCategory.GTR, raw_str(reason))
                elif token.startswith("PRET"):
                    out[pid] = (EORCategory.STR, raw_str(reason))
                else:
                    out[pid] = (EORCategory.UNKNOWN, raw_str(reason) if reason else None)
        return out

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        path = self.root / _DEMOGRAPHICS
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))

        missing = _REQUIRED - set(rows[0] if rows else {})
        if missing:
            raise MissingColumnError(f"LUMIERE {_DEMOGRAPHICS.name} missing columns: {missing}")

        eor_map = self._eor_by_patient()

        out: dict[str, ClinicalRecord] = {}
        for r in rows:
            pid = r["Patient"].strip()
            eor, eor_raw = eor_map.get(pid, (EORCategory.UNKNOWN, None))
            idh_raw = r.get("IDH (WT: wild type)")
            mgmt_raw = r.get("MGMT qualitative")
            out[pid] = ClinicalRecord(
                patient_id=pid,
                age=to_float(r["Age at surgery (years)"]),
                os_days=days_from_weeks(r["Survival time (weeks)"]),
                # No vital-status column exists. See the module docstring.
                os_event=None,
                # GBM by construction under WHO 2016, with no grade column —
                # the same shape as UPENN, which the grade rule already admits.
                who_grade=None, who_grade_raw=None,
                eor=eor, eor_raw=eor_raw,
                idh_status=normalize_idh(idh_raw),
                mgmt_methylation=normalize_mgmt(mgmt_raw), mgmt_raw=raw_str(mgmt_raw),
                codeletion_1p19q=None, clinical_row_found=True,
            )
        log.debug("LUMIERE: loaded %d clinical records", len(out))
        return out

    def clinical_key(self, session: RawSession) -> str:
        return session.patient_id

    # ── layout ────────────────────────────────────────────────────────── #

    def check_layout(self) -> list[LayoutIssue]:
        issues: list[LayoutIssue] = []

        if not self.root.exists():
            issues.append(LayoutIssue(
                severity="error", check="root directory",
                expected=str(self.root),
                fix=f"Create or mount the LUMIERE data directory at {self.root}",
            ))
            return issues

        images = self.root / _IMAGES
        if not images.exists() or not any(images.glob("Patient-*")):
            issues.append(LayoutIssue(
                severity="error", check="patient directories (train/Patient-*)",
                expected=str(images / "Patient-*"),
                fix=f"Extract the LUMIERE imaging archive so patient directories "
                    f"land in {images}",
            ))

        for rel in (_DEMOGRAPHICS, _RATINGS):
            if not (self.root / rel).exists():
                issues.append(LayoutIssue(
                    severity="error", check=rel.name,
                    expected=str(self.root / rel),
                    fix=f"Place {rel.name} at the top level of {self.root}",
                ))

        return issues
