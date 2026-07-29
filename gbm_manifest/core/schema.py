"""Manifest schema: the immutable-facts contract for the unified GBM-OS dataset."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional


class Dataset(str, Enum):
    BRATS2020 = "brats2020"
    RHUH_GBM = "rhuh_gbm"
    UPENN_GBM = "upenn_gbm"
    UCSF_PDGM = "ucsf_pdgm"


class Modality(str, Enum):
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"
    FLAIR = "flair"
    SEG = "seg"


class SegConvention(str, Enum):
    BRATS_LEGACY = "brats_legacy"   # {0,1,2,4}  ET = 4
    RHUH = "rhuh"                   # {0,1,2,3}  ET = 3 -> remap 3->4 at load


class AcquisitionContext(str, Enum):
    """Where a session sits relative to the first resection.

    session_index == 0 only means "earliest session on record", which is not
    the same claim. A dataset whose earliest scan is a post-operative one — a
    radiotherapy planning study, a longitudinal cohort whose first two studies
    straddle surgery in the same week — would be selected as a preoperative
    baseline with nothing to catch it. Recording the fact makes the eligibility
    rule enforceable rather than assumed.
    """
    PREOP = "preop"      # before any resection
    POSTOP = "postop"    # after the first resection, follow-up included
    UNKNOWN = "unknown"


class EORCategory(str, Enum):
    GTR = "GTR"
    NTR = "NTR"                     # near-total: >90% resected, thin residual rim
    STR = "STR"
    BIOPSY = "biopsy"
    NON_GTR = "non_GTR"             # UPENN: >90% resection flag answered "N"
    UNKNOWN = "unknown"


class VitalEvent(int, Enum):
    CENSORED = 0
    DECEASED = 1


class DedupMethod(str, Enum):
    NONE = "none"
    SEG_HASH = "seg_hash"
    T1CE_HASH = "t1ce_hash"
    DEMOGRAPHIC = "demographic"


class DedupConfidence(str, Enum):
    UNIQUE = "unique"
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"


@dataclass(slots=True)
class RawSession:
    dataset: Dataset
    patient_id: str
    session_index: int
    acquisition_context: AcquisitionContext
    t1_path: Optional[str]
    t1ce_path: Optional[str]
    t2_path: Optional[str]
    flair_path: Optional[str]
    seg_path: Optional[str]
    seg_convention: Optional[SegConvention]
    seg_source: Optional[str] = None

    @property
    def global_session_key(self) -> str:
        return f"{self.dataset.value}__{self.patient_id}__tp{self.session_index}"

    @property
    def present(self) -> dict[str, bool]:
        return {
            "t1": self.t1_path is not None,
            "t1ce": self.t1ce_path is not None,
            "t2": self.t2_path is not None,
            "flair": self.flair_path is not None,
            "seg": self.seg_path is not None,
        }


@dataclass(slots=True)
class ClinicalRecord:
    patient_id: str
    age: Optional[float] = None
    os_days: Optional[float] = None
    os_event: Optional[int] = None
    who_grade: Optional[int] = None
    who_grade_raw: Optional[str] = None
    eor: Optional[EORCategory] = None
    eor_raw: Optional[str] = None
    idh_status: Optional[str] = None
    mgmt_methylation: Optional[str] = None
    mgmt_raw: Optional[str] = None
    codeletion_1p19q: Optional[str] = None
    clinical_row_found: bool = True


@dataclass(slots=True)
class ManifestRow:
    # identity
    dataset: str
    patient_id: str
    session_index: int
    acquisition_context: str
    global_session_key: str
    # paths (relative to dataset root)
    t1_path: Optional[str]
    t1ce_path: Optional[str]
    t2_path: Optional[str]
    flair_path: Optional[str]
    seg_path: Optional[str]
    seg_convention: Optional[str]
    seg_source: Optional[str]
    # presence flags
    has_t1: bool
    has_t1ce: bool
    has_t2: bool
    has_flair: bool
    has_seg: bool
    # normalised clinical facts
    age: Optional[float]
    os_days: Optional[float]
    os_event: Optional[int]
    who_grade: Optional[int]
    who_grade_raw: Optional[str]
    eor: Optional[str]
    eor_raw: Optional[str]
    idh_status: Optional[str]
    mgmt_methylation: Optional[str]
    mgmt_raw: Optional[str]
    codeletion_1p19q: Optional[str]
    clinical_row_found: bool
    # dedup identity
    duplicate_group_id: Optional[str]
    dedup_method: str
    dedup_confidence: str


MANIFEST_COLUMNS: list[str] = [f.name for f in fields(ManifestRow)]

# Bump whenever the MEANING of a stored column changes, even if the column set
# does not. Cached stage artifacts written under an older version are discarded
# rather than reused (see infra/cache.py).
#
#   2 — UPENN clinical records keyed per session; BraTS grade retained for
#       subjects absent from survival_info; dedup hash evidence scoped to
#       same-pipeline pairs.
#   3 — acquisition_context recorded per session, so "preoperative" stops being
#       inferred from session_index == 0.
SCHEMA_VERSION: int = 3

DERIVED_NOT_STORED: tuple[str, ...] = (
    "os_class", "is_baseline", "is_longitudinal",
    "is_structural_complete", "partition", "fold",
)

SEG_CONVENTION_BY_DATASET: dict[Dataset, SegConvention] = {
    Dataset.BRATS2020: SegConvention.BRATS_LEGACY,
    Dataset.UPENN_GBM: SegConvention.BRATS_LEGACY,
    Dataset.UCSF_PDGM: SegConvention.BRATS_LEGACY,
    Dataset.RHUH_GBM: SegConvention.RHUH,
}

INTENSITY_PRENORMALISED: dict[Dataset, bool] = {
    Dataset.BRATS2020: False,
    Dataset.UPENN_GBM: False,
    Dataset.UCSF_PDGM: False,
    Dataset.RHUH_GBM: True,
}


# Derivations are deliberately NOT implemented here.
#
# This module defines the manifest CONTRACT — what a row is and what a column
# means. Turning facts into modelling quantities (os_class, is_baseline,
# is_longitudinal, is_structural_complete) is a selection-layer concern and
# lives in gbm_os.manifest, once.
#
# Second copies used to live here. They drifted: the scalar derive_os_class
# guarded on None but not NaN, so a session with no survival data was classified
# as a LONG survivor, while the vectorised version in gbm_os returned null. The
# tested implementation was not the one the pipeline called.
