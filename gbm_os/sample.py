from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SampleSpec:
    """Resolved metadata + paths for a single imaging session.

    Paths are absolute (data_root / relative_path); None where the modality is absent.
    `present` mirrors the has_* flags from the manifest.
    """

    # imaging
    paths: dict[str, Optional[str]]   # modality -> absolute path or None
    present: dict[str, bool]          # modality -> presence flag

    # identity
    dataset: str
    patient_id: str
    session_index: int
    global_session_key: str

    # derived
    is_baseline: bool
    is_longitudinal: bool
    is_structural_complete: bool
    intensity_prenormalised: bool
    seg_convention: Optional[str]

    # clinical facts
    age: Optional[float]
    os_days: Optional[float]
    os_event: Optional[int]
    os_class: Optional[int]
    who_grade: Optional[int]
    eor: Optional[str]
    idh_status: Optional[str]
    mgmt_methylation: Optional[str]

    # dedup identity
    duplicate_group_id: Optional[str]
    dedup_confidence: str
    is_duplicate_flagged: bool   # True when flagged as dup but NOT dropped
