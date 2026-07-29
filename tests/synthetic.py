"""Builds a miniature on-disk replica of all four dataset layouts.

The tree is deliberately small but encodes the exact structural hazards the real
datasets contain, so adapter defects are reproducible without /mnt/disk1:

  BraTS  — 4 subjects, only 2 in survival_info.csv, all 4 graded in name_mapping.csv
           (a subject absent from survival_info still has a KNOWN grade)
  UPENN  — patient 00001 has BOTH _11 and _21 rows in clinical_info.csv, with
           different age and a follow-up EOR of "Not Applicable"
  RHUH   — one longitudinal patient with sessions 0 and 1
  UCSF   — 4-digit directory id vs 3-digit CSV id
  LUMIERE— Patient-001 has week-000-1 (Pre-Op) and week-000-2 (Post-Op) in the
           SAME week, so ordering alone cannot tell them apart; Patient-002 has
           no preoperative study at all, so its earliest session is a follow-up
           sitting at session_index 0; Patient-003 has an unrated session

BraTS_Training_001 and UPENN-GBM-00002_11 are seeded as a demographic match with
byte-identical segmentations, so the dedup hash tiers have something to confirm.
"""
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

_SHAPE = (4, 4, 4)


def _write_nii(path: Path, seed: int, gzip: bool = True) -> None:
    """Write a tiny deterministic NIfTI volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 100, size=_SHAPE).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── BraTS-2020 ────────────────────────────────────────────────────────────── #

def _build_brats(root: Path) -> None:
    train = root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    subjects = ["BraTS20_Training_001", "BraTS20_Training_002",
                "BraTS20_Training_003", "BraTS20_Training_004"]

    for i, sid in enumerate(subjects):
        for tok in ("t1", "t1ce", "t2", "flair", "seg"):
            # seg of subject 001 is seeded to collide with UPENN-GBM-00002_11
            seed = 900 if (sid == subjects[0] and tok == "seg") else i * 10 + hash(tok) % 7
            _write_nii(train / sid / f"{sid}_{tok}.nii", seed)

    # Only 001 and 002 appear in survival_info — 003 and 004 are OS-less.
    _write_csv(
        train / "survival_info.csv",
        ["Brats20ID", "Age", "Survival_days", "Extent_of_Resection"],
        [
            {"Brats20ID": "BraTS20_Training_001", "Age": "60.0",
             "Survival_days": "400", "Extent_of_Resection": "GTR"},
            {"Brats20ID": "BraTS20_Training_002", "Age": "45.5",
             "Survival_days": "ALIVE (361 days later)", "Extent_of_Resection": "STR"},
        ],
    )

    # ...but name_mapping grades ALL FOUR. 003 is HGG, 004 is LGG.
    _write_csv(
        train / "name_mapping.csv",
        ["Grade", "BraTS_2020_subject_ID"],
        [
            {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_001"},
            {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_002"},
            {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_003"},
            {"Grade": "LGG", "BraTS_2020_subject_ID": "BraTS20_Training_004"},
        ],
    )


# ── UPENN-GBM ─────────────────────────────────────────────────────────────── #

def _build_upenn(root: Path) -> None:
    struct = root / "imaging" / "NIfTI" / "images_structural"
    segm = root / "imaging" / "NIfTI" / "images_segm"

    for i, sid in enumerate(["UPENN-GBM-00001_11", "UPENN-GBM-00001_21",
                             "UPENN-GBM-00002_11"]):
        for tok in ("T1", "T1GD", "T2", "FLAIR"):
            _write_nii(struct / sid / f"{sid}_{tok}.nii.gz", i * 20 + len(tok))
        # 00002_11's seg is seeded to match BraTS_Training_001's seg exactly.
        seed = 900 if sid == "UPENN-GBM-00002_11" else 500 + i
        _write_nii(segm / f"{sid}_segm.nii.gz", seed)

    # Patient 00001 has BOTH sessions. The _21 row carries a different age and
    # an EOR of "Not Applicable" — collapsing on a stripped id loses the _11 facts.
    _write_csv(
        root / "clinical_info.csv",
        ["ID", "Age_at_scan_years", "Survival_from_surgery_days_UPDATED",
         "Survival_Status", "GTR_over90percent", "IDH1", "MGMT"],
        [
            {"ID": "UPENN-GBM-00001_11", "Age_at_scan_years": "50.00",
             "Survival_from_surgery_days_UPDATED": "500", "Survival_Status": "Deceased",
             "GTR_over90percent": "Y", "IDH1": "NOS/NEC", "MGMT": "Methylated"},
            {"ID": "UPENN-GBM-00001_21", "Age_at_scan_years": "51.50",
             "Survival_from_surgery_days_UPDATED": "500", "Survival_Status": "Deceased",
             "GTR_over90percent": "Not Applicable", "IDH1": "NOS/NEC", "MGMT": "Methylated"},
            {"ID": "UPENN-GBM-00002_11", "Age_at_scan_years": "60.00",
             "Survival_from_surgery_days_UPDATED": "400", "Survival_Status": "Deceased",
             "GTR_over90percent": "N", "IDH1": "Wildtype", "MGMT": "Unmethylated"},
        ],
    )


# ── RHUH-GBM ──────────────────────────────────────────────────────────────── #

def _build_rhuh(root: Path) -> None:
    pid = "RHUH-0001"
    for idx in (0, 1):
        for tok in ("t1", "t1ce", "t2", "flair", "segmentations"):
            _write_nii(root / pid / str(idx) / f"{pid}_{idx}_{tok}.nii.gz",
                       700 + idx + len(tok))

    _write_csv(
        root / "manifests" / "clinical_info.csv",
        ["patient_id", "age", "os_days", "right_censored", "who_grade",
         "eor_category", "idh_status"],
        [{"patient_id": pid, "age": "55.0", "os_days": "300",
          "right_censored": "no", "who_grade": "4", "eor_category": "GTR",
          "idh_status": "wildtype"}],
    )


# ── UCSF-PDGM ─────────────────────────────────────────────────────────────── #

def _build_ucsf(root: Path) -> None:
    pid = "UCSF-PDGM-0004"          # 4-digit on disk
    d = root / "UCSF-PDGM-v5" / f"{pid}_nifti"
    for tok in ("T1", "T1c", "T2", "FLAIR", "tumor_segmentation"):
        _write_nii(d / f"{pid}_{tok}.nii.gz", 800 + len(tok))

    _write_csv(
        root / "UCSF-PDGM-metadata_v5.csv",
        ["ID", "Age at MRI", "OS", "1-dead 0-alive", "WHO CNS Grade", "EOR",
         "IDH", "MGMT status"],
        # Demographics deliberately match BraTS_Training_002 (age 45.5, 361 days,
        # alive) so the pair becomes a demographic candidate. Their segmentations
        # differ, because the two cohorts were annotated independently — that
        # mismatch must NOT be read as proof they are different people.
        [{"ID": "UCSF-PDGM-004",     # 3-digit in the CSV
          "Age at MRI": "45.5", "OS": "361", "1-dead 0-alive": "0",
          "WHO CNS Grade": "4", "EOR": "GTR", "IDH": "Wildtype",
          "MGMT status": "positive"}],
    )


# ── LUMIERE ───────────────────────────────────────────────────────────────── #

def _build_lumiere(root: Path) -> None:
    """Three patients covering the hazards ordering alone cannot resolve.

    Patient-001  week-000-1 Pre-Op, week-000-2 Post-Op — same week, so only the
                 expert rating separates them.
    Patient-002  week-000 Post-Op only — no preoperative study exists, so its
                 earliest session must NOT be treated as a baseline.
    Patient-003  week-000 Pre-Op, week-012 unrated — the unrated follow-up is
                 later than a known preoperative study, so it is postoperative.
    """
    sessions = {
        "Patient-001": ["week-000-1", "week-000-2"],
        "Patient-002": ["week-000"],
        "Patient-003": ["week-000", "week-012"],
    }
    for pid, weeks in sessions.items():
        for wk in weeks:
            for tok in ("t1_skull_strip", "ct1_skull_strip", "t2_skull_strip",
                        "flair_skull_strip", "seg_mask"):
                _write_nii(root / "train" / pid / wk / f"{tok}.nii",
                           900 + len(pid) + len(wk) + len(tok), gzip=False)

    _write_csv(
        root / "LUMIERE-Demographics_Pathology.csv",
        ["Patient", "Survival time (weeks)", "Sex", "Age at surgery (years)",
         "IDH (WT: wild type)", "IDH method", "MGMT qualitative", "MGMT quantitative"],
        # 'not methylated' and 'R132H mut' are the spellings that used to
        # normalise wrongly; 'na' survival exercises the unlabelled path.
        [{"Patient": "Patient-001", "Survival time (weeks)": "72", "Sex": "female",
          "Age at surgery (years)": "57", "IDH (WT: wild type)": "WT",
          "IDH method": "BES", "MGMT qualitative": "not methylated",
          "MGMT quantitative": "0.00%"},
         {"Patient": "Patient-002", "Survival time (weeks)": "40", "Sex": "male",
          "Age at surgery (years)": "61", "IDH (WT: wild type)": "R132H mut",
          "IDH method": "IHC", "MGMT qualitative": "methylated",
          "MGMT quantitative": "na"},
         {"Patient": "Patient-003", "Survival time (weeks)": "na", "Sex": "male",
          "Age at surgery (years)": "48", "IDH (WT: wild type)": "na",
          "IDH method": "na", "MGMT qualitative": "na",
          "MGMT quantitative": "na"}],
    )

    rating = ("Rating (according to RANO, PD: Progressive disease, SD: Stable "
              "disease, PR: Partial response, CR: Complete response, "
              "Pre-Op: Pre-Operative, Post-Op: Post-Operative)")
    reason = ("Rating rationale (CRET: complete resection of the enhancing tumor, "
              "PRET: partial resection of the enhancing tumor, T2-Progr.: "
              "T2-Progression, L: Lesion)")
    _write_csv(
        root / "LUMIERE-ExpertRating-v202211.csv",
        ["Patient", "Date", "LessThan3Months", "NonMeasurableLesions", rating, reason],
        [{"Patient": "Patient-001", "Date": "week-000-1", "LessThan3Months": "",
          "NonMeasurableLesions": "", rating: "Pre-Op", reason: ""},
         {"Patient": "Patient-001", "Date": "week-000-2", "LessThan3Months": "",
          "NonMeasurableLesions": "", rating: "Post-Op", reason: "CRET"},
         {"Patient": "Patient-002", "Date": "week-000", "LessThan3Months": "",
          "NonMeasurableLesions": "", rating: "Post-Op", reason: "PRET"},
         {"Patient": "Patient-003", "Date": "week-000", "LessThan3Months": "",
          "NonMeasurableLesions": "", rating: "Pre-Op", reason: ""}],
    )


def build_synthetic_datasets(base: Path) -> dict[str, Path]:
    """Create all five dataset trees under `base`; return dataset -> root."""
    roots = {
        "brats2020": base / "BraTS-2020",
        "rhuh_gbm": base / "RHUH-GBM",
        "upenn_gbm": base / "UPENN-GBM",
        "ucsf_pdgm": base / "UCSF-PDGM",
        "lumiere": base / "LUMIERE",
    }
    _build_brats(roots["brats2020"])
    _build_rhuh(roots["rhuh_gbm"])
    _build_upenn(roots["upenn_gbm"])
    _build_ucsf(roots["ucsf_pdgm"])
    _build_lumiere(roots["lumiere"])
    return roots
