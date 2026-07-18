"""cohort_summary: per-dataset statistics report for a CohortView."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gbm_os.cohort import CohortView

logger = logging.getLogger(__name__)

_MODALITIES = ["t1", "t1ce", "t2", "flair", "seg"]
_CONTINUOUS = ["age", "os_days"]
_CATEGORICAL = ["eor", "idh_status", "mgmt_methylation", "who_grade"]


# ── public API ────────────────────────────────────────────────────────────── #

@dataclass
class CohortSummary:
    """Statistics report returned by cohort_summary().

    Attributes
    ----------
    clinical:       Continuous demographics & survival per dataset (patient-level).
    imaging:        Modality presence rates per dataset (session-level).
    distributions:  Dict of categorical variable → count/% table per dataset.
    spatial:        Volume shape & voxel spacing per dataset × modality.
                    None unless cohort_summary(..., scan_headers=True).
    """
    clinical: pd.DataFrame
    imaging: pd.DataFrame
    distributions: dict[str, pd.DataFrame]
    spatial: Optional[pd.DataFrame]

    def __str__(self) -> str:
        return _format(self)

    def print(self) -> None:
        print(str(self))


def cohort_summary(view: "CohortView", scan_headers: bool = False) -> CohortSummary:
    """Compute a statistics report for a CohortView.

    Parameters
    ----------
    view:           Any CohortView — the result of Cohort.select(), split().fold(), etc.
    scan_headers:   Read NIfTI headers to report volume shape and voxel spacing.
                    Requires nibabel. Runs one header read per session × modality —
                    use on a filtered view, not the full manifest.
    """
    df = view.to_frame()
    return CohortSummary(
        clinical=_clinical_stats(df),
        imaging=_imaging_stats(df),
        distributions=_categorical_distributions(df),
        spatial=_spatial_stats(view) if scan_headers else None,
    )


# ── helpers ───────────────────────────────────────────────────────────────── #

def _patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, patient_id): baseline session (index=0) or lowest available."""
    return (
        df.sort_values("session_index")
        .drop_duplicates(subset=["dataset", "patient_id"], keep="first")
        .reset_index(drop=True)
    )


def _clinical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous clinical stats per dataset at patient level, with a TOTAL row."""
    pat = _patient_level(df)
    n_longitudinal_by_dataset = (
        df.groupby(["dataset", "patient_id"])["session_index"]
        .count()
        .gt(1)
        .groupby(level="dataset")
        .sum()
        .astype(int)
    )

    rows = []
    for dataset, g in pat.groupby("dataset", sort=True):
        row: dict = {
            "dataset": dataset,
            "n_patients": len(g),
            "n_sessions": int(df["dataset"].eq(dataset).sum()),
            "n_longitudinal": int(n_longitudinal_by_dataset.get(dataset, 0)),
        }
        for col in _CONTINUOUS:
            s = pd.to_numeric(g[col], errors="coerce")
            valid = s.dropna()
            row[f"{col}_mean"] = round(float(valid.mean()), 1) if len(valid) else np.nan
            row[f"{col}_median"] = round(float(valid.median()), 1) if len(valid) else np.nan
            row[f"{col}_std"] = round(float(valid.std()), 1) if len(valid) else np.nan
            row[f"{col}_pct_missing"] = round(s.isna().mean() * 100, 1)
        ev = pd.to_numeric(g["os_event"], errors="coerce")
        row["event_rate_pct"] = round(float(ev.mean()) * 100, 1) if ev.notna().any() else np.nan
        rows.append(row)

    # TOTAL
    total: dict = {
        "dataset": "TOTAL",
        "n_patients": pat["patient_id"].nunique(),
        "n_sessions": len(df),
        "n_longitudinal": int(n_longitudinal_by_dataset.sum()),
    }
    for col in _CONTINUOUS:
        s = pd.to_numeric(pat[col], errors="coerce")
        valid = s.dropna()
        total[f"{col}_mean"] = round(float(valid.mean()), 1) if len(valid) else np.nan
        total[f"{col}_median"] = round(float(valid.median()), 1) if len(valid) else np.nan
        total[f"{col}_std"] = round(float(valid.std()), 1) if len(valid) else np.nan
        total[f"{col}_pct_missing"] = round(s.isna().mean() * 100, 1)
    ev_all = pd.to_numeric(pat["os_event"], errors="coerce")
    total["event_rate_pct"] = round(float(ev_all.mean()) * 100, 1) if ev_all.notna().any() else np.nan
    rows.append(total)

    return pd.DataFrame(rows).set_index("dataset")


def _imaging_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Modality presence rates per dataset at session level, with a TOTAL row."""
    rows = []
    for dataset, g in df.groupby("dataset", sort=True):
        row: dict = {"dataset": dataset, "n_sessions": len(g)}
        for mod in _MODALITIES:
            col = f"has_{mod}"
            row[f"pct_{mod}"] = (
                round(g[col].astype(bool).mean() * 100, 1) if col in g.columns else np.nan
            )
        row["pct_complete"] = (
            round(g["is_structural_complete"].astype(bool).mean() * 100, 1)
            if "is_structural_complete" in g.columns else np.nan
        )
        rows.append(row)

    total: dict = {"dataset": "TOTAL", "n_sessions": len(df)}
    for mod in _MODALITIES:
        col = f"has_{mod}"
        total[f"pct_{mod}"] = (
            round(df[col].astype(bool).mean() * 100, 1) if col in df.columns else np.nan
        )
    total["pct_complete"] = (
        round(df["is_structural_complete"].astype(bool).mean() * 100, 1)
        if "is_structural_complete" in df.columns else np.nan
    )
    rows.append(total)

    return pd.DataFrame(rows).set_index("dataset")


def _categorical_distributions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Value-count and percentage table per categorical variable, split by dataset.

    To extend: add column names to _CATEGORICAL at the top of this file.
    Each returned DataFrame has one row per observed value and one column per dataset
    (plus TOTAL), with values as "N (P%)" strings.
    """
    pat = _patient_level(df)
    datasets = sorted(pat["dataset"].unique())
    groups = {d: pat[pat["dataset"] == d] for d in datasets}
    groups["TOTAL"] = pat

    result: dict[str, pd.DataFrame] = {}
    for col in _CATEGORICAL:
        if col not in pat.columns:
            continue
        frames: dict[str, pd.Series] = {}
        for label, g in groups.items():
            counts = g[col].fillna("missing").value_counts()
            pcts = (counts / len(g) * 100).round(1)
            frames[label] = counts.astype(str) + " (" + pcts.astype(str) + "%)"
        result[col] = pd.concat(frames, axis=1).fillna("0 (0.0%)")

    return result


def _spatial_stats(view: "CohortView") -> pd.DataFrame:
    """Read NIfTI headers to collect volume shape and voxel spacing.

    Returns a DataFrame indexed by (dataset, modality) with mean/median/std
    of H, W, D dimensions and x/y/z voxel spacing in mm.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required for scan_headers=True. "
            "Install with: pip install gbm-os[load]"
        ) from e

    records: list[dict] = []
    n_read = 0
    for spec in view:
        for mod, path in spec.paths.items():
            if path is None or not spec.present.get(mod, False):
                continue
            try:
                img = nib.load(path)
                shape = img.shape
                zooms = img.header.get_zooms()
                records.append({
                    "dataset": spec.dataset,
                    "modality": mod,
                    "H": int(shape[0]),
                    "W": int(shape[1]),
                    "D": int(shape[2]) if len(shape) > 2 else None,
                    "spacing_x_mm": round(float(zooms[0]), 3) if len(zooms) > 0 else None,
                    "spacing_y_mm": round(float(zooms[1]), 3) if len(zooms) > 1 else None,
                    "spacing_z_mm": round(float(zooms[2]), 3) if len(zooms) > 2 else None,
                })
                n_read += 1
            except Exception as exc:
                logger.warning(
                    "Header read failed — %s / %s: %s",
                    spec.global_session_key, mod, exc,
                )
    logger.info("scan_headers: read %d NIfTI headers", n_read)

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    stat_cols = ["H", "W", "D", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]
    agg = (
        raw.groupby(["dataset", "modality"])[stat_cols]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    counts = raw.groupby(["dataset", "modality"]).size().rename("n_scanned")
    return agg.join(counts)


# ── formatting ────────────────────────────────────────────────────────────── #

def _format(s: CohortSummary) -> str:
    n_sessions = s.clinical.loc["TOTAL", "n_sessions"] if "TOTAL" in s.clinical.index else "?"
    n_patients = s.clinical.loc["TOTAL", "n_patients"] if "TOTAL" in s.clinical.index else "?"
    n_datasets = len(s.clinical.index) - 1  # exclude TOTAL

    sep = "─" * 72
    header = (
        f"{'═' * 72}\n"
        f"  COHORT SUMMARY   {n_patients} patients · {n_sessions} sessions · {n_datasets} datasets\n"
        f"{'═' * 72}"
    )

    sections = [header]

    # --- clinical ---
    sections.append(f"\n{sep}\n  Demographics & survival  (patient-level)\n{sep}")
    clin_display = s.clinical.rename(columns={
        "n_patients": "n_pat", "n_sessions": "n_ses", "n_longitudinal": "n_long",
        "age_mean": "age_μ", "age_median": "age_med", "age_std": "age_σ",
        "age_pct_missing": "age_miss%",
        "os_days_mean": "os_μ", "os_days_median": "os_med", "os_days_std": "os_σ",
        "os_days_pct_missing": "os_miss%", "event_rate_pct": "event%",
    })
    sections.append(clin_display.to_string())

    # --- imaging ---
    sections.append(f"\n{sep}\n  Modality completeness  (session-level)\n{sep}")
    img_display = s.imaging.rename(columns={
        "n_sessions": "n_ses",
        "pct_t1": "T1%", "pct_t1ce": "T1ce%", "pct_t2": "T2%",
        "pct_flair": "FLAIR%", "pct_seg": "Seg%", "pct_complete": "4-mod%",
    })
    sections.append(img_display.to_string())

    # --- categorical ---
    for col, dist in s.distributions.items():
        sections.append(f"\n{sep}\n  {col}\n{sep}")
        sections.append(dist.to_string())

    # --- spatial ---
    if s.spatial is not None and not s.spatial.empty:
        sections.append(f"\n{sep}\n  Volume shape & voxel spacing  (from NIfTI headers)\n{sep}")
        sections.append(s.spatial.to_string())

    return "\n".join(sections) + "\n"
