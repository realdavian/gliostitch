# Package Architecture: gbm_manifest vs gbm_os

This repo contains two distinct Python packages. They are co-located because they share
a schema contract, but they serve different purposes, different users, and run at
different times.

---

## gbm_manifest — data engineering pipeline

**Purpose:** turn raw, heterogeneous dataset directories on disk into a single validated
`master_manifest.csv`.

**When it runs:** once offline, or whenever source data changes. Not imported during
training.

**What it owns:**
- Per-dataset adapters (BraTS2020, UCSF-PDGM, UPENN-GBM, RHUH-GBM) that handle
  each dataset's quirks: directory layout, file naming, clinical CSV parsing, ID
  mismatches, dual-encoded survival fields
- Deduplication across datasets (demographic → segmentation hash → image hash)
- Schema enforcement (`MANIFEST_COLUMNS`, column order, null hygiene)
- Cohort pre-filtering (`CohortSelector`): baseline-only, structural completeness,
  grade IV, GTR, OS availability, partition assignment (train / external_test)
- Pre-flight layout verification (`gbm-manifest verify-layout`)

**Output:** `output/master_manifest.csv` — one row per imaging session, columns
defined by `MANIFEST_COLUMNS` in `gbm_manifest/core/schema.py`.

**Entrypoint:** CLI — `gbm-manifest build` (or stage-by-stage subcommands).

---

## gbm_os — cohort access library

**Purpose:** provide a Python API for downstream ML code to query patients, filter
cohorts, split folds, and receive fully-resolved `SampleSpec` objects.

**When it runs:** imported at training time. Reads the CSV that `gbm_manifest`
produced.

**What it owns:**
- `Cohort.from_manifest(path, data_roots)` — loads and validates the CSV
- `Cohort.select(...)` / `CohortView.select(...)` — composable, chainable cohort
  filtering (baseline, modalities, arbitrary column filters, lambda predicates)
- `CohortView.split(k, seed)` → `FoldCollection` — stratified group k-fold CV
- `SampleSpec` — resolved metadata + absolute paths for one imaging session
- Backend adapters: `.to_monai()`, `.to_torchio()`, `.to_torch()`
- Transform helpers: age normalisation, intensity stacking, seg handling

**Entrypoint:** Python import — `from gbm_os import Cohort`.

---

## The contract between them

`gbm_os/manifest.py` imports `MANIFEST_COLUMNS` directly from
`gbm_manifest/core/schema.py` and validates every CSV on load:

```python
missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Manifest is missing required columns: {missing}. "
                     "Regenerate the CSV with a compatible version of gbm_manifest.")
```

This means:
- **Schema drift is caught at load time**, not silently mid-training.
- `MANIFEST_COLUMNS` in `schema.py` is the single source of truth. To add a column,
  edit it there — both packages pick up the change automatically.
- `gbm_os` does **not** call into `gbm_manifest` at runtime beyond this import.

---

## Analogy

| | gbm_manifest | gbm_os |
|---|---|---|
| Role | Data pipeline | Application library |
| Analogy | `alembic` (runs migrations) | `sqlalchemy` (queries the DB) |
| Runs | Offline, once | At training time, every run |
| User | Data engineer / pipeline operator | ML researcher / training script |
| Output | `master_manifest.csv` | PyTorch / MONAI / TorchIO Dataset |

---

## When to extend which package

| Task | Package |
|---|---|
| Add a new dataset | `gbm_manifest` — new adapter in `adapters/` |
| Add a new clinical column | `gbm_manifest` — extend `ManifestRow` + `MANIFEST_COLUMNS` in `core/schema.py` |
| Add a new cohort filter knob | `gbm_os` — extend `SelectionCriteria` in `selection.py` |
| Add a new ML backend (e.g. Keras) | `gbm_os` — new backend in `backends/` |
| Change OS class thresholds | `gbm_os` — `CohortConfig.os_thresholds` |
| Change deduplication logic | `gbm_manifest` — `stages/deduplicate.py` |
| Change cohort pre-filter defaults | `gbm_manifest` — `stages/cohort.py` |

---

## Pre-flight sequence

```
gbm-manifest verify-layout   # check dataset directories before starting
gbm-manifest build           # produce master_manifest.csv
# then in training code:
cohort = Cohort.from_manifest("output/master_manifest.csv", data_roots={...})
```
