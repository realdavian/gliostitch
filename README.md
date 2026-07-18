# GBM Survival Data Pipeline

A reproducible data-engineering pipeline and cohort-access package for multi-site GBM MRI survival research. Produces a single unified `master_manifest.csv` over four public datasets and exposes a composable Python API for cohort selection, cross-validation splitting, and framework-native dataset loading.

Built for the RO1 grant reproducibility requirement — every cohort selection is derived at load time from immutable manifest facts, not baked into the CSV.

---

## Datasets

| Dataset | Patients | Sessions | Modalities | Seg convention | Notes |
|---|---|---|---|---|---|
| BraTS-2020 | 369 | 369 | T1, T1ce, T2, FLAIR, Seg | BraTS `{0,1,2,4}` | 133 without OS data |
| RHUH-GBM | 40 | 120 | T1, T1ce, T2, FLAIR, Seg | RHUH `{0,1,2,3}` | Longitudinal; intensities pre-z-scored |
| UPENN-GBM | 630 | 671 | T1, T1ce, T2, FLAIR, Seg | BraTS `{0,1,2,4}` | External test set; ~36% incomplete |
| UCSF-PDGM | 501 | 501 | T1, T1ce, T2, FLAIR, Seg | BraTS `{0,1,2,4}` | Includes WHO grade 2/3 |

**Total:** 1,540 patients · 1,661 sessions · 31-column manifest

---

## Repository Structure

```
gbm-surv-data-pipeline/
├── gbm_manifest/          # Pipeline: audit → standardise → dedup → manifest
│   ├── adapters/          # Per-dataset adapters (one file per dataset)
│   ├── core/schema.py     # Enums, ManifestRow, MANIFEST_COLUMNS — shared contract
│   ├── stages/            # audit, standardize, deduplicate, cohort, manifest
│   └── infra/             # fs, hashing, parallel, io
├── gbm_os/                # Cohort-access package (load, select, split, train)
│   ├── cohort.py          # Cohort, CohortView, FoldCollection
│   ├── manifest.py        # load_manifest + schema validation
│   ├── summary.py         # cohort_summary() statistics report
│   ├── transforms/        # remap_seg, SegRemapd, foreground_zscore, AgeNormalizer
│   └── backends/          # torch, monai, torchio
├── notebooks/
│   └── 01_gbm_os_usage.ipynb
├── config/pipeline.yaml
├── tests/
└── output/                # master_manifest.csv (generated)
```

---

## Installation

**Requirements:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/realdavian/gbm-surv-data-pipeline
cd gbm-surv-data-pipeline
uv sync
source activate.sh
```

### Framework backends

MONAI and torchio require PyTorch. Install torch with your CUDA version **before** the extras — otherwise pip resolves the generic CPU build from PyPI.

```bash
# 1. Install PyTorch with your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install framework extras
pip install -e ".[all]"          # nibabel + monai + torchio
pip install -e ".[monai]"        # MONAI only
pip install -e ".[torchio]"      # torchio only
```

---

## Building the Manifest

Dataset roots are configured in `config/pipeline.yaml`. Raw data is assumed to already be on disk under `/mnt/disk1/datasets/`.

```bash
source activate.sh

# Full pipeline (audit → standardise → dedup → assemble)
manifest build --config config/pipeline.yaml

# Individual stages
manifest audit
manifest standardize
manifest dedup
manifest assemble
manifest validate --manifest output/master_manifest.csv

# Verbose logging
manifest build --config config/pipeline.yaml --verbose
```

Output: `output/master_manifest.csv` — 1,661 rows, 31 columns, byte-identical on re-run.

---

## Using the Cohort Package (`gbm_os`)

See **`notebooks/01_gbm_os_usage.ipynb`** for a fully executed end-to-end walkthrough.

### Load

```python
from gbm_os import Cohort, CohortConfig

config = CohortConfig(
    data_roots={
        "brats2020": "/mnt/disk1/datasets/BraTS-2020",
        "rhuh_gbm":  "/mnt/disk1/datasets/RHUH-GBM",
        "upenn_gbm": "/mnt/disk1/datasets/UPENN-GBM",
        "ucsf_pdgm": "/mnt/disk1/datasets/UCSF-PDGM",
    },
    external={"upenn_gbm"},
    priority=["brats2020", "rhuh_gbm", "ucsf_pdgm"],
)

cohort = Cohort.from_manifest("output/master_manifest.csv",
                               data_roots=config.data_roots,
                               config=config)
```

### Explore

```python
from gbm_os import cohort_summary

s = cohort_summary(cohort.select())
s.print()          # demographics, survival, modality completeness, distributions

s.clinical         # pd.DataFrame — patient-level stats per dataset
s.imaging          # pd.DataFrame — modality presence rates
s.distributions    # dict[str, pd.DataFrame] — EOR / IDH / MGMT / WHO grade

# Include NIfTI header scan (volume shape + voxel spacing per dataset × modality)
cohort_summary(cohort.select(datasets=["rhuh_gbm"]), scan_headers=True).spatial
```

### Select

```python
view = cohort.select(
    baseline_only=True,
    require_complete=True,
    filters={"eor": "GTR", "has_os": True},
    where=lambda r: r["dataset"] != "ucsf_pdgm" or r["who_grade"] == 4,
    resolve_duplicates="drop",
)
# → 488 sessions

train_view    = view.select(partition="train")     # 377 — BraTS + RHUH + UCSF
external_view = view.select(partition="external")  # 111 — UPENN
```

### Load volumes

```python
# PyTorch / nibabel (raw float32 arrays, no framework dependency beyond nibabel)
dataset = view.to_torch(include_seg=True)
item = dataset[0]
# item["image"]  → np.ndarray [C, H, W, D]  raw voxels
# item["seg"]    → np.ndarray [1, H, W, D]  RHUH labels already remapped to BraTS {0,1,2,4}

# MONAI (lazy path-based; add SegRemapd after LoadImaged)
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from gbm_os.transforms import SegRemapd

transforms = Compose([
    LoadImaged(keys=["t1ce", "seg"]),
    EnsureChannelFirstd(keys=["t1ce", "seg"]),
    SegRemapd(seg_key="seg"),   # remaps RHUH 3→4 using seg_convention from the data dict
])
monai_ds = view.to_monai(transforms=transforms, include_seg=True)

# torchio (seg eagerly loaded and remapped at Subject construction)
tio_ds = view.to_torchio(include_seg=True)
```

### Intensity normalisation

Normalisation is not applied automatically — the package surfaces the `intensity_prenormalised` flag per sample so your pipeline can branch correctly for RHUH (pre-z-scored) vs the other three datasets.

```python
from gbm_os.transforms import foreground_zscore

spec = next(iter(view))
image = dataset[0]["image"]   # [C, H, W, D]

normalised = np.stack([
    foreground_zscore(image[c], intensity_prenormalised=spec.intensity_prenormalised)
    for c in range(image.shape[0])
])
```

### Cross-validation splits

```python
folds = train_view.split(k=5, seed=42)   # stratified by os_class, grouped by patient_id

for i in range(folds.k):
    train_ds = folds.fold(i, split="train").to_torch()
    val_ds   = folds.fold(i, split="val").to_torch()
```

### Age normalisation (leakage-safe)

```python
from gbm_os.transforms import AgeNormalizer

norm = AgeNormalizer()
norm.fit(folds.fold(0, split="train"))           # stats from train only
train_ages = norm.transform(folds.fold(0, split="train"))
val_ages   = norm.transform(folds.fold(0, split="val"))   # uses train stats
```

---

## Manifest Design

The manifest stores **immutable facts only** — paths, presence flags, raw + normalised clinical values, dedup identity. Modelling choices (`os_class`, `is_baseline`, `partition`, `fold`) are derived at load time and never written to disk.

**Deduplication** runs across all baseline sessions (not a filtered cohort). Three tiers:
1. Demographic fingerprint (candidate generation)
2. Segmentation mask MD5 (confirmed match)
3. T1ce MD5 (tiebreaker)

Scope is limited to BraTS-anchored pairs — BraTS redistributes UPENN's BraTS-pipeline output, making seg hashes meaningful. UCSF/RHUH cross-pairs used independent intensity pipelines so hash matching is not applied.

---

## Development

```bash
source activate.sh
pytest                  # 150 tests, ~1.5 s
pytest -v               # verbose
pytest tests/test_p6_summary.py   # specific file
```

### Adding a new dataset

1. Add an adapter in `gbm_manifest/adapters/<dataset>.py` — implement `discover()`, `load_clinical()`, `clinical_key()`
2. Register it with `@register_adapter(Dataset.<NAME>)`
3. Import it in `gbm_manifest/adapters/__init__.py`
4. Add the dataset root to `config/pipeline.yaml`
5. Add the new `Dataset` enum value and its `seg_convention` / `intensity_prenormalised` entries to `gbm_manifest/core/schema.py` — `gbm_os` picks them up automatically
