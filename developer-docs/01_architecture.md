# Package Architecture: gbm_manifest vs gbm_os

This repo contains two Python packages. They are co-located because they share a schema
contract, but they serve different purposes, different users, and run at different times.

The split is not organisational. It encodes one rule:

> **`gbm_manifest` records what is true. `gbm_os` decides what to do about it.**

Every design question below resolves against that sentence.

---

## gbm_manifest — data engineering pipeline

**Purpose:** turn raw, heterogeneous dataset directories on disk into a single validated
`master_manifest.csv`.

**When it runs:** once offline, or whenever source data changes. Not imported during
training.

**What it owns:**
- Per-dataset adapters (BraTS2020, UCSF-PDGM, UPENN-GBM, RHUH-GBM) that handle each
  dataset's quirks: directory layout, file naming, clinical CSV parsing, ID mismatches,
  dual-encoded survival fields
- Deduplication across datasets (demographic → segmentation hash → image hash), with
  hash evidence scoped to cohorts that share an annotation pipeline
- Schema enforcement (`MANIFEST_COLUMNS`, column order, null hygiene)
- Stage caching and invalidation (`infra/cache.py`)
- Pre-flight layout verification (`gbm-manifest verify-layout`)

**What it deliberately does NOT own:** eligibility criteria, survival thresholds, the
held-out cohort, duplicate priority, or any other research decision. See
*The cohort stage* below.

**Output:** `output/master_manifest.csv` — one row per imaging session (1661 rows,
31 columns), defined by `MANIFEST_COLUMNS` in `gbm_manifest/core/schema.py`.

**Entrypoint:** CLI — `gbm-manifest build`, or stage-by-stage subcommands.

---

## gbm_os — cohort access library

**Purpose:** provide a Python API for downstream ML code to query patients, filter
cohorts, split folds, and receive fully-resolved `SampleSpec` objects.

**When it runs:** imported at training time. Reads the CSV that `gbm_manifest` produced;
never re-parses the raw datasets.

**What it owns:**
- `Cohort.from_manifest(path, data_roots)` — loads and validates the CSV
- **Runtime derivations** — `os_class`, `is_baseline`, `is_longitudinal`,
  `is_structural_complete`, `has_os`. Computed on load, never written back. These have
  exactly one implementation, in `manifest.py`.
- **Study definitions** (`studies.py`) — the research decisions: eligibility criteria,
  survival thresholds, held-out cohort, duplicate priority
- `Cohort.select(...)` / `CohortView.select(...)` — composable, chainable filtering
- **Provenance** (`provenance.py`) — every criterion applied and every row it dropped
- `CohortView.split(k, seed)` → `FoldCollection` — stratified group k-fold CV
- `SampleSpec` — resolved metadata + absolute paths for one imaging session
- Backend adapters: `.to_monai()`, `.to_torchio()`, `.to_torch()`
- Transform helpers: seg remap, intensity normalisation, leakage-safe age normalisation

**Entrypoint:** Python import — `from gbm_os import Cohort`.

---

## The contract between them

### Direction

`gbm_os` may import the manifest **contract** and nothing else:

```python
from gbm_manifest.core.schema import MANIFEST_COLUMNS, INTENSITY_PRENORMALISED
```

It must never import the adapters, the stages, or anything that touches raw data. Every
CSV is validated on load, so schema drift surfaces immediately rather than silently
mid-training.

`MANIFEST_COLUMNS` is the single source of truth. To add a column, edit `ManifestRow` in
`schema.py`; both packages pick it up.

> Spec 02 §2 originally forbade this import outright. Taken literally that forces a second
> copy of 31 column names inside `gbm_os` — precisely the drift the contract exists to
> prevent. The spec was amended, not the code.

### The cohort stage — the one reverse dependency

`gbm_manifest/stages/cohort.py` imports `gbm_os`, which is the only place in the pipeline
that does. It is a **deferred** import (inside the function), so the pipeline core stays
independent of the selection layer.

This is deliberate. Spec 01 §8 places cohort selection in the loader, not the manifest,
but also permits the pipeline to *emit* a derived `cohort/selected.csv` so the cohort
table is regenerable from one command. The stage therefore owns no criteria — it loads a
named study, applies it, and writes the result:

```python
study = get_study(self.cfg.study)      # "gbm-os" from pipeline.yaml
view = study.load(manifest_path, self.data_roots)
write_csv(view.to_frame(), sel_path)
write_csv(view.exclusions(), exc_path)
```

Two tests enforce this: one asserts no eligibility vocabulary (`GTR`, `who_grade`,
`session_index`) appears in the stage source; another asserts no module in `gbm_manifest`
imports `gbm_os` at top level.

### Study choice cannot reach the manifest

Building under a different study produces a **byte-identical** `master_manifest.csv`; only
`cohort/selected.csv` changes. This is what makes the manifest reusable — if a study could
alter it, the facts would depend on the question being asked. Locked in by
`tests/manifest/test_invariants.py::TestStudyChoiceDoesNotReachTheManifest`.

---

## Studies — where research decisions live

A **study** is a declarative, versioned set of eligibility criteria and cohort policies.
No predicates, so it can be printed, diffed, and pasted into a methods section.

```bash
gbm-manifest studies            # list
gbm-manifest studies gbm-os     # full definition
```

| Study | Criteria | Cohort |
|---|---|---|
| `gbm-os` **(canonical)** | baseline ∩ complete ∩ GTR ∩ grade-IV ∩ has-OS | 502 — 377 train / 125 external |
| `gbm-os-m6` | reconciliation only; omits has-OS | 508 — 377 train / 131 external |

`gbm-os-m6` exists solely to reproduce the 131 figure that spec 01 M6 originally recorded,
before the `has-OS` criterion was reconciled between the two specs. It is not the study
cohort; `studies.CANONICAL` names the one that is.

**Censoring is not filtered by any study.** The manifest records censored outcomes, and
whether a model may use them is a modelling decision made downstream:

```python
view = GBM_OS_STUDY.apply(cohort)          # 502, censoring intact
deceased = view.select(filters={"os_event": 1})   # 390, complete-case
```

Baking it into the study would make the alternative unreachable. See
[Reconciliations](03_design_decisions.md#reconciliations) for the single BraTS case this
turns on.

---

## Provenance

Every `CohortView` carries the criteria that produced it, and every row they dropped:

```python
view.provenance()    # SelectionTrace — step-by-step flow
view.exclusions()    # DataFrame, one row per drop, tagged with the reason
```

```
input                      1661
baseline_only              1661 →   1521  (−140)
require_complete           1521 →   1126  (−395)
filter:eor                 1126 →    526  (−600)
filter:who_grade            526 →    509  (−17)
filter:has_os               509 →    503  (−6)
resolve_duplicates          503 →    502  (−1)
selected                    502
```

Criteria are applied sequentially rather than as one combined mask, so each dropped row is
attributed to the **first** criterion that removed it. The reasons therefore partition the
drops exactly: `len(view) + len(view.exclusions()) == n_input`. That is what makes the
cohort table defensible to a reviewer — and what the pipeline's `exclusions.csv` is a thin
serialisation of.

---

## Caching

Each cached artifact carries a `.meta.json` sidecar recording a fingerprint of everything
that determined its contents — config subtree, `SCHEMA_VERSION`, and the upstream stage's
fingerprint. Fingerprints chain, so invalidating an early stage cascades forward.

A missing, unreadable, or mismatched sidecar counts as a cache miss. Testing only for file
existence — the previous behaviour — let an artifact written by deleted code be served
indefinitely as current.

**Bump `SCHEMA_VERSION` in `core/schema.py` whenever the meaning of a stored column
changes**, even if the column set does not. Artifacts from an older version are then
discarded rather than reused.

---

## Analogy

| | gbm_manifest | gbm_os |
|---|---|---|
| Role | Data pipeline | Application library |
| Analogy | `alembic` (runs migrations) | `sqlalchemy` (queries the DB) |
| Runs | Offline, once | At training time, every run |
| User | Data engineer / pipeline operator | ML researcher / training script |
| Output | `master_manifest.csv` | PyTorch / MONAI / TorchIO Dataset |
| Records | Facts | Decisions |

---

## When to extend which package

| Task | Package |
|---|---|
| Add a new dataset | `gbm_manifest` — new adapter in `adapters/` |
| Add a new clinical column | `gbm_manifest` — extend `ManifestRow` + `MANIFEST_COLUMNS`, bump `SCHEMA_VERSION` |
| Change deduplication logic | `gbm_manifest` — `stages/deduplicate.py` |
| Declare two cohorts share an annotation pipeline | `gbm_manifest` — `dedup.same_pipeline_pairs` in `pipeline.yaml` |
| Change eligibility criteria | `gbm_os` — edit or add a `StudyDefinition` in `studies.py` |
| Change OS class thresholds | `gbm_os` — `StudyDefinition.os_thresholds` |
| Change the held-out cohort or duplicate priority | `gbm_os` — `StudyDefinition` |
| Add a new cohort filter knob | `gbm_os` — extend `SelectionCriteria` in `selection.py` |
| Add a new ML backend (e.g. Keras) | `gbm_os` — new backend in `backends/` |

If a change is about **what the data is**, it belongs in `gbm_manifest`. If it is about
**which subset you want**, it belongs in `gbm_os`.

---

## Installation

The core stays light so a consumer who only reads the manifest is not made to install the
pipeline:

```bash
pip install -e .              # pandas, numpy, pydantic — selection + path resolution
pip install -e ".[pipeline]"  # + typer, pyyaml, pyarrow, nibabel — to run the pipeline
pip install -e ".[load]"      # + nibabel — to read volumes
pip install -e ".[monai]"     # or [torchio] / [torch]
pip install -e ".[all]"       # everything
```

On a CUDA host install `torch` first with the matching build, then the backend extras
resolve against it.

An AST test (`tests/test_packaging.py`) fails on a top-level import of `typer`, `pyarrow`,
`yaml`, `nibabel`, `monai`, `torchio` or `torch` in any core `gbm_os` module. Defer such
imports inside the function that needs them.

---

## Tests

Grouped by the package under test, so a failure names its layer:

```
tests/
  manifest/   normalize · adapters · invariants · cache · schema
  os/         derivations · selection · where_predicate · studies · provenance
              duplicate_policy · path_resolution · cv_splits · cohort_summary · transforms
  test_packaging.py    spans both
  conftest.py          fixtures
  synthetic.py         miniature four-dataset tree
```

`tests/manifest` runs against a **synthetic dataset tree** built on the fly — the real
adapters and the real pipeline, no `/mnt/disk1` required, ~0.6s. The fixture encodes the
structural hazards the production data contains (UPENN `_11`/`_21` clinical rows, BraTS
subjects graded but absent from `survival_info`, a shared segmentation across cohorts), so
join defects are catchable in CI.

`tests/os` reads the production manifest and skips with a clear reason when it has not been
generated, so a fresh clone still gets a green suite.

```bash
pytest                  # everything
pytest tests/manifest   # pipeline only, no manifest needed
pytest tests/os         # selection layer
```

---

## Pre-flight sequence

```bash
gbm-manifest verify-layout   # check dataset directories before starting
gbm-manifest build           # produce master_manifest.csv + cohort/
```

```python
# then in training code:
from gbm_os import Cohort
from gbm_os.studies import GBM_OS_STUDY

cohort = Cohort.from_manifest("output/master_manifest.csv", data_roots={...})
view = GBM_OS_STUDY.apply(cohort)
```
