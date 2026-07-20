# Extending the System

Recipes for the changes people actually make. Each states where the change goes, what
else must move with it, and how to prove it worked.

Before starting, know which side of the boundary you are on:

> **`gbm_manifest` records what is true. `gbm_os` decides what to do about it.**

If your change is about **what the data is**, it belongs in `gbm_manifest`. If it is about
**which subset you want**, it belongs in `gbm_os`. Getting this wrong is the most common
way to break the design — see *Rules that must not bend* at the end.

---

## Add a new dataset

The adapter layer is fully decoupled: a fifth dataset is one new file plus one import.

### 1. Verify the layout on disk first

Do not write the adapter from a data dictionary. Every path and column name is an
assumption until you have run it against the real directory — a mistyped column silently
nulls an entire field rather than raising.

```bash
python - <<'PY'
import pandas as pd, glob
print(glob.glob("/mnt/disk1/datasets/NEW-DATASET/**/*.csv", recursive=True))
df = pd.read_csv("/mnt/disk1/datasets/NEW-DATASET/clinical.csv")
print(list(df.columns))
print(df.head())
PY
```

Record: where the clinical CSV lives, its exact column names, the patient-id format on
disk versus in the CSV, and the modality filename tokens.

### 2. Register the dataset

`gbm_manifest/core/schema.py`:

```python
class Dataset(str, Enum):
    ...
    NEW_DATASET = "new_dataset"

SEG_CONVENTION_BY_DATASET = {..., Dataset.NEW_DATASET: SegConvention.BRATS_LEGACY}
INTENSITY_PRENORMALISED   = {..., Dataset.NEW_DATASET: False}
```

`gbm_manifest/pipeline.py` — add to `_DATASET_MAP`.

### 3. Write the adapter

`gbm_manifest/adapters/new_dataset.py`. It satisfies the `DatasetAdapter` Protocol
structurally — no base class to inherit.

```python
@register_adapter(Dataset.NEW_DATASET)
class NewDatasetAdapter:
    name = Dataset.NEW_DATASET

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def discover(self) -> Iterator[RawSession]:
        """LEFT-SCAN: yield every session found on disk.

        Never skip a subject for missing modalities or missing clinical data —
        those become null paths and null fields, resolved downstream.
        """

    def load_clinical(self) -> dict[str, ClinicalRecord]:
        """Keyed by whatever clinical_key() returns."""

    def clinical_key(self, session: RawSession) -> str:
        """Bridge the on-disk id to the CSV id."""

    def check_layout(self) -> list[LayoutIssue]:
        """Pre-flight: report missing directories with an actionable fix."""
```

Then import it in `gbm_manifest/adapters/__init__.py` so it self-registers, and add the
root to `config/pipeline.yaml`.

### 4. The three things that go wrong

**Assert your columns exist.** `df.get(col)` returns `None` on a typo and nulls the field
for every row, invisibly.

```python
_REQUIRED = {"ID", "OS", "Age", "EOR"}
missing = _REQUIRED - set(df.columns)
if missing:
    raise MissingColumnError(f"clinical.csv missing columns: {missing}")
```

**Assert your clinical key is unique.** If the CSV has one row per *session* but you key
on *patient*, later rows silently overwrite earlier ones. This exact defect put a
follow-up scan's age and EOR onto 41 UPENN baseline sessions.

```python
if len(out) != len(df):
    raise ClinicalKeyCollision(
        f"{len(df)} rows collapsed to {len(out)} records — key is not unique"
    )
```

**Build clinical records over the union of your sources.** If grade comes from one file
and survival from another, iterating only the survival file discards a known grade for
every subject missing from it. Keep the fact; let `clinical_row_found` record the absence.

### 5. Reuse the normalisers

Never re-implement vocabulary mapping. `adapters/normalize.py` is the tested single source
for all four survival encodings, grade, EOR, IDH and MGMT.

```python
from .normalize import (parse_survival_days, event_from_status, normalize_grade,
                        normalize_eor_categorical, normalize_idh, normalize_mgmt)
```

Extend it only when a dataset produces a raw value none of them cover — and add the
assertion to `tests/manifest/test_normalize.py` in the same commit.

### 6. Add it to the synthetic fixture

`tests/synthetic.py`. This is what makes the adapter testable in CI without `/mnt/disk1`.
Encode your dataset's *structural hazard*, not just a happy path — the id mismatch, the
per-session clinical rows, whatever will bite later.

```python
def _build_new_dataset(root: Path) -> None:
    ...

def build_synthetic_datasets(base: Path) -> dict[str, Path]:
    roots = {..., "new_dataset": base / "NEW-DATASET"}
    _build_new_dataset(roots["new_dataset"])
```

Then extend `TestLeftScan::test_all_sessions_present` in
`tests/manifest/test_adapters.py` with the expected count.

### 7. If it shares an annotation pipeline with an existing cohort

Only then may hash comparison be treated as evidence. Declare it:

```yaml
dedup:
  same_pipeline_pairs:
    - [brats2020, upenn_gbm]
    - [new_dataset, brats2020]
```

Between independently annotated cohorts a hash mismatch proves nothing, so those pairs
stay demographic candidates rather than being ruled out.

### 8. Verify

```bash
gbm-manifest verify-layout          # layout check
pytest tests/manifest               # synthetic-tree contracts
gbm-manifest audit                  # real discovery count
```

Reconcile the discovery count against the directory listing before trusting anything
downstream.

---

## Add a manifest column

Only for **facts** — something true of the session regardless of what anyone models.
Modelling quantities are derived at load time and never stored; see the next recipe.

`gbm_manifest/core/schema.py`:

```python
@dataclass
class ManifestRow:
    ...
    ki67_index: Optional[float]     # add in the position you want it in the CSV

SCHEMA_VERSION: int = 3             # bump — see below
```

`MANIFEST_COLUMNS` derives from the dataclass fields automatically, and `gbm_os` validates
against it on load, so both packages pick the column up with no further edits.

Then:
1. Add the field to `ClinicalRecord` and populate it in each adapter that has it (leave
   it `None` elsewhere — a partial column is fine, that is what null means).
2. Map it through `stages/standardize.py`.
3. **Bump `SCHEMA_VERSION`.** Cached artifacts from the previous version are then
   discarded rather than silently reused.

> Bump the version whenever the **meaning** of a column changes, even if the column set
> does not. A recomputed value under an unchanged name is exactly the case that stale
> caches hide.

---

## Add a derived quantity

Anything computed *from* the manifest rather than read from disk — a risk band, a
composite score, a completeness variant.

`gbm_os/manifest.py`, inside `load_manifest`:

```python
df["is_elderly"] = df["age"] >= 65
```

It becomes available to every selection path immediately (`filters={"is_elderly": True}`),
and is never written back to the CSV.

**Exactly one implementation.** Do not add a scalar helper in `schema.py` alongside the
vectorised one here. A second copy of `derive_os_class` used to exist there and had
drifted: it classified a null survival time as *long*, while this one returned null. The
tested implementation was not the one the pipeline called.

Add boundary tests — including the missing-value case — to `tests/os/test_derivations.py`.

---

## Add or change a study

A study is the research decision layer. Declarative, versioned, no predicates, so it can
be printed, diffed and pasted into a methods section.

`gbm_os/studies.py`:

```python
MY_STUDY = StudyDefinition(
    name="gbm-os-elderly",
    version="1",
    description="Overall survival in patients aged 65+, held out on UPENN-GBM.",
    baseline_only=True,
    require_complete=True,
    filters={
        "eor":        "GTR",
        "who_grade":  [4, None],   # None admits cohorts that record no grade
        "has_os":     True,
        "is_elderly": True,
    },
    os_thresholds=(300, 450),
    external_datasets=frozenset({"upenn_gbm"}),
    priority=("brats2020", "ucsf_pdgm", "rhuh_gbm", "upenn_gbm"),
    resolve_duplicates="drop",
)

STUDIES = {..., MY_STUDY.name: MY_STUDY}
```

Select it in `config/pipeline.yaml` (`cohort.study`) or apply it directly:

```python
view = MY_STUDY.apply(cohort)
print(view.provenance())        # the flow, criterion by criterion
```

**Bump `version` when criteria change**, and leave the old definition in place if any
published result depends on it. A study is a claim about how a number was produced.

**What must not go in a study:** anything that is a modelling assumption rather than an
eligibility rule. Censoring is the standard example — the manifest records censored
outcomes and the study keeps them, because whether a model may use them is decided at
selection time:

```python
deceased = view.select(filters={"os_event": 1})    # complete-case, traced like any filter
```

Filtering it inside the study would make the alternative unreachable.

---

## Add a selection criterion

For a filter shape the built-ins cannot express. `gbm_os/selection.py`:

```python
@dataclass
class SelectionCriteria:
    ...
    min_age: Optional[float] = None
```

Then a step inside `apply_criteria`, using the `step()` helper so it is traced like every
other criterion:

```python
if criteria.min_age is not None:
    step("min_age", f"age >= {criteria.min_age}", current["age"] >= criteria.min_age)
```

Thread it through both `Cohort.select` and `CohortView.select` signatures.

Criteria are applied **sequentially**, not as one combined mask — that is what lets each
dropped row be attributed to the first criterion that removed it. Keep it that way; the
provenance arithmetic (`len(view) + len(exclusions) == n_input`) depends on it.

---

## Add a loading backend

`gbm_os/backends/<framework>_backend.py`. Take a `CohortView`, emit the shared contract:
`image [C,H,W,D]` with modalities stacked in the requested order, `modality_mask [C]`,
the clinical scalars, and `seg` when `include_seg`.

```python
class KerasDataset:
    def __init__(self, view, config, **kw): ...
```

Add a `.to_keras()` method on `CohortView`, an optional-dependency extra in
`pyproject.toml`, and **import the framework inside the function, never at module level** —
an AST test fails the build otherwise, because the core must stay installable on
pandas + numpy + pydantic alone.

Add the sample to `tests/os/test_backend_parity.py` so it is checked against the other
backends for the same fixed sample.

---

## Rules that must not bend

These are enforced by tests. If one fails, the design is being violated, not the test.

| Rule | Why | Enforced by |
|---|---|---|
| The manifest is a facts superset | A fact discarded at ingest is unrecoverable downstream | `test_adapters.py` |
| Never drop a subject during discovery | Missing data is a null field, not a missing row | `TestLeftScan` |
| Annotate duplicates, never delete | Keep/drop is a selection policy, not a manifest edit | `test_invariants.py` |
| Study choice cannot change the manifest | Otherwise the facts depend on the question asked | `TestStudyChoiceDoesNotReachTheManifest` |
| No research decisions in `gbm_manifest` | Eligibility belongs to the study, not the pipeline | `TestPhase1Delegates` |
| No top-level `gbm_os` import in the pipeline | Keeps the pipeline independent of the selection layer | `test_packaging.py` |
| No heavy imports in the `gbm_os` core | The light install must keep working | `test_packaging.py` |
| Same inputs produce a byte-identical manifest | Reproducibility | `TestReproducibility` |
| Every derived quantity has one implementation | Two copies drift, and the tested one may not be the live one | `test_derivations.py` |

---

## Before you commit

```bash
pytest                       # 237 tests, ~4 s
pytest tests/manifest        # pipeline only — no manifest needed
gbm-manifest build           # end-to-end against real data
```

Then confirm what you actually committed:

```bash
git show --stat HEAD
```

A clean `git commit` is not evidence the files went in — `.gitignore` can silently drop
them, and has.
