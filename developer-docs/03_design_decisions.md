# Design Decisions

Why the system is shaped the way it is. Each entry states the decision, what it rules out,
and what breaks if it is reversed — so a future change can be made deliberately rather
than by accident.

Several were arrived at the hard way. Where a decision exists because something went
wrong, that is recorded, because the failure is the argument.

---

## The manifest is a facts superset, not a cohort

`master_manifest.csv` holds one row per imaging session — 1661 rows across four datasets —
and records only what is true of that session: identity, relative paths, presence flags,
raw and normalised clinical values, duplicate identity.

It does **not** store `os_class`, `is_baseline`, `is_longitudinal`,
`is_structural_complete`, `partition` or `fold`. Those are modelling choices, derived at
load time by `gbm_os.manifest`.

**Why.** A fact discarded at ingest is unrecoverable; a derivation recomputed at load costs
milliseconds. Storing a modelling choice also fixes it for every future consumer — the
moment `os_class` is a column, every study inherits one banding.

**What breaks if reversed.** The manifest stops being reusable across studies. A second
research question would need a second manifest, and the two would drift.

**Consequence you will notice.** Rows exist that no study will ever select — 133 BraTS
subjects with no survival data, 430 incomplete UPENN sessions. That is correct. They are
facts about the data, and a completeness analysis needs them.

---

## Discovery is a left-scan: nothing is dropped

`discover()` yields every session found on disk. A missing modality becomes a null path; a
missing clinical row becomes null fields plus `clinical_row_found=False`. No subject is
ever skipped for incompleteness.

**Why.** Filtering during discovery makes the filter invisible. A count that comes out
wrong three stages later cannot be traced back to a subject that was never yielded.

**What breaks if reversed.** Exclusion accounting stops summing. The pipeline currently
guarantees `selected + excluded == 1661`; that identity is what makes the cohort table
defensible.

---

## Duplicates are annotated, never deleted

Cross-dataset duplicates get `duplicate_group_id`, `dedup_method` and `dedup_confidence`.
Both members stay in the manifest. Which one to keep is a selection policy
(`resolve_duplicates="drop"|"flag"|"keep"`), applied downstream.

**Why.** The right member depends on the question. A study holding out UPENN wants the
BraTS twin; a study of annotation variability wants both.

---

## A hash is only evidence between cohorts that share a pipeline

Deduplication runs three tiers: demographic fingerprint (candidate generation), then
segmentation MD5, then T1ce MD5. The hash tiers run **only** on dataset pairs declared in
`dedup.same_pipeline_pairs` — by default BraTS↔UPENN, the sole literal-reuse path, since
BraTS-2020 redistributes UPENN's BraTS-pipeline output.

Between independently annotated cohorts, a segmentation mismatch means the two institutions
drew the tumour differently. It says nothing about whether it is the same person.

**Why this is explicit.** The code originally ruled a pair out on *any* hash mismatch. That
silently demoted 44 genuine demographic candidates (40 BraTS↔UCSF, 4 BraTS↔RHUH) to
`unique`, while 86 UCSF↔UPENN pairs with identical evidence stayed candidates — purely
because BraTS was not on one side. Same evidence, opposite verdict.

**What breaks if reversed.** Silent loss of candidate duplicates, biased toward whichever
cohort happens to be on one side of the comparison.

---

## Research decisions live in `gbm_os`, never in the pipeline

Eligibility criteria, survival thresholds, the held-out cohort and duplicate priority are
declared as a `StudyDefinition` in `gbm_os/studies.py`. The pipeline's cohort stage owns no
criteria — it loads a named study, applies it, and writes the result.

**Why.** The pipeline answers "what is in the data". A study answers "which subset am I
studying". Mixing them means changing a research question requires editing pipeline source
and regenerating artifacts.

**How it is enforced.** Building under a different study produces a **byte-identical**
`master_manifest.csv`; only `cohort/selected.csv` changes. If a study could alter the
manifest, the facts would depend on the question being asked. Two tests hold the line: one
asserts no eligibility vocabulary appears in the cohort stage, another asserts no
`gbm_manifest` module imports `gbm_os` at top level.

**Studies are declarative on purpose** — no lambdas, no predicates — so a definition can be
printed, diffed, versioned, and pasted into a methods section. `gliostitch studies gbm-os`
prints the cohort you actually ran.

---

## Censoring is a modelling choice, not an eligibility rule

No study filters on `os_event`. The manifest records censored outcomes and studies keep
them. Whether a model may use a patient who was alive at last follow-up is decided at
selection time:

```python
view = GBM_OS_STUDY.apply(cohort)                 # 502, censoring intact
deceased = view.select(filters={"os_event": 1})   # 390, complete-case
```

**Why.** Filtering earlier makes the alternative unreachable — you cannot recover a
censoring-aware analysis from a cohort that already dropped censored cases. Filtering later
costs one line and is traced like any other criterion.

This turns on a single subject; see *Training is 377* below.

---

## Every derived quantity has exactly one implementation

`os_class`, `is_baseline`, `is_longitudinal`, `is_structural_complete` and `has_os` are
defined once, in `gbm_os/manifest.py`.

**Why this is stated as a rule.** Two implementations of `derive_os_class` used to exist —
a vectorised one in `gbm_os` and a scalar one in `gbm_manifest.core.schema`. They had
drifted. The scalar version guarded on `None` but not `NaN`, so a session with no survival
data was classified as a **long survivor**, while the other returned null. The unit test
covered the correct one; the pipeline called the other.

**What breaks if reversed.** Silent, plausible-looking mislabelling, invisible to a test
suite that happens to exercise the other copy.

---

## `gbm_os` imports the contract, never the pipeline

`gbm_os` imports `MANIFEST_COLUMNS` and `INTENSITY_PRENORMALISED` from
`gbm_manifest.core.schema`, and nothing else. Adapters, stages and anything touching raw
data are off limits.

**Why not zero coupling.** The original spec forbade the import outright. Taken literally
that forces a second copy of 31 column names inside `gbm_os` — precisely the drift the
contract exists to prevent. Importing the contract is correct; importing the pipeline is
not. The spec was amended, not the code.

**The one reverse dependency** is `gbm_manifest/stages/cohort.py`, which imports `gbm_os`
to emit the derived cohort. The import is deferred inside the function so the pipeline core
stays independent of the selection layer.

---

## Caches are keyed on inputs, not on file existence

Each cached artifact carries a `.meta.json` sidecar recording a fingerprint of everything
that determined its contents: the relevant config subtree, `SCHEMA_VERSION`, and the
upstream stage's fingerprint. Fingerprints chain, so invalidating an early stage cascades.

**Why.** Testing only for existence lets an artifact written by deleted code be served
indefinitely as current. That happened: a `cohort/selected.csv` containing a `partition`
column survived the commit that removed partition assignment, and looked entirely normal.

**Consequence.** `SCHEMA_VERSION` must be bumped whenever the *meaning* of a stored column
changes, even if the column set does not.

---

## Reconciliations

Three documented figures were wrong. Recorded here because reproducing an error to match a
number is worse than the discrepancy.

### Training is 377, not 376

The difference is `BraTS20_Training_084`: GTR, grade IV, `os_days=361`, `os_event=0`. It is
the only censored subject among BraTS's 236 survival rows, and the only one whose
`Survival_days` is not a bare integer — `"ALIVE (361 days later)"`. A reconnaissance script
parsing that column as an integer drops it silently, which is where 376 came from.
`parse_survival_days` handles the dual encoding, so the subject is retained.

Reaching 376 would require dropping BraTS's one censored case while keeping RHUH's 7 and
UCSF's 104 — an inconsistency, not a rule. Applying `os_event == 1` uniformly gives 265.

### External is 125, not 131

The two build specs disagreed: one omitted the has-OS criterion the other required. Six
UPENN patients meet every imaging and surgical criterion but carry no survival annotation,
so a survival model has no label to train on or evaluate against. They are excluded.
Training is unaffected — 377 either way, since all six are UPENN.

`gbm-os-no-survival-filter` reproduces the original 131 for reconciliation only.

### RHUH has 40 patients, not 43

One spec section said 43, another said 40. Disk says 40 across 120 sessions.

### The lesson

Two of these concealed nothing; one hid a real defect for a while, because a wrong target
made a wrong result look expected. **Treat a documented number as a claim to verify against
the data, and never tune a filter until it reproduces a figure.**

---

## The defect worth knowing about

`UPENN-GBM/clinical_info.csv` holds one row per *session*. 41 patients carry both a `_11`
baseline and a `_21` follow-up row. The adapter keyed its clinical dictionary on the
suffix-stripped patient id, so the follow-up row overwrote the baseline — carrying an EOR of
`"Not Applicable"` and the follow-up scan's age onto 41 baseline sessions.

```
UPENN baseline EOR   before   GTR 326 / non_GTR 206 / unknown 79
                     after    GTR 362 / non_GTR 211 / unknown 38   (== the _11 rows on disk)
```

36 GTR cases were silently reclassified as unknown, and 14 of them belonged in the external
arm. Nothing failed; the numbers simply looked plausible.

**What it changed.** `load_clinical` now asserts `len(records) == len(rows)` and raises
`ClinicalKeyCollision` otherwise. Every adapter should carry that assertion — it is cheap,
and this class of bug is invisible without it.
