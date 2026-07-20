# Developer Documentation

| # | Document | Read it when |
|---|---|---|
| 01 | [Package Architecture](01_architecture.md) | Orienting for the first time, or deciding which package a change belongs in |
| 02 | [Phase 1 Spec — Unified Manifest](02_spec_phase1_manifest.md) | Working on adapters, dedup, or the manifest schema |
| 03 | [Phase 2 Spec — Cohort Access](03_spec_phase2_cohort_access.md) | Working on selection, studies, splits, or backends |
| 04 | [Integration Impact](04_integration_impact.md) | Migrating `SwinUNETR-LoRA-MVP` onto these packages |

**Start with 01.** It states the rule the whole design turns on — *`gbm_manifest` records
what is true, `gbm_os` decides what to do about it* — and every other document assumes it.

---

## About the specs

02 and 03 are the original build specs, kept as-written except where reality contradicted
them. Where a spec was wrong, the correction is inline and marked, rather than silently
applied — the reasoning matters more than the number:

- **02 §12 M6** — training is 377, not 376; external is 125, not 131. Both corrections
  carry their derivation.
- **03 §2** — the dependency rule originally forbade the schema import `gbm_os` correctly
  relies on. Amended to permit importing the *contract* and forbid importing the pipeline.
- **03 §13 P2** — the grade rule is `who_grade in [4, None]`, not the earlier
  `grade-IV[UCSF]` wording, which happened to give the same answer only because every GTR
  BraTS and RHUH case is grade IV.

Treat a number in a spec as a claim to verify, not as ground truth. Two of them were
wrong, and one of those hid a real defect for a while.

---

## Where things live

```
gbm_manifest/        pipeline: adapters, stages, schema contract
gbm_os/              selection: studies, provenance, splits, backends
config/pipeline.yaml dataset roots, dedup tolerances, which study to emit
output/              generated — manifest, cohort, dedup decisions (gitignored)
tests/manifest/      pipeline tests, run against a synthetic dataset tree
tests/os/            selection tests, run against the built manifest
notebooks/           01_gbm_os_usage.ipynb — worked walkthrough
```
