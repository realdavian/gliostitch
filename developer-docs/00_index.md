# Developer Documentation

For working on or extending `gbm_manifest` and `gbm_os`.

| # | Document | Read it when |
|---|---|---|
| 01 | [Package Architecture](01_architecture.md) | Orienting for the first time, or deciding which package a change belongs in |
| 02 | [Extending the System](02_extending.md) | Adding a dataset, a column, a study, a criterion, or a backend |

**Start with 01.** It states the rule the whole design turns on — *`gbm_manifest` records
what is true, `gbm_os` decides what to do about it* — and 02 assumes it throughout.

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
internal/            original build specs and migration notes (not published)
```

`internal/` holds the specs the system was built from and the migration analysis for
`SwinUNETR-LoRA-MVP`. They are development history rather than reference material —
useful for *why* something is the way it is, not for how to use or extend it. Neither
`internal/` nor these developer docs ship in the published package.

---

## Quick reference

```bash
gbm-manifest verify-layout      # check dataset directories before building
gbm-manifest build              # produce master_manifest.csv + cohort/
gbm-manifest studies            # list study definitions
gbm-manifest studies gbm-os     # describe one

pytest                          # 237 tests, ~4 s
pytest tests/manifest           # pipeline only — synthetic data, no manifest needed
```

```python
from gbm_os import Cohort
from gbm_os.studies import GBM_OS_STUDY

cohort = Cohort.from_manifest("output/master_manifest.csv", data_roots={...})
view = GBM_OS_STUDY.apply(cohort)     # 502 sessions — 377 train / 125 external
print(view.provenance())              # how it got there
```

---

## A caution about the specs

Numbers in `internal/` came from an early reconnaissance pass and are not all correct.
Two were wrong, and one of those concealed a real defect for a while. The corrections are
recorded inline where they occur, with their derivation.

Treat a number in a spec as a claim to verify against the data, never as ground truth —
and never tune a filter until it reproduces a documented figure.
