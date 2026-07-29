# Developer Documentation

For working on or extending `gbm_manifest` and `gbm_os`.

| # | Document | Read it when |
|---|---|---|
| 01 | [Package Architecture](01_architecture.md) | Orienting for the first time, or deciding which package a change belongs in |
| 02 | [Extending the System](02_extending.md) | Adding a dataset, a column, a study, a criterion, or a backend |
| 03 | [Design Decisions](03_design_decisions.md) | Something looks odd and you want to know whether it is deliberate |
| 04 | [Versioning and Releases](04_versioning.md) | Cutting a release, or deciding which number to bump |

**Start with 01.** It states the rule the whole design turns on — *`gbm_manifest` records
what is true, `gbm_os` decides what to do about it* — and the others assume it.

**Read 03 before changing anything structural.** Several decisions exist because something
went wrong; the failure is the argument, and it is recorded alongside the rule.

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

---

## Quick reference

```bash
gliostitch verify-layout      # check dataset directories before building
gliostitch build              # produce master_manifest.csv + cohort/
gliostitch studies            # list study definitions
gliostitch studies gbm-os     # describe one

pytest                          # 237 tests, ~4 s
pytest tests/manifest           # pipeline only — synthetic data, no manifest needed
```

```python
from gbm_os import Cohort
from gbm_os.studies import GBM_OS_STUDY

cohort = Cohort.from_manifest("output/master_manifest.csv", data_roots={...})
view = GBM_OS_STUDY.apply(cohort)     # 531 sessions — 406 train / 125 external
print(view.provenance())              # how it got there
```

---

## The cohort, at a glance

```
input                      2109     every imaging session on disk
baseline_only              2109 →   1590
require_complete           1590 →   1195
filter:eor                 1195 →    572
filter:who_grade            572 →    555
filter:has_os               555 →    547
filter:acquisition_context  547 →    532
resolve_duplicates          532 →    531
selected                    531     406 training / 125 external
```

Reasons partition the drops exactly, so `selected + excluded == 2109` always holds.
