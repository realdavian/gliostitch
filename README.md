# gliostitch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21454905.svg)](https://doi.org/10.5281/zenodo.21454905)
[![PyPI](https://img.shields.io/pypi/v/gliostitch)](https://pypi.org/project/gliostitch/)
[![Python](https://img.shields.io/pypi/pyversions/gliostitch)](https://pypi.org/project/gliostitch/)
[![Licence](https://img.shields.io/badge/licence-GPL--3.0--or--later-blue)](LICENSE)
[![CI](https://github.com/realdavian/gliostitch/actions/workflows/ci.yml/badge.svg)](https://github.com/realdavian/gliostitch/actions/workflows/ci.yml)

Reproducible cohort selection over four public glioma MRI datasets — stitched into one
manifest of facts, with a study definition layered on top.

Assembling a multi-cohort GBM study means reconciling four different directory layouts,
four clinical CSV conventions, three ID formats, two segmentation label schemes, and an
unknown number of patients who appear in more than one dataset. Most of that work gets
done once, in a notebook, and is difficult to reproduce six months later.

This project does it as a pipeline. It emits one manifest of **facts** about every imaging
session, and a separate layer that turns those facts into a **cohort** according to a
named, versioned study definition. The split matters: the manifest never changes when the
research question does.

```
1661 imaging sessions  →  study definition  →  502 sessions (377 training / 125 external)
                                                with a reason attached to all 1159 exclusions
```

**Datasets:** BraTS-2020 · UCSF-PDGM · UPENN-GBM · RHUH-GBM
*(not redistributed — you supply the data and configure its location)*

---

## Install

```bash
pip install gliostitch              # selection + path resolution
pip install "gliostitch[pipeline]"  # + build the manifest yourself
pip install "gliostitch[load]"      # + read NIfTI volumes
pip install "gliostitch[monai]"     # + framework backends: monai / torchio / torch
```

The core is `pandas`, `numpy` and `pydantic` only. If you just want to query a manifest
someone else built, you never install the pipeline.

---

## Quickstart

### Build the manifest

Point `config/pipeline.yaml` at your dataset roots, then:

```bash
gliostitch verify-layout   # check the directories before doing any work
gliostitch build           # → output/master_manifest.csv
```

### Select a cohort

```python
from gbm_os import Cohort
from gbm_os.studies import GBM_OS_STUDY

cohort = Cohort.from_manifest(
    "output/master_manifest.csv",
    data_roots={"brats2020": "/data/BraTS-2020", ...},
)

view = GBM_OS_STUDY.apply(cohort)
print(len(view))                      # 502

for spec in view:                     # SampleSpec: resolved paths + clinical metadata
    print(spec.paths["t1ce"], spec.os_days, spec.os_class)
```

Or express criteria directly — all optional, all composable:

```python
view = cohort.select(
    datasets=["ucsf_pdgm", "rhuh_gbm"],
    baseline_only=True,
    require_complete=True,
    filters={"eor": "GTR", "who_grade": [4, None]},
    where=lambda r: r["age"] is not None and r["age"] >= 18,
    resolve_duplicates="drop",
)
```

### Load it

```python
ds = view.to_monai()          # or .to_torchio() / .to_torch()
folds = view.split(k=5, seed=42)      # stratified, patient-grouped, deterministic
train = folds.fold(0, "train")
```

---

## Why it is built this way

**The manifest records facts, not decisions.** One row per imaging session, 31 columns,
covering identity, paths, modality presence, and normalised clinical values. It does not
store `os_class`, `is_baseline` or `partition` — those are modelling choices, derived at
load time. A fact discarded at ingest is unrecoverable; a derivation recomputed on load
costs milliseconds.

**A study is a separate, versioned object.** Eligibility criteria, survival thresholds and
the held-out cohort live in `gbm_os.studies`, declaratively — no lambdas, so a definition
can be printed, diffed and pasted into a methods section.

```bash
$ gliostitch studies gbm-os
gbm-os v2
Overall-survival classification over baseline preoperative GBM MRI, held out on UPENN-GBM.

Eligibility:
  - baseline session only (session_index == 0)
  - all four structural modalities present
  - eor == 'GTR'
  - who_grade in [4, None]
  - has_os == True
```

Changing the study cannot change the manifest — building under a different one produces a
byte-identical CSV. That property is enforced by a test.

**Every exclusion is accounted for.** Selection records what each criterion removed, so a
cohort table is derivable rather than asserted:

```python
>>> print(view.provenance())
input                      1661
baseline_only              1661 →   1521  (−140)
require_complete           1521 →   1126  (−395)
filter:eor                 1126 →    526  (−600)
filter:who_grade            526 →    509  (−17)
filter:has_os               509 →    503  (−6)
resolve_duplicates          503 →    502  (−1)
selected                    502

>>> view.exclusions()      # every dropped session, tagged with its reason
```

`len(view) + len(view.exclusions())` always equals the input.

**Duplicates are annotated, never deleted.** Patients appearing in more than one dataset
are flagged with a group id and a confidence level. Which copy to keep is a selection
policy, not a manifest edit.

---

## Documentation

| | |
|---|---|
| [Architecture](developer-docs/01_architecture.md) | How the two packages relate and which one a change belongs in |
| [Extending](developer-docs/02_extending.md) | Adding a dataset, column, study, criterion or backend |
| [Design decisions](developer-docs/03_design_decisions.md) | Why it is shaped this way, and what breaks if reversed |
| [Versioning](developer-docs/04_versioning.md) | Release process, and which number to bump |
| [Usage notebook](notebooks/01_gbm_os_usage.ipynb) | Worked end-to-end walkthrough |

---

## Development

```bash
source activate.sh
pytest                   # 246 tests
pytest tests/manifest    # pipeline only — runs against a synthetic dataset tree,
                         # no real data needed, ~1 s
```

The pipeline suite builds a miniature four-dataset tree on the fly and runs the real
adapters against it, so join contracts are testable without access to the MRI data. Tests
that need a built manifest skip themselves with a reason.

---

## Status

Beta. The cohort figures above are current as of the latest entry in
[CHANGELOG.md](CHANGELOG.md), which records data-affecting changes separately from code
changes — a fix that moves a cohort count is breaking for anyone who published against the
old one.

## Citation

If you use this in published work, please cite it. GitHub's *Cite this repository* button
reads [`CITATION.cff`](CITATION.cff) and will produce BibTeX or APA for you.

> Lim, Wei Xin (Davian). *gliostitch: reproducible cohort selection over public glioma MRI
> datasets*. Zenodo, 2026. https://doi.org/10.5281/zenodo.21454905

**DOI:** [10.5281/zenodo.21454905](https://doi.org/10.5281/zenodo.21454905) — the *concept* DOI, which always
resolves to the latest version. To pin a specific one, cite its version DOI instead
(v0.1.0 is [10.5281/zenodo.21454906](https://doi.org/10.5281/zenodo.21454906)).

ORCID: [0009-0005-2683-1528](https://orcid.org/0009-0005-2683-1528)

Please also cite the source datasets — BraTS-2020, UCSF-PDGM, UPENN-GBM and RHUH-GBM each
have their own citation requirements, and this project redistributes none of them.

## Licence

[GNU General Public License v3.0 or later](LICENSE).

You may use, modify and redistribute this software, including commercially. If you
distribute a modified version, or software that incorporates this one, that work must also
be released under the GPL. Running it to produce research results places no obligation on
your results — only on distributed *software*.
