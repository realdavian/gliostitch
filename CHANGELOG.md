# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because the output of this project is a research artifact, entries record **what changed
about the data**, not only what changed about the code. A fix that moves a cohort count is
a breaking change for anyone who published against the old one, and is released as MINOR
at minimum regardless of how small the code change was — see
[developer-docs/04_versioning.md](developer-docs/04_versioning.md).

---

## [0.1.1](https://github.com/realdavian/gliostitch/compare/v0.1.0...v0.1.1) (2026-07-20)


### Documentation

* record the Zenodo DOI in the repository ([e329312](https://github.com/realdavian/gliostitch/commit/e3293128377ca5dee7f6e4a076a4273c5d036efb))

## [0.1.0] - 2026-07-20

First release. Supersedes the pre-release 0.1.0 development series.

### Cohort-affecting fixes

These change the selected cohort. **Any result computed before them should be re-run.**

- **UPENN clinical records are keyed per session, not per patient.**
  `clinical_info.csv` holds one row per session, and 41 patients carry both a `_11`
  baseline and a `_21` follow-up row. Keying on the suffix-stripped patient id let the
  follow-up row overwrite the baseline, carrying an EOR of `"Not Applicable"` and the
  follow-up scan's age onto 41 baseline sessions.

  ```
  UPENN baseline EOR   before   GTR 326 / non_GTR 206 / unknown 79
                       after    GTR 362 / non_GTR 211 / unknown 38
  ```

  36 GTR cases were silently reclassified as unknown; 14 belonged in the external arm.
  The external cohort moves from 111 to **125**.

- **BraTS grade is retained for subjects with no survival row.** `name_mapping.csv` grades
  all 369 subjects, but records were built only from `survival_info.csv` (236 rows), so
  133 subjects lost a known grade — 76 LGG and 57 HGG became indistinguishable. Grade now
  survives independently of survival data; `clinical_row_found` still marks the 133 as
  OS-less.

- **Segmentation-hash mismatches no longer rule out cross-pipeline pairs.** A hash is only
  evidence between cohorts sharing an annotation pipeline. Ruling out on any mismatch
  demoted 44 genuine demographic candidates (40 BraTS↔UCSF, 4 BraTS↔RHUH) to `unique`,
  while 86 UCSF↔UPENN pairs with identical evidence stayed candidates. Same-pipeline pairs
  are now declared in `dedup.same_pipeline_pairs`.

- **A null survival time can no longer be classified as a long survivor.** Two
  implementations of `derive_os_class` existed and had drifted; the one the pipeline called
  guarded on `None` but not `NaN`. Now a single implementation, in `gbm_os.manifest`.

### Added

- **Study definitions** (`gbm_os.studies`) — eligibility criteria as declarative, versioned
  objects rather than ad-hoc arguments. `gliostitch studies` lists them.
- **Selection provenance** (`CohortView.provenance()` / `.exclusions()`) — every criterion
  applied and every row it dropped, with reasons partitioning the drops exactly, so
  `selected + excluded == 1661` always holds.
- **Cache invalidation** — each artifact carries a fingerprint sidecar covering config and
  `SCHEMA_VERSION`. Previously caches keyed on file existence alone, so output from deleted
  code was served as current.
- **Nullable set filters** — `who_grade in [4, None]` expresses "grade IV where grade is
  recorded" as one rule.
- `ClinicalKeyCollision`, raised when clinical rows collapse onto a shared key.
- Synthetic four-dataset fixture, so adapter contracts are testable without the real data.

### Changed

- **Research decisions moved out of the pipeline.** `stages/cohort.py` hardcoded the study;
  it now loads a named study from `gbm_os` and owns no criteria. Building under a different
  study yields a byte-identical manifest.
- **`where=` predicates receive Python nulls.** The documented idiom
  `r["os_days"] is not None` was always true against a raw pandas row, so the filter
  silently did nothing.
- **Core install is now pandas + numpy + pydantic.** `typer`, `pyyaml`, `pyarrow` and
  `nibabel` moved to a `[pipeline]` extra; a `[torch]` extra was added; `ipykernel` moved
  to the dev group.
- `CohortSummary.print()` → `.report()`, which returns the text instead of printing it.
- `gbm_manifest.config.CohortConfig` → `CohortStageConfig`, ending the name collision with
  `gbm_os.CohortConfig`.
- Tests grouped by the package under test; `pytest tests/manifest` runs the pipeline suite
  against synthetic data with no real datasets required.

### Documentation

- Design decisions, extension recipes, and an architecture overview in `developer-docs/`.
- Corrected three documented figures against the data: training is **377** not 376
  (`BraTS20_Training_084`, the only censored BraTS subject, was dropped by integer parsing
  of a dual-encoded field); external is **125** not 131 (six UPENN patients have no
  survival annotation); RHUH has **40** patients not 43.

### Cohort

| | Sessions |
|---|---|
| Manifest (facts superset) | 1661 |
| `gbm-os` study | 502 — 377 training / 125 external |
| with `os_event == 1` | 390 |

---

Releases from 0.2.0 onward are prepared automatically by release-please from
Conventional Commit messages; see
[developer-docs/04_versioning.md](developer-docs/04_versioning.md).
