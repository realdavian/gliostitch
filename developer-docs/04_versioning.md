# Versioning and Releases

This project follows [Semantic Versioning 2.0.0](https://semver.org), with one adaptation
that matters for research software.

---

## The adaptation: output is part of the public interface

Semver defines compatibility in terms of the **API**. For a data pipeline that is not
enough. A patch-level bug fix that leaves every function signature untouched can still
change which patients are in your cohort — and anyone who published against the old
numbers now has results that no longer reproduce.

So this project treats **the values it emits** as part of the public interface:

> **If a change moves a cohort count, a manifest value, or a duplicate verdict, it is at
> minimum a MINOR release, and it is recorded in a dedicated CHANGELOG section — however
> small the code change was.**

The alternative silently rewrites history. The C-1 fix in the first release moved the
external arm from 111 to 125 sessions; shipped as a patch, that would have quietly
invalidated every result computed against it.

---

## What each bump means here

| | Trigger |
|---|---|
| **MAJOR** | An incompatible API change; a manifest column removed or renamed; a change to what an existing column *means*; a study definition altered such that published numbers no longer reproduce |
| **MINOR** | New functionality — a dataset, study, backend, criterion; a new manifest column; **or any bug fix that changes emitted values** |
| **PATCH** | Fixes with no effect on output: crashes, error messages, performance, docs, types |

Pre-1.0 (`0.x.y`), the guarantee is weaker by convention: minor bumps may break things. Say
so plainly in the changelog rather than relying on the convention being known.

---

## Two version numbers, moving independently

| | Meaning | Lives in |
|---|---|---|
| `__version__` | The release. Semver, as above. | `pyproject.toml` |
| `SCHEMA_VERSION` | The manifest *format*. Governs cache invalidation. | `core/schema.py` |

They are deliberately separate. A release can change the API without touching the schema;
a schema bump can happen inside one release series.

**Bump `SCHEMA_VERSION` whenever the meaning of a stored column changes, even if the
column set does not.** Cached artifacts written under an older schema are then discarded
rather than silently reused — the failure mode that let output from deleted code be served
as current.

The package version is declared **once**, in `pyproject.toml`, and read back from installed
metadata:

```python
>>> import gbm_os; gbm_os.__version__
'0.1.0'
```

There is no second literal in source to drift. A test asserts both packages report the same
value as `pyproject.toml`.

---

## Commit messages drive the version

Releases are prepared by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org) messages, so the subject
line you write determines the next version number.

```
feat: add RHUH follow-up sessions      → MINOR
fix: correct UCSF zero-pad join        → PATCH
docs: clarify the grade rule           → no release
cohort: restore 14 GTR cases to UPENN  → MINOR, own changelog section
```

`cohort:` is a project-specific type for changes that move emitted values. Use it whenever
a cohort count, manifest value or duplicate verdict changes — it forces the MINOR bump the
policy above requires and files the entry under a section telling readers to re-run.

A breaking change is a `!` after the type, or a `BREAKING CHANGE:` footer.

---

## Cutting a release

Nothing is tagged by hand. The loop is:

1. **Merge work into `main`** through a PR.
2. **release-please opens or updates a release PR** against `main`, containing the version
   bump in `pyproject.toml`, the new `CHANGELOG.md` section, and the updated version in
   `CITATION.cff`.
3. **Review that PR.** This is the checkpoint that matters — see below.
4. **Merge it.** That creates the tag.
5. The tag triggers `release.yml`, which re-runs the full suite on 3.11/3.12/3.13, refuses
   to continue if the tag disagrees with `pyproject.toml`, builds, runs
   `twine check --strict`, publishes to PyPI via **Trusted Publishing** (OIDC, no stored
   token), and creates the GitHub release that Zenodo turns into a DOI.

Nothing publishes from a branch, and nothing publishes without passing tests.

### Why the release PR is not auto-merged

Because no tool can enforce the rule at the top of this document. If a fix changes a cohort
count but was committed as `fix:`, release-please will propose a PATCH — which this project
says is wrong. The release PR is where a human catches that.

To override the computed version, add a footer to any commit in the release, or edit the
release PR:

```
Release-As: 0.2.0
```

### Bootstrapping

`0.1.0` was tagged by hand, because its changelog was written before this automation
existed and regenerating it from commit history would have buried the narrative. Its entry
in `.release-please-manifest.json` is what tells release-please where to start counting.
Every release from `0.2.0` is automated.

---

## One-time setup before the first release

**PyPI Trusted Publishing** — at <https://pypi.org/manage/account/publishing/>, add a
pending publisher:

```
PyPI project name : gliostitch
Owner             : realdavian
Repository        : gliostitch
Workflow          : release.yml
Environment       : pypi
```

No token is generated or stored; PyPI verifies the workflow's OIDC identity at publish
time.

**Zenodo** — requires the repository to be public. Enable it for the repo at
<https://zenodo.org/account/settings/github/>, then the *next* GitHub release mints a DOI.
Add the badge to the README and the DOI to `CITATION.cff` afterwards.
