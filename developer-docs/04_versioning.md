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

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml, per the table above.
# 2. Move CHANGELOG's [Unreleased] into a dated section for the new version.
# 3. Update the version and date in CITATION.cff.
# 4. Merge to main through a PR — direct pushes are blocked.
# 5. Tag from main:

git checkout main && git pull
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

The tag triggers `.github/workflows/release.yml`, which:

1. **Re-runs the full suite** on 3.11/3.12/3.13 — a tag never publishes untested code
2. **Refuses to continue if the tag disagrees with `pyproject.toml`**, so a published
   version is always reproducible from the repository
3. Builds and runs `twine check --strict`
4. Publishes to PyPI via **Trusted Publishing** (OIDC — no API token stored anywhere)
5. Creates the GitHub release, which is what Zenodo watches to mint a DOI

Nothing publishes from a branch, and nothing publishes without passing tests.

---

## One-time setup before the first release

**PyPI Trusted Publishing** — at <https://pypi.org/manage/account/publishing/>, add a
pending publisher:

```
PyPI project name : gliostitch
Owner             : realdavian
Repository        : gbm-surv-data-pipeline
Workflow          : release.yml
Environment       : pypi
```

No token is generated or stored; PyPI verifies the workflow's OIDC identity at publish
time.

**Zenodo** — requires the repository to be public. Enable it for the repo at
<https://zenodo.org/account/settings/github/>, then the *next* GitHub release mints a DOI.
Add the badge to the README and the DOI to `CITATION.cff` afterwards.
