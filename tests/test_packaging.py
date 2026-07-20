"""The core of gbm_os must stay installable with pandas + numpy + pydantic.

Spec 02 §5: selection and path resolution work with no loading backend and no
pipeline tooling installed. A stray top-level import of typer, pyarrow, yaml or
nibabel in a core module silently makes the light install impossible, and
nothing else would catch it.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

# Spec 02 §4 — the core is manifest + selection + cohort + sample + config.
# studies and provenance are pure declarations built on them.
CORE_MODULES = [
    "gbm_os/__init__.py",
    "gbm_os/config.py",
    "gbm_os/manifest.py",
    "gbm_os/selection.py",
    "gbm_os/cohort.py",
    "gbm_os/sample.py",
    "gbm_os/studies.py",
    "gbm_os/provenance.py",
]

HEAVY = {"typer", "pyarrow", "yaml", "nibabel", "monai", "torchio", "torch"}


def _toplevel_imports(path: Path) -> set[str]:
    """Modules imported at import time — deferred imports inside functions are fine."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in tree.body:                       # top level only
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_has_no_heavy_imports(module):
    offenders = _toplevel_imports(REPO / module) & HEAVY
    assert not offenders, (
        f"{module} imports {sorted(offenders)} at module level — "
        "defer it inside the function that needs it"
    )


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((REPO / "pyproject.toml").read_text())


class TestDeclaredDependencies:
    def test_core_deps_are_light(self, pyproject):
        deps = {d.split(">")[0].split("=")[0].strip()
                for d in pyproject["project"]["dependencies"]}
        assert deps == {"pandas", "numpy", "pydantic"}

    def test_pipeline_extra_exists(self, pyproject):
        extras = pyproject["project"]["optional-dependencies"]
        names = {d.split(">")[0].strip() for d in extras["pipeline"]}
        assert {"typer", "pyyaml", "pyarrow", "nibabel"} <= names

    def test_torch_extra_exists(self, pyproject):
        """Spec 02 §12 lists a [torch] extra and torch_backend.py is written."""
        assert "torch" in pyproject["project"]["optional-dependencies"]

    def test_notebook_tooling_is_not_a_runtime_dep(self, pyproject):
        deps = " ".join(pyproject["project"]["dependencies"])
        assert "ipykernel" not in deps and "jupyter" not in deps


def test_gbm_manifest_depends_on_gbm_os_only_in_the_cohort_stage():
    """The cohort stage emits a derived artifact, so it may use the selection
    layer. Nothing else in the pipeline may, and even there the import is
    deferred so the pipeline core stays independent."""
    offenders = []
    for path in (REPO / "gbm_manifest").rglob("*.py"):
        if "gbm_os" in _toplevel_imports(path):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], f"top-level gbm_os imports in {offenders}"


class TestOptionalDependencyErrors:
    """Hitting the core/pipeline boundary must say what to install.

    A bare ModuleNotFoundError naming 'typer' tells a user nothing — they did
    not ask for typer, they asked for a manifest.
    """

    def test_message_names_the_fix(self):
        from gbm_manifest._deps import pipeline_extra_required

        msg = pipeline_extra_required("typer", "run the command line")
        assert "typer" in msg
        assert "run the command line" in msg
        assert 'pip install "gliostitch[pipeline]"' in msg

    @pytest.mark.parametrize("module, package", [
        ("gbm_manifest/cli.py", "typer"),
        ("gbm_manifest/config.py", "yaml"),
    ])
    def test_pipeline_only_imports_are_guarded(self, module, package):
        """Every pipeline-only import must be wrapped, or the core install
        fails with an unhelpful traceback."""
        source = (REPO / module).read_text()
        assert "pipeline_extra_required" in source, (
            f"{module} imports {package} without an actionable error"
        )


class TestVersioning:
    """One version, declared once, reachable from both packages."""

    def test_packages_agree_with_pyproject(self, pyproject):
        import gbm_manifest
        import gbm_os

        declared = pyproject["project"]["version"]
        assert gbm_manifest.__version__ == declared
        assert gbm_os.__version__ == declared

    def test_version_is_semver(self, pyproject):
        import re

        assert re.fullmatch(r"\d+\.\d+\.\d+([-.].+)?",
                            pyproject["project"]["version"])

    def test_no_duplicate_version_literal_in_source(self, pyproject):
        """The version must be read from metadata, not restated in code."""
        declared = pyproject["project"]["version"]
        for pkg in ("gbm_manifest", "gbm_os"):
            for path in (REPO / pkg).rglob("*.py"):
                assert f'"{declared}"' not in path.read_text(), (
                    f"{path} hardcodes the version — read it from _version instead"
                )

    def test_schema_version_is_independent(self):
        """SCHEMA_VERSION versions the manifest format, not the release."""
        from gbm_manifest.core.schema import SCHEMA_VERSION

        assert isinstance(SCHEMA_VERSION, int)

    def test_licence_is_declared(self, pyproject):
        assert pyproject["project"]["license"] == "GPL-3.0-or-later"
        assert (REPO / "LICENSE").exists()

    def test_citation_metadata_matches_release(self, pyproject):
        import yaml

        cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
        assert cff["version"] == pyproject["project"]["version"], (
            "CITATION.cff version is stale — update it when bumping the release"
        )
        assert cff["license"] == pyproject["project"]["license"]


class TestReleaseAutomation:
    """release-please drives the version; the three files must not drift."""

    def test_manifest_matches_pyproject(self, pyproject):
        import json

        manifest = json.loads((REPO / ".release-please-manifest.json").read_text())
        assert manifest["."] == pyproject["project"]["version"], (
            "release-please would compute the next version from the wrong base"
        )

    def test_citation_carries_the_update_annotation(self):
        """Without the annotation release-please silently leaves CITATION stale."""
        cff = (REPO / "CITATION.cff").read_text()
        version_line = next(l for l in cff.splitlines() if l.startswith("version:"))
        assert "x-release-please-version" in version_line

    def test_cohort_commit_type_is_configured(self):
        """`cohort:` must map to its own changelog section, per the versioning policy."""
        import json

        cfg = json.loads((REPO / "release-please-config.json").read_text())
        types = {s["type"]: s for s in cfg["changelog-sections"]}
        assert "cohort" in types
        assert not types["cohort"].get("hidden", False)

    def test_release_type_is_python(self):
        import json

        cfg = json.loads((REPO / "release-please-config.json").read_text())
        assert cfg["release-type"] == "python"
        assert "CITATION.cff" in cfg["packages"]["."]["extra-files"]


class TestReleasePublishHandoff:
    """release-please tags and releases using the default GITHUB_TOKEN, and
    GitHub does not fire `on: push: tags` for a push made with that token —
    deliberate loop prevention on GitHub's side, not a bug here. v0.1.1 was
    tagged and released on GitHub while never reaching PyPI because of exactly
    this: release.yml's only trigger was the tag push.

    release-please.yml must call release.yml directly instead of depending on
    the tag push to retrigger it. These assertions pin the wiring so a future
    edit can't silently drop the hand-off and reintroduce the gap.
    """

    @pytest.fixture(scope="class")
    def release_please_workflow(self):
        import yaml

        return yaml.safe_load((REPO / ".github/workflows/release-please.yml").read_text())

    @pytest.fixture(scope="class")
    def release_workflow(self):
        import yaml

        return yaml.safe_load((REPO / ".github/workflows/release.yml").read_text())

    def test_release_please_exposes_the_outputs_publish_needs(self, release_please_workflow):
        outs = release_please_workflow["jobs"]["release-please"]["outputs"]
        assert "release_created" in outs
        assert "tag_name" in outs
        # Root-path (".") components use unprefixed action outputs — no
        # "<path>--" prefix. Getting this wrong makes the outputs empty and
        # the hand-off silently never fires again.
        assert outs["release_created"] == "${{ steps.release.outputs.release_created }}"
        assert outs["tag_name"] == "${{ steps.release.outputs.tag_name }}"

    def test_publish_job_calls_release_yml_directly(self, release_please_workflow):
        pub = release_please_workflow["jobs"]["publish"]
        assert pub["uses"] == "./.github/workflows/release.yml"
        assert pub["if"] == "needs.release-please.outputs.release_created == 'true'"
        assert pub["with"]["tag"] == "${{ needs.release-please.outputs.tag_name }}"

    def test_publish_job_skips_the_duplicate_github_release(self, release_please_workflow):
        # release-please already created the GitHub release for this tag.
        assert release_please_workflow["jobs"]["publish"]["with"]["skip_github_release"] is True

    def test_publish_job_grants_oidc_for_trusted_publishing(self, release_please_workflow):
        assert release_please_workflow["jobs"]["publish"]["permissions"]["id-token"] == "write"

    def test_release_yml_accepts_the_handoff(self, release_workflow):
        # YAML parses the bare `on:` key as boolean True.
        triggers = release_workflow[True]
        assert "workflow_call" in triggers
        call_inputs = triggers["workflow_call"]["inputs"]
        assert set(call_inputs) == {"tag", "skip_github_release"}
        assert call_inputs["tag"]["required"] is True

    def test_manual_recovery_path_exists(self, release_workflow):
        """workflow_dispatch with a tag input, for retroactively publishing a
        release-please tag that never reached PyPI (as v0.1.1 did)."""
        triggers = release_workflow[True]
        dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
        assert "tag" in dispatch_inputs and dispatch_inputs["tag"]["required"] is True
        assert "skip_github_release" in dispatch_inputs

    def test_verify_and_build_pin_the_ref_to_the_tag(self, release_workflow):
        """Without this, a workflow_call run (triggered from a `main`-branch
        push) would check out `main`'s tip via the default ref rather than the
        tag it was asked to publish."""
        for job_name in ("verify", "build"):
            steps = release_workflow["jobs"][job_name]["steps"]
            checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout"))
            assert checkout["with"]["ref"] == "${{ inputs.tag || github.ref_name }}"

    def test_github_release_job_is_conditional_on_the_skip_flag(self, release_workflow):
        assert release_workflow["jobs"]["github-release"]["if"] == \
            "${{ inputs.skip_github_release != true }}"

    def test_publish_reports_success_only_if_pypi_actually_has_it(self, release_workflow):
        """Every prior workflow reported success for v0.1.1 while it never
        reached PyPI. `publish` reporting success only means the upload API
        call returned 200 — confirm against the real index rather than
        trusting that."""
        confirm = release_workflow["jobs"]["confirm"]
        assert confirm["needs"] == "publish"
        run_script = confirm["steps"][-1]["run"]
        assert "pypi.org/pypi/gliostitch" in run_script
