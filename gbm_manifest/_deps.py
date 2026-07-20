"""Actionable errors for missing optional dependencies.

The core install ships the selection layer only — pandas, numpy, pydantic. The
pipeline needs typer, pyyaml, pyarrow and nibabel, which live in the [pipeline]
extra. Hitting that boundary should tell you what to install, not just name a
module you have never heard of.
"""
from __future__ import annotations

_MESSAGE = """\
{package} is needed to {purpose}, but it is not installed.

The default install ships the cohort-selection layer only. To run the pipeline:

    pip install "gliostitch[pipeline]"
"""


def pipeline_extra_required(package: str, purpose: str) -> str:
    return _MESSAGE.format(package=package, purpose=purpose)
