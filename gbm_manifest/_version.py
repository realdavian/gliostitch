"""Single source of truth for the package version.

The version is declared once, in pyproject.toml, and read back from installed
metadata — so there is no second copy in source to drift out of step. Both
packages expose the same value as `__version__`.

Not to be confused with `core.schema.SCHEMA_VERSION`, which versions the
*manifest format* and governs cache invalidation. They move independently: a
release can change the API without touching the schema, and a schema bump can
happen within a single release series.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _metadata_version

_DISTRIBUTION = "gliostitch"


def get_version() -> str:
    try:
        return _metadata_version(_DISTRIBUTION)
    except PackageNotFoundError:  # running from a source tree, not installed
        return "0.0.0.dev0"


__version__ = get_version()
