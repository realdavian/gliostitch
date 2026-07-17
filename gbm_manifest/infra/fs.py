"""Filesystem guards: enforce the read-only-over-data invariant."""
from __future__ import annotations

from pathlib import Path

from ..core.exceptions import DatasetRootViolation

_DATASET_ROOTS: list[Path] = []


def register_dataset_roots(roots: list[Path]) -> None:
    global _DATASET_ROOTS
    _DATASET_ROOTS = [Path(r).resolve() for r in roots]


def assert_output_path_safe(path: Path) -> None:
    """Raise DatasetRootViolation if path resolves under any registered dataset root."""
    resolved = Path(path).resolve()
    for root in _DATASET_ROOTS:
        try:
            resolved.relative_to(root)
            raise DatasetRootViolation(
                f"Output path {resolved} is under dataset root {root}. "
                "The pipeline must never write under a dataset root."
            )
        except ValueError:
            pass  # not under this root — OK


def safe_mkdir(path: Path) -> Path:
    assert_output_path_safe(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
