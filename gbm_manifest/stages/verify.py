"""verify-layout stage: pre-flight check of each dataset's on-disk structure."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .. import adapters as _adapters_mod  # noqa: F401 — triggers self-registration
from ..adapters.base import get_adapter
from ..core.layout import LayoutIssue
from ..core.schema import Dataset

if TYPE_CHECKING:
    from ..config import PipelineConfig


def run_layout_check(config: "PipelineConfig") -> dict[str, list[LayoutIssue]]:
    """Call check_layout() for every enabled dataset in config."""
    results: dict[str, list[LayoutIssue]] = {}
    for ds_name, ds_cfg in config.datasets.items():
        if not ds_cfg.enabled:
            continue
        try:
            dataset = Dataset(ds_name)
        except ValueError:
            continue
        from pathlib import Path
        adapter = get_adapter(dataset, Path(ds_cfg.root))
        results[ds_name] = adapter.check_layout()
    return results
