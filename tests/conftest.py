"""Shared fixtures for gbm_manifest and gbm_os tests.

Two fixture families:

  synthetic_*  — a miniature four-dataset tree built on the fly (tests/synthetic.py).
                 Runs the REAL adapters and the REAL pipeline, needs no /mnt/disk1,
                 and is what guards the adapter join contracts in CI.

  cohort       — the production manifest at output/master_manifest.csv. Skipped
                 with a clear reason when it has not been generated, so a fresh
                 clone gets a green (partially skipped) suite instead of errors.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MANIFEST_PATH = Path(__file__).parent.parent / "output" / "master_manifest.csv"

DATA_ROOTS = {
    "brats2020": "/mnt/disk1/datasets/BraTS-2020",
    "rhuh_gbm": "/mnt/disk1/datasets/RHUH-GBM",
    "ucsf_pdgm": "/mnt/disk1/datasets/UCSF-PDGM",
    "upenn_gbm": "/mnt/disk1/datasets/UPENN-GBM",
}


# ── synthetic tree ────────────────────────────────────────────────────────── #

@pytest.fixture(scope="session")
def synthetic_roots(tmp_path_factory) -> dict[str, Path]:
    """Miniature on-disk replica of all four dataset layouts."""
    from tests.synthetic import build_synthetic_datasets

    base = tmp_path_factory.mktemp("datasets")
    return build_synthetic_datasets(base)


@pytest.fixture(scope="session")
def synthetic_config_path(synthetic_roots, tmp_path_factory) -> Path:
    """A pipeline.yaml wired to the synthetic tree."""
    out = tmp_path_factory.mktemp("synthetic_output")
    cfg_dir = tmp_path_factory.mktemp("synthetic_config")
    cfg_path = cfg_dir / "pipeline.yaml"

    cfg_path.write_text(yaml.safe_dump({
        "datasets": {
            name: {"root": str(root), "enabled": True}
            for name, root in synthetic_roots.items()
        },
        "output_dir": str(out),
        "workers": 2,
        "dedup": {
            "demographic_age_tol": 1.0,
            "demographic_days_tol": 5.0,
            "close_age_tol": 0.5,
            "close_days_tol": 2.0,
        },
        "cohort": {"os_short_max_days": 300.0, "os_mid_max_days": 450.0},
    }))
    return cfg_path


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_config_path):
    """The manifest produced by running the real pipeline over the synthetic tree."""
    from gbm_manifest.config import load_config
    from gbm_manifest.pipeline import Pipeline

    return Pipeline(load_config(synthetic_config_path)).build(force=True)


def _adapter(dataset_name: str, roots: dict[str, Path]):
    from gbm_manifest.adapters.base import get_adapter
    from gbm_manifest.pipeline import _DATASET_MAP

    return get_adapter(_DATASET_MAP[dataset_name], roots[dataset_name])


@pytest.fixture(scope="session")
def synthetic_adapters(synthetic_roots):
    """dataset name -> instantiated adapter over the synthetic tree."""
    return {name: _adapter(name, synthetic_roots) for name in synthetic_roots}


# ── production manifest ───────────────────────────────────────────────────── #

@pytest.fixture(scope="session")
def cohort():
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"{MANIFEST_PATH} not found — run `gbm-manifest build` to generate it. "
            "Tests that exercise adapter and pipeline logic run without it."
        )

    from gbm_os import Cohort, CohortConfig

    config = CohortConfig(
        data_roots={k: Path(v) for k, v in DATA_ROOTS.items()},
        partition_map={"external": {"upenn_gbm"}},
        priority=["brats2020", "rhuh_gbm", "ucsf_pdgm"],
    )
    return Cohort.from_manifest(MANIFEST_PATH, data_roots=DATA_ROOTS, config=config)
