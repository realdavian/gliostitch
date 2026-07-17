"""Shared fixtures for gbm_os tests."""
from __future__ import annotations

from pathlib import Path

import pytest

MANIFEST_PATH = Path(__file__).parent.parent / "output" / "master_manifest.csv"

DATA_ROOTS = {
    "brats2020": "/mnt/disk1/datasets/BraTS-2020",
    "rhuh_gbm": "/mnt/disk1/datasets/RHUH-GBM",
    "ucsf_pdgm": "/mnt/disk1/datasets/UCSF-PDGM",
    "upenn_gbm": "/mnt/disk1/datasets/UPENN-GBM",
}


@pytest.fixture(scope="session")
def cohort():
    from gbm_os import Cohort, CohortConfig

    config = CohortConfig(
        data_roots={k: Path(v) for k, v in DATA_ROOTS.items()},
        partition_map={"external": {"upenn_gbm"}},
        priority=["brats2020", "rhuh_gbm", "ucsf_pdgm"],
    )
    return Cohort.from_manifest(MANIFEST_PATH, data_roots=DATA_ROOTS, config=config)
