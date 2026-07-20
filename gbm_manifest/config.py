"""Load and validate pipeline.yaml configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    from ._deps import pipeline_extra_required

    # ImportError, not SystemExit: this is library code and a caller may want
    # to handle it.
    raise ModuleNotFoundError(
        pipeline_extra_required("PyYAML", "read config/pipeline.yaml")
    ) from exc


class DatasetConfig(BaseModel):
    root: Path
    enabled: bool = True


class DedupConfig(BaseModel):
    demographic_age_tol: float = 1.0
    demographic_days_tol: float = 5.0
    close_age_tol: float = 0.5
    close_days_tol: float = 2.0

    # Dataset pairs that share a segmentation/intensity pipeline, so a hash
    # comparison between them is meaningful evidence in BOTH directions.
    # BraTS-2020 redistributes UPENN's BraTS-pipeline output, which is the only
    # literal-reuse path across these four cohorts. Everywhere else the cohorts
    # were annotated and normalised independently, so a hash MISMATCH says
    # nothing about whether two rows are the same person (spec 01 §9) — such
    # pairs stay demographic candidates rather than being ruled out.
    same_pipeline_pairs: list[list[str]] = [["brats2020", "upenn_gbm"]]

    def shares_pipeline(self, a: str, b: str) -> bool:
        return any({a, b} == set(pair) for pair in self.same_pipeline_pairs)


class CohortStageConfig(BaseModel):
    """Which study the cohort stage should emit.

    Deliberately thin: eligibility rules, survival thresholds, the held-out
    cohort and duplicate priority are properties of the STUDY and live in
    gbm_os.studies, so they cannot drift between the pipeline and the
    selection layer. This only names which one to run.
    """

    study: str = "gbm-os"


class PipelineConfig(BaseModel):
    datasets: dict[str, DatasetConfig]
    output_dir: Path
    workers: int = 4
    dedup: DedupConfig = DedupConfig()
    cohort: CohortStageConfig = CohortStageConfig()

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(v).expanduser().resolve()

    def dataset_roots(self) -> list[Path]:
        return [Path(d.root).resolve() for d in self.datasets.values()]


TEMPLATE = """\
# gliostitch pipeline configuration.
#
# Point each dataset at the directory you extracted it into, then run:
#     gliostitch verify-layout -c {path}
#     gliostitch build         -c {path}
#
# Set `enabled: false` for any dataset you do not have — the pipeline builds a
# manifest from whichever subset is available.

datasets:
  brats2020:
    root: /path/to/BraTS-2020
    enabled: true
  rhuh_gbm:
    root: /path/to/RHUH-GBM
    enabled: true
  upenn_gbm:
    root: /path/to/UPENN-GBM
    enabled: true
  ucsf_pdgm:
    root: /path/to/UCSF-PDGM
    enabled: true

# Where the manifest and derived cohort are written.
output_dir: ./output

# Parallelism for discovery and hashing.
workers: 8

dedup:
  demographic_age_tol: 1.0
  demographic_days_tol: 5.0
  close_age_tol: 0.5
  close_days_tol: 2.0
  # Dataset pairs sharing a segmentation/intensity pipeline, where a hash
  # comparison is meaningful evidence in both directions. Between independently
  # annotated cohorts a mismatch proves nothing, so those stay candidates.
  same_pipeline_pairs:
    - [brats2020, upenn_gbm]

# Which study the cohort stage emits. Run `gliostitch studies` to list them.
cohort:
  study: gbm-os
"""


def write_template(path: Path) -> None:
    """Write a starter configuration to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.format(path=path))


def load_config(path: Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No configuration at {path}.\n\n"
            f"Create a starter one with:\n\n"
            f"    gliostitch init -c {path}\n\n"
            f"then edit it to point at your dataset directories."
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
