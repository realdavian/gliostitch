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


def load_config(path: Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
