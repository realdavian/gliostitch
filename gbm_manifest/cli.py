"""CLI entrypoint: manifest <subcommand>."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

try:
    import typer
except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
    from ._deps import pipeline_extra_required

    # SystemExit rather than a traceback: this module is only ever reached
    # through the console script, where a stack trace helps nobody.
    raise SystemExit(
        pipeline_extra_required("typer", "run the gbm-manifest command line")
    ) from exc

from .config import load_config
from .logging_setup import setup_logging

app = typer.Typer(help="GBM-OS unified manifest pipeline.")
log = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path("config/pipeline.yaml")


def _get_pipeline(config: Path, workers: Optional[int] = None):
    from .pipeline import Pipeline
    cfg = load_config(config)
    if workers is not None:
        cfg.workers = workers
    return Pipeline(cfg)


@app.command()
def build(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    workers: Annotated[Optional[int], typer.Option("--workers", "-n")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Run the full pipeline: audit → standardize → dedup → manifest → cohort."""
    setup_logging(verbose=verbose,
                  log_file=load_config(config).output_dir / "pipeline.log")
    pipeline = _get_pipeline(config, workers)
    manifest = pipeline.build(force=force)
    typer.echo(f"Done. Manifest: {len(manifest)} rows.")


@app.command()
def audit(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    workers: Annotated[Optional[int], typer.Option("--workers", "-n")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Run audit stage only (discover sessions on disk)."""
    setup_logging(verbose=verbose)
    pipeline = _get_pipeline(config, workers)
    inv = pipeline.run_audit(force=force)
    total = sum(len(df) for df in inv.values())
    typer.echo(f"Audit done. {total} sessions discovered.")


@app.command()
def standardize(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Run standardize stage (requires prior audit output)."""
    setup_logging(verbose=verbose)
    pipeline = _get_pipeline(config)
    inv = pipeline.run_audit(force=False)
    std = pipeline.run_standardize(inv, force=force)
    total = sum(len(df) for df in std.values())
    typer.echo(f"Standardize done. {total} sessions.")


@app.command()
def dedup(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    workers: Annotated[Optional[int], typer.Option("--workers", "-n")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Run deduplication stage."""
    setup_logging(verbose=verbose)
    pipeline = _get_pipeline(config, workers)
    inv = pipeline.run_audit(force=False)
    std = pipeline.run_standardize(inv, force=False)
    combined = pipeline.run_dedup(std, force=force)
    typer.echo(f"Dedup done. {len(combined)} total rows.")


@app.command()
def assemble(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Assemble the final master_manifest.csv."""
    setup_logging(verbose=verbose)
    pipeline = _get_pipeline(config)
    inv = pipeline.run_audit(force=False)
    std = pipeline.run_standardize(inv, force=False)
    combined = pipeline.run_dedup(std, force=False)
    manifest = pipeline.run_manifest(combined, force=force)
    typer.echo(f"Manifest: {len(manifest)} rows → {pipeline.out / 'master_manifest.csv'}")


@app.command()
def validate(
    manifest: Annotated[Path, typer.Option("--manifest", "-m")] = Path("output/master_manifest.csv"),
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Validate manifest: check column order, path existence, null hygiene."""
    setup_logging(verbose=verbose)
    import pandas as pd
    from .core.schema import MANIFEST_COLUMNS
    cfg = load_config(config)

    df = pd.read_csv(manifest)
    errors = []

    # Check column order
    if list(df.columns) != MANIFEST_COLUMNS:
        errors.append(f"Column order mismatch.\n  Expected: {MANIFEST_COLUMNS}\n  Got: {list(df.columns)}")

    # Check no "NaN" strings
    for col in df.columns:
        nan_str = (df[col].astype(str) == "NaN").sum()
        if nan_str:
            errors.append(f"Column {col!r} has {nan_str} literal 'NaN' strings")

    # Check path existence (sample up to 50 rows per dataset)
    path_cols = ["t1_path", "t1ce_path", "t2_path", "flair_path", "seg_path"]
    for ds_name, ds_cfg in cfg.datasets.items():
        root = Path(ds_cfg.root)
        sub = df[df["dataset"] == ds_name].dropna(subset=["t1_path"]).head(50)
        for _, row in sub.iterrows():
            for col in path_cols:
                if pd.notna(row.get(col)):
                    p = root / row[col]
                    if not p.exists():
                        errors.append(f"Missing file: {p}")

    if errors:
        typer.echo(f"VALIDATION FAILED ({len(errors)} errors):", err=True)
        for e in errors[:20]:
            typer.echo(f"  {e}", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"OK: {len(df)} rows, {len(df.columns)} columns, paths verified.")


@app.command()
def studies(
    name: Annotated[Optional[str], typer.Argument()] = None,
):
    """List the available study definitions, or describe one."""
    from gbm_os.studies import STUDIES, get_study

    if name is not None:
        typer.echo(get_study(name).describe())
        return

    for study in STUDIES.values():
        typer.echo(f"{study.name:<12} v{study.version}  {study.description}")
    typer.echo("\nRun `gbm-manifest studies <name>` for the full definition.")


@app.command("verify-layout")
def verify_layout(
    config: Annotated[Path, typer.Option("--config", "-c")] = _DEFAULT_CONFIG,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Check each dataset's on-disk directory structure before running the pipeline."""
    setup_logging(verbose=verbose)
    from .stages.verify import run_layout_check

    cfg = load_config(config)
    results = run_layout_check(cfg)

    error_count = 0
    for ds_name, issues in results.items():
        ds_cfg = cfg.datasets[ds_name]
        typer.echo(f"\nChecking {ds_name}  @  {ds_cfg.root}")
        if not issues:
            typer.echo("  ✓  all checks passed")
            continue
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        for issue in errors:
            typer.echo(f"  ✗  [error] {issue.check}")
            typer.echo(f"     Expected : {issue.expected}")
            typer.echo(f"     Fix      : {issue.fix}")
        for issue in warnings:
            typer.echo(f"  ⚠  [warning] {issue.check}")
            typer.echo(f"     Expected : {issue.expected}")
            typer.echo(f"     Fix      : {issue.fix}")
        error_count += len(errors)

    typer.echo("")
    if error_count:
        typer.echo(
            f"Summary: {error_count} error(s) found — fix the above before running "
            "`gbm-manifest build`",
            err=True,
        )
        raise typer.Exit(1)
    else:
        typer.echo("Summary: all layout checks passed.")


if __name__ == "__main__":
    app()
