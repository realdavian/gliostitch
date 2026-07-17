"""Parquet and CSV read/write helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fs import assert_output_path_safe


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    assert_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
