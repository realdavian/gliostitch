"""LayoutIssue — shared dataclass for verify-layout results.

Kept in core/ so adapters can import it without risk of circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LayoutIssue:
    severity: Literal["error", "warning"]
    check: str      # short label, e.g. "training subdirectory"
    expected: str   # absolute path that was expected
    fix: str        # one-line human instruction
