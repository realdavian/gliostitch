"""Provenance for a selection: what each criterion removed, and why.

A cohort count on its own is not defensible — a reviewer needs the flow from the
full facts superset down to the selected set, with a reason attached to every
dropped session. Phase 1 wrote that accounting to exclusions.csv; carrying it on
the view means the reasoning travels with the selection instead of being
reconstructed afterwards.

Criteria are recorded in application order and each excluded row is attributed
to the FIRST criterion that removed it, so the reasons partition the drops
exactly: sum(step.n_dropped) == len(exclusions).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EXCLUSION_REASON = "exclusion_reason"


@dataclass(frozen=True)
class SelectionStep:
    """One criterion applied, with its effect on the row count."""

    criterion: str
    detail: str
    n_before: int
    n_after: int

    @property
    def n_dropped(self) -> int:
        return self.n_before - self.n_after

    def __str__(self) -> str:
        return (f"{self.criterion:<24} {self.n_before:>6} → {self.n_after:>6}"
                f"  (−{self.n_dropped})")


@dataclass
class SelectionTrace:
    """Ordered record of the criteria applied to produce a view."""

    n_input: int = 0
    steps: list[SelectionStep] = field(default_factory=list)
    _excluded: list[pd.DataFrame] = field(default_factory=list)

    def record(self, criterion: str, detail: str, before: pd.DataFrame,
               after: pd.DataFrame) -> None:
        self.steps.append(SelectionStep(
            criterion=criterion, detail=detail,
            n_before=len(before), n_after=len(after),
        ))
        dropped = before.loc[before.index.difference(after.index)]
        if not dropped.empty:
            tagged = dropped.copy()
            tagged[EXCLUSION_REASON] = criterion
            self._excluded.append(tagged)

    @property
    def n_output(self) -> int:
        return self.steps[-1].n_after if self.steps else self.n_input

    def to_frame(self) -> pd.DataFrame:
        """The criterion-by-criterion flow, as a table."""
        return pd.DataFrame([{
            "criterion": s.criterion,
            "detail": s.detail,
            "n_before": s.n_before,
            "n_after": s.n_after,
            "n_dropped": s.n_dropped,
        } for s in self.steps])

    def exclusions(self) -> pd.DataFrame:
        """Every dropped row, tagged with the criterion that dropped it."""
        if not self._excluded:
            return pd.DataFrame()
        return pd.concat(self._excluded, ignore_index=True)

    def __str__(self) -> str:
        if not self.steps:
            return f"no criteria applied ({self.n_input} rows)"
        lines = [f"{'input':<24} {self.n_input:>6}"]
        lines += [str(s) for s in self.steps]
        lines.append(f"{'selected':<24} {self.n_output:>6}")
        return "\n".join(lines)
