"""DatasetAdapter Protocol + registry."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ..core.layout import LayoutIssue
from ..core.schema import ClinicalRecord, Dataset, RawSession


@runtime_checkable
class DatasetAdapter(Protocol):
    name: Dataset

    def __init__(self, root: Path) -> None: ...

    def discover(self) -> Iterator[RawSession]: ...

    def load_clinical(self) -> dict[str, ClinicalRecord]: ...

    def clinical_key(self, session: RawSession) -> str: ...

    def check_layout(self) -> list[LayoutIssue]: ...


_REGISTRY: dict[Dataset, type] = {}


def register_adapter(name: Dataset):
    def _wrap(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(
                f"adapter already registered for {name.value!r}: "
                f"{_REGISTRY[name].__name__}"
            )
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_adapter(name: Dataset, root: Path) -> DatasetAdapter:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        have = sorted(d.value for d in _REGISTRY)
        raise KeyError(
            f"no adapter registered for {name.value!r}; registered: {have}"
        ) from exc
    return cls(root)


def available_adapters() -> list[Dataset]:
    return sorted(_REGISTRY, key=lambda d: d.value)
