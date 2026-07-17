"""Thin wrappers for ThreadPoolExecutor and ProcessPoolExecutor."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def thread_map(fn: Callable[[T], R], items: Iterable[T],
               workers: int = 4, desc: str = "") -> list[R]:
    """Run fn(item) in a ThreadPool; return results in submission order."""
    items = list(items)
    if not items:
        return []
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            exc = fut.exception()
            if exc:
                log.error("%s item[%d] failed: %s", desc, i, exc)
                raise exc
            results[i] = fut.result()
    return results  # type: ignore[return-value]


def process_map(fn: Callable[[T], R], items: Iterable[T],
                workers: int = 4, desc: str = "") -> list[R]:
    """Run fn(item) in a ProcessPool; return results in submission order."""
    items = list(items)
    if not items:
        return []
    results: list[R | None] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            exc = fut.exception()
            if exc:
                log.error("%s item[%d] failed: %s", desc, i, exc)
                raise exc
            results[i] = fut.result()
    return results  # type: ignore[return-value]
