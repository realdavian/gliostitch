"""Stage cache validity.

Stages cache their output so a re-run skips completed work. Testing only that
the file exists is not enough: an artifact written by an earlier configuration,
or by an earlier version of the schema, is indistinguishable from a current one
and gets served indefinitely as if it were fresh.

Each cached artifact therefore carries a sidecar recording the fingerprint of
everything that determined its contents. The cache is reusable only on an exact
match; anything else — a changed threshold, a bumped schema, a missing or
unreadable sidecar — counts as a miss and the stage recomputes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SUFFIX = ".meta.json"


def fingerprint(**parts: Any) -> str:
    """Stable short digest of the inputs that determine a stage's output."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def meta_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + _SUFFIX)


def is_valid(artifact: Path, fp: str) -> bool:
    """True only if `artifact` exists and was written under fingerprint `fp`."""
    if not artifact.exists():
        return False

    mp = meta_path(artifact)
    if not mp.exists():
        log.info("cache: %s has no sidecar — treating as stale", artifact.name)
        return False

    try:
        recorded = json.loads(mp.read_text()).get("fingerprint")
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("cache: unreadable sidecar for %s (%s) — treating as stale",
                    artifact.name, exc)
        return False

    if recorded != fp:
        log.info("cache: %s was written under a different configuration "
                 "(%s != %s) — recomputing", artifact.name, recorded, fp)
        return False

    return True


def record(artifact: Path, fp: str) -> None:
    """Write the sidecar describing what produced `artifact`."""
    meta_path(artifact).write_text(json.dumps({
        "fingerprint": fp,
        "artifact": artifact.name,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
