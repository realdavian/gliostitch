"""NIfTI array hashing with (path, size, mtime) cache."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

# Process-local cache: (path_str, size, mtime) -> hex_digest
_CACHE: dict[tuple[str, int, float], str] = {}


def _cache_key(path: Path) -> tuple[str, int, float]:
    st = os.stat(path)
    return (str(path), st.st_size, st.st_mtime)


def hash_seg(path: Path) -> Optional[str]:
    """MD5 of the seg array cast to int16."""
    return _hash_array(path, dtype=np.int16)


def hash_t1ce(path: Path) -> Optional[str]:
    """MD5 of the T1ce array cast to float32."""
    return _hash_array(path, dtype=np.float32)


def _hash_array(path: Path, dtype) -> Optional[str]:
    if not path.exists():
        return None
    key = _cache_key(path)
    if key in _CACHE:
        return _CACHE[key]
    try:
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj, dtype=dtype)
        digest = hashlib.md5(arr.tobytes()).hexdigest()
        _CACHE[key] = digest
        return digest
    except Exception as exc:
        log.warning("Failed to hash %s: %s", path, exc)
        return None


def get_shape_affine(path: Path) -> Optional[tuple[tuple, tuple]]:
    """Return (shape, affine_row0) for a quick pre-filter."""
    if not path.exists():
        return None
    try:
        img = nib.load(str(path))
        return (tuple(img.shape), tuple(img.affine[0].tolist()))
    except Exception:
        return None
