"""Deduplication stage: three-tier demographic → seg_hash → t1ce_hash.

Scope: the hash tiers run only on pairs that share a segmentation/intensity
pipeline (DedupConfig.same_pipeline_pairs — BraTS/UPENN by default). Between
independently annotated cohorts a hash mismatch is not evidence of anything,
so those pairs stay demographic candidates instead of being ruled out.
"""
from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import DedupConfig
from ..core.schema import Dataset
from ..infra import cache
from ..infra.io import write_csv, write_parquet

log = logging.getLogger(__name__)


def _str_or_none(v) -> Optional[str]:
    """Convert pandas NaN / None / float to None; leave valid strings intact."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    return s if s and s != "nan" else None


@dataclass
class _Pair:
    key_a: str
    key_b: str
    dataset_a: str
    dataset_b: str
    age_diff: float
    days_diff: float
    sublabel: str          # "close" | "loose"
    seg_path_a: Optional[str]
    seg_path_b: Optional[str]
    t1ce_path_a: Optional[str]
    t1ce_path_b: Optional[str]
    root_a: Path
    root_b: Path


def _hash_seg_worker(args: tuple[str, str]) -> tuple[str, Optional[str]]:
    """Top-level function for ProcessPool pickling."""
    path_str, dtype_str = args
    from ..infra.hashing import hash_seg, hash_t1ce
    p = Path(path_str)
    if dtype_str == "seg":
        return path_str, hash_seg(p)
    return path_str, hash_t1ce(p)


class Deduplicator:
    def __init__(self, dataset_roots: dict[str, Path], output_dir: Path,
                 cfg: DedupConfig, workers: int = 4,
                 fingerprint: str = "") -> None:
        self.dataset_roots = dataset_roots  # dataset_value -> root Path
        self.output_dir = output_dir / "dedup"
        self.cfg = cfg
        self.workers = workers
        self.fingerprint = fingerprint

    def run(self, standardized: dict[Dataset, pd.DataFrame],
            force: bool = False) -> pd.DataFrame:
        decisions_path = self.output_dir / "decisions.csv"
        combined_path = self.output_dir / "combined_annotated.parquet"

        if not force and cache.is_valid(combined_path, self.fingerprint):
            log.info("dedup: loading cached results")
            return pd.read_parquet(combined_path)

        # Combine all standardized dataframes, baseline sessions only for dedup
        all_dfs = []
        for ds, df in standardized.items():
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)

        # Initialize dedup columns
        combined["duplicate_group_id"] = None
        combined["dedup_method"] = "none"
        combined["dedup_confidence"] = "unique"

        # Work on baseline sessions only for candidate generation
        baseline = combined[combined["session_index"] == 0].copy()

        candidates = self._find_demographic_candidates(baseline)
        log.info("dedup: found %d demographic candidate pairs", len(candidates))

        decisions = []
        group_counter = 0
        # Track confirmed groups: key -> group_id
        key_to_group: dict[str, str] = {}

        # Batch hash lookups — only for pairs whose hashes are meaningful.
        hash_jobs: list[tuple[str, str]] = []
        for pair in candidates:
            if self.cfg.shares_pipeline(pair.dataset_a, pair.dataset_b):
                for pval, root, dt in [
                    (pair.seg_path_a,   pair.root_a, "seg"),
                    (pair.seg_path_b,   pair.root_b, "seg"),
                    (pair.t1ce_path_a,  pair.root_a, "t1ce"),
                    (pair.t1ce_path_b,  pair.root_b, "t1ce"),
                ]:
                    p = _str_or_none(pval)
                    if p:
                        hash_jobs.append((str(root / p), dt))

        # Compute hashes in parallel
        hash_cache: dict[tuple[str, str], Optional[str]] = {}
        if hash_jobs:
            unique_jobs = list(set(hash_jobs))
            log.info("dedup: computing %d hashes (workers=%d)", len(unique_jobs), self.workers)
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futs = {pool.submit(_hash_seg_worker, job): job for job in unique_jobs}
                for fut in as_completed(futs):
                    job = futs[fut]
                    try:
                        _, digest = fut.result()
                        hash_cache[job] = digest
                    except Exception as exc:
                        log.warning("hash failed for %s: %s", job[0], exc)
                        hash_cache[job] = None

        for pair in candidates:
            # A hash comparison is only evidence between cohorts that share an
            # annotation/intensity pipeline. Elsewhere both a match and a
            # mismatch are uninformative, so we neither hash nor rule out.
            is_brats_pair = self.cfg.shares_pipeline(pair.dataset_a, pair.dataset_b)

            seg_hash_a = seg_hash_b = t1ce_hash_a = t1ce_hash_b = None
            if is_brats_pair:
                sp_a = _str_or_none(pair.seg_path_a)
                sp_b = _str_or_none(pair.seg_path_b)
                tc_a = _str_or_none(pair.t1ce_path_a)
                tc_b = _str_or_none(pair.t1ce_path_b)
                if sp_a:
                    seg_hash_a = hash_cache.get((str(pair.root_a / sp_a), "seg"))
                if sp_b:
                    seg_hash_b = hash_cache.get((str(pair.root_b / sp_b), "seg"))
                if tc_a:
                    t1ce_hash_a = hash_cache.get((str(pair.root_a / tc_a), "t1ce"))
                if tc_b:
                    t1ce_hash_b = hash_cache.get((str(pair.root_b / tc_b), "t1ce"))

            # Determine verdict
            if not is_brats_pair:
                verdict = "UNVERIFIABLE"
                method = "demographic"
                confidence = "candidate"
            elif seg_hash_a and seg_hash_b and seg_hash_a == seg_hash_b:
                verdict = "CONFIRMED"
                method = "seg_hash"
                confidence = "confirmed"
            elif seg_hash_a and seg_hash_b and seg_hash_a != seg_hash_b:
                verdict = "RULED_OUT"
                method = "seg_hash"
                confidence = "unique"
            elif t1ce_hash_a and t1ce_hash_b and t1ce_hash_a == t1ce_hash_b:
                verdict = "CONFIRMED"
                method = "t1ce_hash"
                confidence = "confirmed"
            elif t1ce_hash_a and t1ce_hash_b and t1ce_hash_a != t1ce_hash_b:
                verdict = "RULED_OUT"
                method = "t1ce_hash"
                confidence = "unique"
            else:
                verdict = "UNVERIFIABLE"
                method = "demographic"
                confidence = "candidate"

            # Assign group IDs for confirmed pairs
            if verdict == "CONFIRMED":
                existing = key_to_group.get(pair.key_a) or key_to_group.get(pair.key_b)
                if existing is None:
                    group_counter += 1
                    existing = f"dup_group_{group_counter:04d}"
                key_to_group[pair.key_a] = existing
                key_to_group[pair.key_b] = existing

            decisions.append({
                "key_a": pair.key_a, "key_b": pair.key_b,
                "dataset_a": pair.dataset_a, "dataset_b": pair.dataset_b,
                "age_diff": pair.age_diff, "days_diff": pair.days_diff,
                "sublabel": pair.sublabel,
                "seg_hash_a": seg_hash_a, "seg_hash_b": seg_hash_b,
                "t1ce_hash_a": t1ce_hash_a, "t1ce_hash_b": t1ce_hash_b,
                "verdict": verdict, "dedup_method": method,
                "dedup_confidence": confidence,
            })

        decisions_df = pd.DataFrame(decisions)
        write_csv(decisions_df, decisions_path)
        log.info("dedup: wrote %d decision rows to %s", len(decisions_df), decisions_path)

        # Annotate the combined dataframe
        for gsk, gid in key_to_group.items():
            mask = combined["global_session_key"] == gsk
            combined.loc[mask, "duplicate_group_id"] = gid
            # Find the dedup_method for this key
            for d in decisions:
                if d["key_a"] == gsk or d["key_b"] == gsk:
                    if d["verdict"] == "CONFIRMED":
                        combined.loc[mask, "dedup_method"] = d["dedup_method"]
                        combined.loc[mask, "dedup_confidence"] = d["dedup_confidence"]
                        break

        # Also mark candidate (unverifiable) pairs
        for d in decisions:
            if d["verdict"] == "UNVERIFIABLE":
                for gsk in (d["key_a"], d["key_b"]):
                    mask = combined["global_session_key"] == gsk
                    if combined.loc[mask, "dedup_confidence"].iloc[0] == "unique":
                        combined.loc[mask, "dedup_method"] = d["dedup_method"]
                        combined.loc[mask, "dedup_confidence"] = d["dedup_confidence"]

        write_parquet(combined, combined_path)
        cache.record(combined_path, self.fingerprint)
        confirmed = sum(1 for d in decisions if d["verdict"] == "CONFIRMED")
        log.info("dedup: %d candidates, %d confirmed dup pairs, %d groups",
                 len(candidates), confirmed, group_counter)
        return combined

    def _find_demographic_candidates(self, baseline: pd.DataFrame) -> list[_Pair]:
        """O(N×M) cross-join on vital_status + age ± tol + os_days ± tol."""
        cfg = self.cfg
        # Only consider sessions with OS data for demographic matching
        has_os = baseline.dropna(subset=["os_days", "os_event", "age"])

        candidates: list[_Pair] = []
        rows = has_os.to_dict("records")
        n = len(rows)
        for i in range(n):
            a = rows[i]
            for j in range(i + 1, n):
                b = rows[j]
                # Skip same-dataset pairs
                if a["dataset"] == b["dataset"]:
                    continue
                # Vital status must match
                if a["os_event"] != b["os_event"]:
                    continue
                age_diff = abs(a["age"] - b["age"])
                days_diff = abs(a["os_days"] - b["os_days"])
                if age_diff > cfg.demographic_age_tol or days_diff > cfg.demographic_days_tol:
                    continue
                sublabel = (
                    "close"
                    if age_diff <= cfg.close_age_tol and days_diff <= cfg.close_days_tol
                    else "loose"
                )
                root_a = self.dataset_roots.get(a["dataset"], Path("/"))
                root_b = self.dataset_roots.get(b["dataset"], Path("/"))
                candidates.append(_Pair(
                    key_a=a["global_session_key"],
                    key_b=b["global_session_key"],
                    dataset_a=a["dataset"], dataset_b=b["dataset"],
                    age_diff=age_diff, days_diff=days_diff, sublabel=sublabel,
                    seg_path_a=_str_or_none(a.get("seg_path")),
                    seg_path_b=_str_or_none(b.get("seg_path")),
                    t1ce_path_a=_str_or_none(a.get("t1ce_path")),
                    t1ce_path_b=_str_or_none(b.get("t1ce_path")),
                    root_a=root_a, root_b=root_b,
                ))
        return candidates
