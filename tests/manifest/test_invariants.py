"""Pipeline-level invariants: dedup evidence scoping and reproducibility.

Spec 01 §9 — a segmentation or intensity hash is only evidence between cohorts
that share an annotation/intensity pipeline. Between independently annotated
cohorts a mismatch proves nothing, so it must not downgrade a demographic
candidate to 'unique'.

Spec 01 §2.4 / §12 M6 — same inputs must yield a byte-identical manifest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def decisions(synthetic_config_path, synthetic_manifest) -> pd.DataFrame:
    from gbm_manifest.config import load_config

    cfg = load_config(synthetic_config_path)
    return pd.read_csv(cfg.output_dir / "dedup" / "decisions.csv")


class TestDedupEvidenceScope:
    def test_same_pipeline_match_is_confirmed(self, decisions):
        """BraTS_001 and UPENN-GBM-00002_11 share a segmentation byte-for-byte."""
        pairs = decisions[decisions["verdict"] == "CONFIRMED"]
        assert len(pairs) == 1
        got = {pairs.iloc[0]["dataset_a"], pairs.iloc[0]["dataset_b"]}
        assert got == {"brats2020", "upenn_gbm"}

    def test_cross_pipeline_mismatch_is_not_evidence(self, decisions):
        """BraTS_002 vs UCSF-004: same demographics, independent annotation.

        The segmentations differ because the cohorts were annotated separately,
        which says nothing about whether it is the same person. The pair must
        survive as a candidate rather than being ruled out.
        """
        m = decisions[
            (decisions["dataset_a"].isin(["brats2020", "ucsf_pdgm"]))
            & (decisions["dataset_b"].isin(["brats2020", "ucsf_pdgm"]))
            & (decisions["dataset_a"] != decisions["dataset_b"])
        ]
        assert len(m) == 1, "expected one brats<->ucsf demographic candidate"
        assert m.iloc[0]["verdict"] != "RULED_OUT"
        assert m.iloc[0]["dedup_confidence"] == "candidate"

    def test_confirmed_pair_shares_a_group_id(self, synthetic_manifest):
        grouped = synthetic_manifest[synthetic_manifest["duplicate_group_id"].notna()]
        assert len(grouped) == 2
        assert grouped["duplicate_group_id"].nunique() == 1
        assert set(grouped["dataset"]) == {"brats2020", "upenn_gbm"}

    def test_duplicates_are_annotated_never_deleted(self, synthetic_manifest):
        """Spec 01 §2.3 — both members stay in the manifest."""
        assert len(synthetic_manifest) == 10


class TestReproducibility:
    def test_two_cold_builds_are_byte_identical(self, synthetic_roots, tmp_path):
        """Spec 01 §12 M6 — same inputs, byte-identical outputs."""
        import yaml

        from gbm_manifest.config import load_config
        from gbm_manifest.pipeline import Pipeline

        digests = []
        for run in ("run_a", "run_b"):
            out = tmp_path / run
            cfg_path = tmp_path / f"{run}.yaml"
            cfg_path.write_text(yaml.safe_dump({
                "datasets": {n: {"root": str(r), "enabled": True}
                             for n, r in synthetic_roots.items()},
                "output_dir": str(out),
                "workers": 2,
            }))
            Pipeline(load_config(cfg_path)).build(force=True)
            digests.append({
                name: _digest(out / name)
                for name in ("master_manifest.csv",
                             "cohort/selected.csv",
                             "cohort/exclusions.csv",
                             "dedup/decisions.csv")
            })

        assert digests[0] == digests[1]

    def test_manifest_column_contract(self, synthetic_manifest):
        from gbm_manifest.core.schema import MANIFEST_COLUMNS

        assert list(synthetic_manifest.columns) == list(MANIFEST_COLUMNS)


class TestStudyChoiceDoesNotReachTheManifest:
    """Which study is configured must not change a single manifest row.

    The manifest is the facts superset; a study is one interpretation of it.
    If a study could alter the manifest, the facts would depend on the question
    being asked and no other study could be run against the same file.
    """

    def _build(self, roots, out, study):
        import yaml

        from gbm_manifest.config import load_config
        from gbm_manifest.pipeline import Pipeline

        cfg_path = out.parent / f"{study}.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "datasets": {n: {"root": str(r), "enabled": True}
                         for n, r in roots.items()},
            "output_dir": str(out),
            "workers": 2,
            "cohort": {"study": study},
        }))
        Pipeline(load_config(cfg_path)).build(force=True)
        return out

    def test_manifest_is_byte_identical_across_studies(self, synthetic_roots, tmp_path):
        a = self._build(synthetic_roots, tmp_path / "a", "gbm-os")
        b = self._build(synthetic_roots, tmp_path / "b", "gbm-os-m6")

        assert _digest(a / "master_manifest.csv") == _digest(b / "master_manifest.csv")

    def test_only_the_derived_cohort_differs(self, synthetic_roots, tmp_path):
        """The study must still be doing something — just downstream of the facts."""
        import pandas as pd

        a = self._build(synthetic_roots, tmp_path / "a", "gbm-os")
        b = self._build(synthetic_roots, tmp_path / "b", "gbm-os-m6")

        rows_a = len(pd.read_csv(a / "master_manifest.csv"))
        rows_b = len(pd.read_csv(b / "master_manifest.csv"))
        assert rows_a == rows_b

        # Both studies write a cohort; they are free to disagree about it.
        assert (a / "cohort" / "selected.csv").exists()
        assert (b / "cohort" / "selected.csv").exists()
