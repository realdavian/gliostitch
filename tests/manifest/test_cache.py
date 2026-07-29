"""Stage caches must invalidate when the inputs that produced them change.

Testing only for file existence lets an artifact written by an older config or
an older schema be served indefinitely as if it were current — which is how
pre-revert cohort output survived a code change unnoticed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gbm_manifest.config import load_config
from gbm_manifest.infra import cache
from gbm_manifest.pipeline import Pipeline


def _write_cfg(path: Path, roots: dict, out: Path, **overrides) -> Path:
    body = {
        "datasets": {n: {"root": str(r), "enabled": True} for n, r in roots.items()},
        "output_dir": str(out),
        "workers": 2,
    }
    body.update(overrides)
    path.write_text(yaml.safe_dump(body))
    return path


class TestFingerprint:
    def test_same_inputs_same_digest(self):
        assert cache.fingerprint(a=1, b="x") == cache.fingerprint(b="x", a=1)

    def test_different_inputs_differ(self):
        assert cache.fingerprint(a=1) != cache.fingerprint(a=2)

    def test_missing_sidecar_is_stale(self, tmp_path):
        art = tmp_path / "thing.csv"
        art.write_text("data")
        assert not cache.is_valid(art, "abc")

    def test_roundtrip_is_valid(self, tmp_path):
        art = tmp_path / "thing.csv"
        art.write_text("data")
        cache.record(art, "abc")
        assert cache.is_valid(art, "abc")
        assert not cache.is_valid(art, "def")

    def test_corrupt_sidecar_is_stale(self, tmp_path):
        art = tmp_path / "thing.csv"
        art.write_text("data")
        cache.meta_path(art).write_text("{not json")
        assert not cache.is_valid(art, "abc")


class TestStageInvalidation:
    def test_changed_study_recomputes_cohort(self, synthetic_roots, tmp_path):
        """Switching study must invalidate the cached cohort and rewrite it.

        Asserted on the sidecar rather than on the CSV: two studies can select
        the same rows from a given input, so identical output does not prove
        the cache was consulted correctly — a refreshed fingerprint does.
        """
        out = tmp_path / "out"
        cfg_path = tmp_path / "a.yaml"
        sel = out / "cohort" / "selected.csv"

        _write_cfg(cfg_path, synthetic_roots, out, cohort={"study": "gbm-os"})
        Pipeline(load_config(cfg_path)).build(force=True)
        first = json.loads(cache.meta_path(sel).read_text())["fingerprint"]

        _write_cfg(cfg_path, synthetic_roots, out, cohort={"study": "gbm-os-any-eor"})
        pipeline = Pipeline(load_config(cfg_path))
        assert not cache.is_valid(sel, pipeline._fp_cohort()), \
            "stale cohort still looked valid after the study changed"

        pipeline.build(force=False)
        second = json.loads(cache.meta_path(sel).read_text())["fingerprint"]
        assert first != second, "cohort was served from a stale cache"

    def test_unchanged_config_reuses_cache(self, synthetic_roots, tmp_path):
        out = tmp_path / "out"
        cfg_path = _write_cfg(tmp_path / "a.yaml", synthetic_roots, out)

        Pipeline(load_config(cfg_path)).build(force=True)
        stamp = cache.meta_path(out / "master_manifest.csv").read_text()

        Pipeline(load_config(cfg_path)).build(force=False)
        assert cache.meta_path(out / "master_manifest.csv").read_text() == stamp

    def test_schema_bump_invalidates(self, synthetic_roots, tmp_path, monkeypatch):
        """An artifact from an older schema version is never reused."""
        out = tmp_path / "out"
        cfg_path = _write_cfg(tmp_path / "a.yaml", synthetic_roots, out)
        Pipeline(load_config(cfg_path)).build(force=True)

        recorded = json.loads(
            cache.meta_path(out / "master_manifest.csv").read_text())["fingerprint"]

        monkeypatch.setattr("gbm_manifest.pipeline.SCHEMA_VERSION", 999)
        assert Pipeline(load_config(cfg_path))._fp_manifest() != recorded

    def test_every_stage_writes_a_sidecar(self, synthetic_roots, tmp_path):
        out = tmp_path / "out"
        cfg_path = _write_cfg(tmp_path / "a.yaml", synthetic_roots, out)
        Pipeline(load_config(cfg_path)).build(force=True)

        for artifact in (
            out / "raw_inventory" / "brats2020.parquet",
            out / "standardized" / "brats2020.parquet",
            out / "dedup" / "combined_annotated.parquet",
            out / "master_manifest.csv",
            out / "cohort" / "selected.csv",
        ):
            assert cache.meta_path(artifact).exists(), f"no sidecar for {artifact.name}"


class TestConfigBootstrap:
    """A stranger installing from PyPI has no config/pipeline.yaml.

    It lives in the repo, not the package, so without a way to generate one the
    pipeline is unusable outside a git checkout — and the failure was a raw
    FileNotFoundError traceback.
    """

    def test_missing_config_explains_how_to_make_one(self, tmp_path):
        from gbm_manifest.config import load_config

        with pytest.raises(FileNotFoundError, match="gliostitch init"):
            load_config(tmp_path / "nope.yaml")

    def test_template_is_valid_and_loadable(self, tmp_path):
        from gbm_manifest.config import load_config, write_template

        path = tmp_path / "config" / "pipeline.yaml"
        write_template(path)
        cfg = load_config(path)

        assert set(cfg.datasets) == {"brats2020", "rhuh_gbm", "upenn_gbm", "ucsf_pdgm"}
        assert cfg.cohort.study == "gbm-os"
        assert cfg.dedup.shares_pipeline("brats2020", "upenn_gbm")
        assert not cfg.dedup.shares_pipeline("brats2020", "ucsf_pdgm")

    def test_template_placeholders_are_obvious(self, tmp_path):
        from gbm_manifest.config import write_template

        path = tmp_path / "pipeline.yaml"
        write_template(path)
        assert "/path/to/" in path.read_text(), "placeholder roots must look unset"
