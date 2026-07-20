"""P3 — duplicate resolution policies."""
from __future__ import annotations

import pytest


# The 4 confirmed BraTS↔UPENN duplicate pairs
DUP_GROUPS = ["dup_group_0001", "dup_group_0002", "dup_group_0003", "dup_group_0004"]


@pytest.fixture(scope="module")
def dup_rows(cohort):
    """All manifest rows belonging to any dup group."""
    view = cohort.select(resolve_duplicates="keep")
    df = view.to_frame()
    return df[df["duplicate_group_id"].notna()]


class TestDupGroupStructure:
    def test_four_groups_exist(self, dup_rows):
        assert set(dup_rows["duplicate_group_id"].unique()) == set(DUP_GROUPS)

    def test_brats_and_upenn_in_each_group(self, dup_rows):
        for gid in DUP_GROUPS:
            members = dup_rows[dup_rows["duplicate_group_id"] == gid]
            datasets = set(members["dataset"].unique())
            assert "brats2020" in datasets, f"{gid} missing brats2020"
            assert "upenn_gbm" in datasets, f"{gid} missing upenn_gbm"


class TestPolicyDrop:
    """resolve_duplicates='drop': external (UPENN) loses to train (BraTS)."""

    def test_upenn_removed_from_conflict_groups(self, cohort):
        view = cohort.select(resolve_duplicates="drop")
        df = view.to_frame()
        dup_in_view = df[df["duplicate_group_id"].notna()]
        # Only brats2020 rows should remain in dup groups
        assert (dup_in_view["dataset"] == "brats2020").all()

    def test_total_dup_rows_reduced(self, cohort, dup_rows):
        view_drop = cohort.select(resolve_duplicates="drop")
        df_drop = view_drop.to_frame()
        remaining_dup = df_drop[df_drop["duplicate_group_id"].notna()]
        original_dup = dup_rows
        assert len(remaining_dup) < len(original_dup)

    def test_brats_rows_preserved(self, cohort):
        view = cohort.select(resolve_duplicates="drop")
        df = view.to_frame()
        dup_in_view = df[df["duplicate_group_id"].notna()]
        assert len(dup_in_view) == 4  # 4 brats2020 rows kept


class TestPolicyFlag:
    """resolve_duplicates='flag': both members kept."""

    def test_both_members_present(self, cohort, dup_rows):
        view = cohort.select(resolve_duplicates="flag")
        df = view.to_frame()
        dup_in_view = df[df["duplicate_group_id"].notna()]
        assert len(dup_in_view) == len(dup_rows)

    def test_group_count_unchanged(self, cohort):
        view = cohort.select(resolve_duplicates="flag")
        df = view.to_frame()
        assert set(df[df["duplicate_group_id"].notna()]["duplicate_group_id"].unique()) == set(DUP_GROUPS)


class TestPolicyKeep:
    """resolve_duplicates='keep': same as flag for dup rows, no marking."""

    def test_all_rows_present(self, cohort, dup_rows):
        view = cohort.select(resolve_duplicates="keep")
        df = view.to_frame()
        dup_in_view = df[df["duplicate_group_id"].notna()]
        assert len(dup_in_view) == len(dup_rows)
