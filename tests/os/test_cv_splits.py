"""P5 — CV splitting: patient-grouped, stratified, deterministic."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def fold_collection(cohort):
    view = cohort.select(
        baseline_only=True,
        require_complete=True,
        filters={"has_os": True},
    )
    return view.split(k=5, seed=42)


class TestSplitDeterminism:
    def test_same_seed_same_folds(self, cohort):
        view = cohort.select(baseline_only=True, require_complete=True,
                             filters={"has_os": True})
        fc1 = view.split(k=5, seed=42)
        fc2 = view.split(k=5, seed=42)
        a1 = fc1.assignments()
        a2 = fc2.assignments()
        assert (a1 == a2).all()

    def test_different_seed_different_folds(self, cohort):
        view = cohort.select(baseline_only=True, require_complete=True,
                             filters={"has_os": True})
        fc1 = view.split(k=5, seed=42)
        fc2 = view.split(k=5, seed=99)
        assert not (fc1.assignments() == fc2.assignments()).all()


class TestPatientGrouping:
    def test_no_patient_in_two_folds(self, cohort):
        """All rows for a given patient must be in the same fold."""
        view = cohort.select(filters={"has_os": True})  # include all sessions
        fc = view.split(k=5, seed=42)
        df = view.to_frame()
        df = df.assign(_fold=fc.assignments().values)

        for (dataset, patient_id), grp in df.groupby(["dataset", "patient_id"]):
            folds = grp["_fold"].unique()
            assert len(folds) == 1, (
                f"Patient {dataset}/{patient_id} appears in folds {folds}"
            )


class TestFoldBalance:
    def test_k_folds_exist(self, fold_collection):
        assert fold_collection.k == 5

    def test_fold_sizes_roughly_equal(self, fold_collection):
        sizes = [len(fold_collection.fold(i, "val")) for i in range(5)]
        assert max(sizes) - min(sizes) <= 15, f"Imbalanced folds: {sizes}"

    def test_train_val_partition_complete(self, fold_collection, cohort):
        """train + val for each fold should reconstruct the full selection."""
        view = cohort.select(baseline_only=True, require_complete=True,
                             filters={"has_os": True})
        total = len(view)
        for i in range(5):
            train_n = len(fold_collection.fold(i, "train"))
            val_n = len(fold_collection.fold(i, "val"))
            assert train_n + val_n == total, f"fold {i}: {train_n}+{val_n} != {total}"


class TestStratification:
    def test_all_classes_in_train_folds(self, fold_collection):
        """Each training fold should contain all os_class values."""
        for i in range(5):
            train_df = fold_collection.fold(i, "train").to_frame()
            classes = set(train_df["os_class"].dropna().unique())
            assert len(classes) >= 2, f"fold {i} train missing classes: {classes}"
