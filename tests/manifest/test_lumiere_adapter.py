"""LUMIERE adapter contract, exercised against the synthetic tree.

Runs without /mnt/disk1. The synthetic cohort encodes the three hazards that
distinguish LUMIERE from the other four datasets:

  Patient-001  two studies in the same week, one either side of surgery
  Patient-002  no preoperative study at all
  Patient-003  an unrated follow-up, and no recorded survival
"""
from __future__ import annotations

import pytest


def _rows(manifest, patient_id):
    df = manifest[(manifest["dataset"] == "lumiere")
                  & (manifest["patient_id"] == patient_id)]
    return df.sort_values("session_index").reset_index(drop=True)


class TestPreoperativeIsReadNotInferred:
    def test_same_week_studies_are_separated_by_the_rating(self, synthetic_manifest):
        """week-000-1 and week-000-2 differ only in the expert rating.

        Both sit in week 0, so any rule based on ordering or week number alone
        would label them identically.
        """
        rows = _rows(synthetic_manifest, "Patient-001")
        assert len(rows) == 2
        assert rows.loc[0, "acquisition_context"] == "preop"
        assert rows.loc[1, "acquisition_context"] == "postop"

    def test_patient_without_a_preoperative_study_has_none(self, synthetic_manifest):
        """Patient-002's earliest session is postoperative and stays that way.

        This is the case that made the criterion load-bearing: session_index is
        0, so baseline_only alone would admit it.
        """
        rows = _rows(synthetic_manifest, "Patient-002")
        assert len(rows) == 1
        assert rows.loc[0, "session_index"] == 0
        assert rows.loc[0, "acquisition_context"] == "postop"

    def test_unrated_session_after_a_known_preop_is_postoperative(self, synthetic_manifest):
        rows = _rows(synthetic_manifest, "Patient-003")
        assert rows.loc[0, "acquisition_context"] == "preop"     # week-000, rated
        assert rows.loc[1, "acquisition_context"] == "postop"    # week-012, unrated


class TestClinicalJoin:
    def test_survival_weeks_become_days(self, synthetic_manifest):
        rows = _rows(synthetic_manifest, "Patient-001")
        assert rows.loc[0, "os_days"] == 72 * 7

    def test_vital_status_is_never_invented(self, synthetic_manifest):
        """LUMIERE ships no event indicator, so os_event stays null.

        The evidence that these patients died is strong (analysis/), but
        applying it is a modelling decision and belongs in gbm_os.
        """
        rows = synthetic_manifest[synthetic_manifest["dataset"] == "lumiere"]
        assert rows["os_event"].isna().all()

    def test_missing_survival_stays_missing(self, synthetic_manifest):
        rows = _rows(synthetic_manifest, "Patient-003")
        assert rows["os_days"].isna().all()

    def test_negated_mgmt_is_not_inverted(self, synthetic_manifest):
        """'not methylated' contains 'meth' and used to normalise to methylated."""
        rows = _rows(synthetic_manifest, "Patient-001")
        assert rows.loc[0, "mgmt_methylation"] == "unmethylated"

    def test_variant_named_idh_is_read(self, synthetic_manifest):
        """LUMIERE writes the variant ('R132H mut'), not the status."""
        rows = _rows(synthetic_manifest, "Patient-002")
        assert rows.loc[0, "idh_status"] == "mutant"

    def test_grade_is_null_not_guessed(self, synthetic_manifest):
        """GBM by construction under WHO 2016, with no grade column.

        The same shape as UPENN, which the study's grade rule already admits.
        """
        rows = synthetic_manifest[synthetic_manifest["dataset"] == "lumiere"]
        assert rows["who_grade"].isna().all()

    def test_resection_extent_comes_from_the_rano_rationale(self, synthetic_manifest):
        """CRET is a complete resection, PRET a partial one."""
        assert _rows(synthetic_manifest, "Patient-001").loc[0, "eor"] == "GTR"
        assert _rows(synthetic_manifest, "Patient-002").loc[0, "eor"] == "STR"


class TestImaging:
    def test_all_four_structural_modalities_resolve(self, synthetic_manifest):
        rows = synthetic_manifest[synthetic_manifest["dataset"] == "lumiere"]
        for col in ("has_t1", "has_t1ce", "has_t2", "has_flair", "has_seg"):
            assert rows[col].all(), col

    def test_contrast_t1_is_mapped_from_ct1(self, synthetic_manifest):
        """LUMIERE names the post-contrast T1 for the agent, not the sequence."""
        rows = _rows(synthetic_manifest, "Patient-001")
        assert rows.loc[0, "t1ce_path"].endswith("ct1_skull_strip.nii")
        assert rows.loc[0, "t1_path"].endswith("t1_skull_strip.nii")

    def test_segmentation_convention_is_its_own(self, synthetic_manifest):
        """DeepBraTumIA is {0,1,2,3} but not the RHUH ordering — ET is 1, not 3.

        Tagging it 'rhuh' would leave enhancing tumour labelled as necrosis.
        """
        rows = synthetic_manifest[synthetic_manifest["dataset"] == "lumiere"]
        assert set(rows["seg_convention"].unique()) == {"lumiere"}


class TestSegRemap:
    def test_lumiere_labels_permute_to_brats(self):
        import numpy as np

        from gbm_os.transforms.seg import remap_seg

        seg = np.array([0, 1, 2, 3])
        out = remap_seg(seg, "lumiere")
        # 1 (ET) -> 4, 2 (necrosis) -> 1, 3 (edema) -> 2
        assert out.tolist() == [0, 4, 1, 2]

    def test_permutation_does_not_overwrite_itself(self):
        """1->4, 2->1 share the value 1; masks must read the original array."""
        import numpy as np

        from gbm_os.transforms.seg import remap_seg

        seg = np.array([1, 1, 2, 2, 3])
        assert remap_seg(seg, "lumiere").tolist() == [4, 4, 1, 1, 2]

    def test_rhuh_is_untouched_by_the_new_table(self):
        import numpy as np

        from gbm_os.transforms.seg import remap_seg

        assert remap_seg(np.array([0, 1, 2, 3]), "rhuh").tolist() == [0, 1, 2, 4]

    def test_unknown_convention_passes_through(self):
        import numpy as np

        from gbm_os.transforms.seg import remap_seg

        seg = np.array([0, 1, 2, 4])
        assert remap_seg(seg, "brats_legacy").tolist() == seg.tolist()
