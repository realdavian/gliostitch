"""Unit tests for gbm_os.transforms — remap_seg, SegRemapd, foreground_zscore."""
from __future__ import annotations

import numpy as np
import pytest

from gbm_os.transforms.seg import SegRemapd, remap_seg
from gbm_os.transforms.intensity import foreground_zscore


class TestRemapSeg:
    def test_rhuh_remaps_3_to_4(self):
        seg = np.array([0, 1, 2, 3], dtype=np.int32)
        out = remap_seg(seg, "rhuh")
        assert list(out) == [0, 1, 2, 4]

    def test_rhuh_leaves_0_1_2_unchanged(self):
        seg = np.array([0, 1, 2], dtype=np.int32)
        out = remap_seg(seg, "rhuh")
        assert list(out) == [0, 1, 2]

    def test_brats_legacy_passthrough(self):
        seg = np.array([0, 1, 2, 4], dtype=np.int32)
        out = remap_seg(seg, "brats_legacy")
        np.testing.assert_array_equal(out, seg)

    def test_unknown_convention_passthrough(self):
        seg = np.array([0, 1, 2, 3], dtype=np.int32)
        out = remap_seg(seg, "unknown_convention")
        np.testing.assert_array_equal(out, seg)

    def test_does_not_mutate_input(self):
        seg = np.array([0, 1, 2, 3], dtype=np.int32)
        original = seg.copy()
        remap_seg(seg, "rhuh")
        np.testing.assert_array_equal(seg, original)

    def test_3d_volume(self):
        seg = np.zeros((5, 5, 5), dtype=np.int32)
        seg[0, 0, 0] = 3
        out = remap_seg(seg, "rhuh")
        assert out[0, 0, 0] == 4
        assert out[1, 1, 1] == 0


class TestSegRemapd:
    def test_passthrough_no_seg_key(self):
        r = SegRemapd()
        d = {"other": 1}
        assert r(d) == {"other": 1}

    def test_passthrough_no_convention_key(self):
        r = SegRemapd()
        seg = np.array([0, 1, 2, 3], dtype=np.int32)
        d = {"seg": seg}
        result = r(d)
        np.testing.assert_array_equal(result["seg"], seg)

    def test_passthrough_both_absent(self):
        r = SegRemapd()
        assert r({}) == {}

    def test_custom_keys(self):
        r = SegRemapd(seg_key="label", convention_key="conv")
        d = {"label": None, "conv": "rhuh"}
        result = r(d)
        assert result["label"] is None  # None seg → passthrough

    def test_non_mutating_on_input_dict(self):
        r = SegRemapd()
        original = {"other": 42}
        result = r(original)
        assert result is not original  # new dict returned


class TestForegroundZscore:
    def test_normalises_nonzero_voxels(self):
        vol = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        out = foreground_zscore(vol)
        fg = out[out > 0]  # after normalisation foreground shifts
        # result should have near-zero mean over foreground of original
        fg_orig_mask = vol > 0.0
        assert abs(out[fg_orig_mask].mean()) < 1e-5

    def test_prenormalised_passthrough(self):
        vol = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = foreground_zscore(vol, intensity_prenormalised=True)
        np.testing.assert_array_equal(out, vol)

    def test_all_background_passthrough(self):
        vol = np.zeros((5,), dtype=np.float32)
        out = foreground_zscore(vol)
        np.testing.assert_array_equal(out, vol)

    def test_constant_foreground_passthrough(self):
        vol = np.array([0.0, 5.0, 5.0, 5.0], dtype=np.float32)
        out = foreground_zscore(vol)
        np.testing.assert_array_equal(out, vol)
