"""Unit tests for value normalisation. No dataset access required."""
import pytest

from gbm_manifest.adapters.normalize import (
    days_from_weeks,
    event_from_censored_flag,
    event_from_int,
    event_from_status,
    normalize_eor_binary,
    normalize_eor_categorical,
    normalize_grade,
    normalize_idh,
    normalize_mgmt,
    parse_survival_days,
    raw_str,
    to_float,
)
from gbm_manifest.core.schema import EORCategory


# --- parse_survival_days (BraTS dual-encode) ---------------------------------
class TestParseSurvivalDays:
    def test_integer_death(self):
        days, event = parse_survival_days("289")
        assert days == 289.0 and event == 1

    def test_alive_string(self):
        days, event = parse_survival_days("ALIVE (361 days later)")
        assert days == 361.0 and event == 0

    def test_blank(self):
        assert parse_survival_days("") == (None, None)

    def test_nan(self):
        import math
        assert parse_survival_days(float("nan")) == (None, None)

    def test_none(self):
        assert parse_survival_days(None) == (None, None)


# --- event_from_status (UPENN) -----------------------------------------------
class TestEventFromStatus:
    def test_deceased(self):
        assert event_from_status("Deceased") == 1

    def test_deceased_uncertain(self):
        assert event_from_status("Deceased - uncertain date") == 1

    def test_alive(self):
        assert event_from_status("Alive") == 0

    def test_lost(self):
        assert event_from_status("Lost to Follow-up") is None

    def test_blank(self):
        assert event_from_status("") is None


# --- event_from_censored_flag (RHUH) -----------------------------------------
class TestEventFromCensoredFlag:
    def test_no_means_event(self):
        assert event_from_censored_flag("no") == 1

    def test_yes_means_censored(self):
        assert event_from_censored_flag("yes") == 0

    def test_blank(self):
        assert event_from_censored_flag("") is None

    def test_none(self):
        assert event_from_censored_flag(None) is None


# --- event_from_int (UCSF) ---------------------------------------------------
class TestEventFromInt:
    def test_dead(self):
        assert event_from_int(1) == 1

    def test_alive(self):
        assert event_from_int(0) == 0

    def test_float_string(self):
        assert event_from_int("1.0") == 1

    def test_blank(self):
        assert event_from_int("") is None


# --- normalize_grade ---------------------------------------------------------
class TestNormalizeGrade:
    def test_hgg_to_4(self):
        assert normalize_grade("HGG") == 4

    def test_lgg_to_2(self):
        assert normalize_grade("LGG") == 2

    def test_numeric_4(self):
        assert normalize_grade(4) == 4

    def test_numeric_string(self):
        assert normalize_grade("3") == 3

    def test_roman_iv(self):
        assert normalize_grade("IV") == 4

    def test_roman_ii(self):
        assert normalize_grade("ii") == 2

    def test_blank(self):
        assert normalize_grade("") is None

    def test_nan(self):
        import math
        assert normalize_grade(float("nan")) is None


# --- normalize_eor_categorical -----------------------------------------------
class TestNormalizeEorCategorical:
    def test_gtr(self):
        assert normalize_eor_categorical("GTR") == EORCategory.GTR

    def test_gross_total(self):
        assert normalize_eor_categorical("Gross Total Resection") == EORCategory.GTR

    def test_str(self):
        assert normalize_eor_categorical("STR") == EORCategory.STR

    def test_biopsy(self):
        assert normalize_eor_categorical("Biopsy") == EORCategory.BIOPSY

    def test_unknown(self):
        assert normalize_eor_categorical("") == EORCategory.UNKNOWN

    def test_ntr(self):
        """RHUH records near-total resection as a bare NTR."""
        assert normalize_eor_categorical("NTR") == EORCategory.NTR

    def test_near_total_spelled_out(self):
        assert normalize_eor_categorical("Near Total Resection") == EORCategory.NTR
        assert normalize_eor_categorical("near-total") == EORCategory.NTR

    def test_ntr_not_matched_inside_a_word(self):
        """'ntr' is a substring of ordinary words — it must not win on one."""
        assert normalize_eor_categorical("contrast-enhancing residual") != EORCategory.NTR

    def test_ntr_does_not_shadow_gtr_or_str(self):
        assert normalize_eor_categorical("GTR") == EORCategory.GTR
        assert normalize_eor_categorical("Subtotal") == EORCategory.STR


# --- normalize_eor_binary (UPENN) --------------------------------------------
class TestNormalizeEorBinary:
    def test_y_to_gtr(self):
        assert normalize_eor_binary("Y") == EORCategory.GTR

    def test_n_to_non_gtr(self):
        assert normalize_eor_binary("N") == EORCategory.NON_GTR

    def test_blank(self):
        assert normalize_eor_binary("") == EORCategory.UNKNOWN


# --- normalize_mgmt ----------------------------------------------------------
class TestNormalizeMgmt:
    def test_methylated(self):
        assert normalize_mgmt("Methylated") == "methylated"

    def test_unmethylated(self):
        assert normalize_mgmt("Unmethylated") == "unmethylated"

    def test_unmeth_before_meth(self):
        # 'unmethylated' contains 'meth' — must match 'unmeth' first
        assert normalize_mgmt("unmethylated") == "unmethylated"

    def test_ucsf_positive(self):
        assert normalize_mgmt("positive") == "methylated"

    def test_ucsf_negative(self):
        assert normalize_mgmt("negative") == "unmethylated"

    def test_blank(self):
        assert normalize_mgmt("") is None

    def test_indeterminate(self):
        assert normalize_mgmt("indeterminate") is None

    def test_not_methylated_is_unmethylated(self):
        """LUMIERE spells it 'not methylated' — which contains 'meth'.

        Matching the positive token first inverts the label, so this guards a
        sign flip rather than a missing value.
        """
        assert normalize_mgmt("not methylated") == "unmethylated"

    def test_other_negated_spellings(self):
        for raw in ("non-methylated", "non methylated", "un-methylated", "no methylation"):
            assert normalize_mgmt(raw) == "unmethylated", raw


# --- normalize_idh -----------------------------------------------------------
class TestNormalizeIdh:
    def test_wildtype(self):
        assert normalize_idh("WT") == "wildtype"

    def test_wildtype_long(self):
        assert normalize_idh("IDH wildtype") == "wildtype"

    def test_mutant(self):
        assert normalize_idh("Mutant") == "mutant"

    def test_blank(self):
        assert normalize_idh("") is None

    def test_variant_named_instead_of_status(self):
        """LUMIERE writes 'R132H mut' — the token is not at the front."""
        assert normalize_idh("R132H mut") == "mutant"

    def test_inconclusive_assay_is_missing_not_wildtype(self):
        """'IDH1 neg, Sequencing required' reports an unfinished test.

        Reading it as wildtype would invent a molecular result.
        """
        assert normalize_idh("IDH1 neg, Sequencing required") is None


# --- days_from_weeks ---------------------------------------------------------
class TestDaysFromWeeks:
    def test_converts(self):
        assert days_from_weeks(72) == 504.0

    def test_blank(self):
        assert days_from_weeks("") is None
        assert days_from_weeks("na") is None
