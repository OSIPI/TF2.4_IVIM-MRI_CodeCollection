"""
Unit tests for body-part aware IVIM initial guesses (Feature #87).

Tests the lookup table in ivim_body_part_defaults.py and its integration
with OsipiBase.__init__().
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on the path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from src.wrappers.ivim_body_part_defaults import (
    IVIM_BODY_PART_DEFAULTS,
    get_body_part_defaults,
    get_available_body_parts,
)
from src.wrappers.OsipiBase import OsipiBase


# ---------------------------------------------------------------------------
# Tests for the standalone lookup table module
# ---------------------------------------------------------------------------


class TestBodyPartDefaults:
    """Tests for get_body_part_defaults() and the lookup table."""

    def test_brain_initial_guess(self):
        """Brain defaults should return literature-sourced values."""
        bp = get_body_part_defaults("brain")
        assert bp["initial_guess"]["f"] == 0.05
        assert bp["initial_guess"]["D"] == 0.0008
        assert bp["initial_guess"]["Dp"] == 0.01

    def test_liver_initial_guess(self):
        """Liver defaults should return literature-sourced values."""
        bp = get_body_part_defaults("liver")
        assert bp["initial_guess"]["f"] == 0.12
        assert bp["initial_guess"]["D"] == 0.001
        assert bp["initial_guess"]["Dp"] == 0.06

    def test_kidney_initial_guess(self):
        """Kidney defaults should return literature-sourced values."""
        bp = get_body_part_defaults("kidney")
        assert bp["initial_guess"]["f"] == 0.20
        assert bp["initial_guess"]["D"] == 0.0019
        assert bp["initial_guess"]["Dp"] == 0.03

    def test_liver_bounds_differ_from_generic(self):
        """Liver bounds should be tighter than generic bounds."""
        liver = get_body_part_defaults("liver")
        generic = get_body_part_defaults("generic")
        # Liver D upper bound should be tighter than generic
        assert liver["bounds"]["D"][1] < generic["bounds"]["D"][1]
        # Liver Dp lower bound should be higher than generic
        assert liver["bounds"]["Dp"][0] >= generic["bounds"]["Dp"][0]

    def test_unknown_body_part_raises_valueerror(self):
        """Unknown body part should raise ValueError with available list."""
        with pytest.raises(ValueError, match="Unknown body part"):
            get_body_part_defaults("elbow")

    def test_case_insensitivity(self):
        """Body part lookup should be case-insensitive."""
        lower = get_body_part_defaults("brain")
        upper = get_body_part_defaults("Brain")
        mixed = get_body_part_defaults("BRAIN")
        assert lower == upper == mixed

    def test_spaces_and_hyphens_normalized(self):
        """Spaces and hyphens should be normalized to underscores."""
        bp1 = get_body_part_defaults("head_and_neck")
        bp2 = get_body_part_defaults("head and neck")
        bp3 = get_body_part_defaults("head-and-neck")
        assert bp1 == bp2 == bp3

    def test_all_body_parts_have_required_keys(self):
        """Every body part entry should have initial_guess, bounds, and thresholds."""
        for name, defaults in IVIM_BODY_PART_DEFAULTS.items():
            assert "initial_guess" in defaults, f"{name} missing initial_guess"
            assert "bounds" in defaults, f"{name} missing bounds"
            assert "thresholds" in defaults, f"{name} missing thresholds"

    def test_all_body_parts_have_valid_parameter_keys(self):
        """Every initial_guess and bounds should have S0, f, Dp, D."""
        required_params = {"S0", "f", "Dp", "D"}
        for name, defaults in IVIM_BODY_PART_DEFAULTS.items():
            ig_keys = set(defaults["initial_guess"].keys())
            assert ig_keys == required_params, f"{name} initial_guess keys: {ig_keys}"
            bounds_keys = set(defaults["bounds"].keys())
            assert bounds_keys == required_params, f"{name} bounds keys: {bounds_keys}"

    def test_all_initial_guesses_within_bounds(self):
        """Every initial guess value should fall within its corresponding bounds."""
        for name, defaults in IVIM_BODY_PART_DEFAULTS.items():
            ig = defaults["initial_guess"]
            bounds = defaults["bounds"]
            for param in ["S0", "f", "Dp", "D"]:
                lo, hi = bounds[param]
                val = ig[param]
                assert lo <= val <= hi, (
                    f"{name}: {param}={val} outside bounds [{lo}, {hi}]"
                )

    def test_get_available_body_parts(self):
        """get_available_body_parts() should return a sorted list."""
        available = get_available_body_parts()
        assert isinstance(available, list)
        assert available == sorted(available)
        assert "brain" in available
        assert "liver" in available
        assert "generic" in available

    def test_generic_matches_original_defaults(self):
        """The 'generic' entry should match the original OsipiBase defaults."""
        generic = get_body_part_defaults("generic")
        assert generic["initial_guess"]["S0"] == 1.0
        assert generic["initial_guess"]["f"] == 0.1
        assert generic["initial_guess"]["Dp"] == 0.01
        assert generic["initial_guess"]["D"] == 0.001
        assert generic["bounds"]["D"] == [0, 0.005]
        assert generic["bounds"]["f"] == [0, 1.0]


# ---------------------------------------------------------------------------
# Tests for OsipiBase integration
# ---------------------------------------------------------------------------


class TestOsipiBaseBodyPart:
    """Tests for body_part integration in OsipiBase.__init__()."""

    def test_body_part_sets_initial_guess(self):
        """body_part='brain' should set brain-specific initial guess."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800], body_part="brain")
        assert fit.initial_guess["f"] == 0.05
        assert fit.initial_guess["D"] == 0.0008

    def test_body_part_sets_bounds(self):
        """body_part='liver' should set liver-specific bounds."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800], body_part="liver")
        assert fit.bounds["Dp"] == [0.01, 0.15]
        assert fit.bounds["D"] == [0.0003, 0.003]

    def test_body_part_none_uses_generic(self):
        """body_part=None (default) should use original generic defaults."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800])
        assert fit.initial_guess["f"] == 0.1
        assert fit.initial_guess["Dp"] == 0.01
        assert fit.initial_guess["D"] == 0.001

    def test_user_initial_guess_overrides_body_part(self):
        """Explicit initial_guess dict should take priority over body_part."""
        custom = {"S0": 1.0, "f": 0.99, "Dp": 0.05, "D": 0.002}
        fit = OsipiBase(
            bvalues=[0, 50, 200, 800],
            body_part="brain",
            initial_guess=custom,
        )
        assert fit.initial_guess["f"] == 0.99  # User value, not brain default
        # But bounds should still be brain-specific
        assert fit.bounds["D"] == [0.0003, 0.002]

    def test_user_bounds_overrides_body_part(self):
        """Explicit bounds dict should take priority over body_part."""
        custom_bounds = {"S0": [0.5, 1.5], "f": [0, 0.5], "Dp": [0, 0.1], "D": [0, 0.01]}
        fit = OsipiBase(
            bvalues=[0, 50, 200, 800],
            body_part="liver",
            bounds=custom_bounds,
        )
        assert fit.bounds["D"] == [0, 0.01]  # User value, not liver default
        # But initial_guess should still be liver-specific
        assert fit.initial_guess["f"] == 0.12

    def test_initial_guess_as_string(self):
        """Passing initial_guess='liver' should work like body_part='liver'."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800], initial_guess="liver")
        assert fit.initial_guess["f"] == 0.12
        assert fit.initial_guess["Dp"] == 0.06
        assert fit.bounds["D"] == [0.0003, 0.003]

    def test_body_part_stored_as_attribute(self):
        """The body_part should be stored on the instance for reference."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800], body_part="kidney")
        assert fit.body_part == "kidney"

    def test_body_part_none_attribute(self):
        """Default body_part should be None."""
        fit = OsipiBase(bvalues=[0, 50, 200, 800])
        assert fit.body_part is None

    def test_unknown_body_part_raises(self):
        """Unknown body part in OsipiBase should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown body part"):
            OsipiBase(bvalues=[0, 50, 200, 800], body_part="elbow")

    def test_all_body_parts_create_valid_instance(self):
        """Every body part should create a valid OsipiBase instance."""
        for bp_name in get_available_body_parts():
            fit = OsipiBase(bvalues=[0, 50, 200, 800], body_part=bp_name)
            assert fit.initial_guess is not None
            assert fit.bounds is not None
            assert fit.thresholds is not None
