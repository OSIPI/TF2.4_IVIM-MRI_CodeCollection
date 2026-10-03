import warnings

import numpy as np
import pytest

from src.wrappers.OsipiBase import OsipiBase

BVALUES = np.array([0, 10, 20, 50, 100, 200, 400, 800])
VALID_BOUNDS = {"S0": [0.7, 1.3], "f": [0, 1.0], "Dp": [0.005, 0.2], "D": [0, 0.005]}
VALID_INITIAL_GUESS = {"S0": 1, "f": 0.1, "Dp": 0.01, "D": 0.001}


def _bounds(**changes):
    bounds = {param: list(limits) for param, limits in VALID_BOUNDS.items()}
    bounds.update(changes)
    return bounds


def _initial_guess(**changes):
    initial_guess = dict(VALID_INITIAL_GUESS)
    initial_guess.update(changes)
    return initial_guess


def _check_warnings(**kwargs):
    """Return the messages of the warnings raised by the bounds/initial guess check."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OsipiBase(bvalues=BVALUES, **kwargs)
    return [str(w.message) for w in caught if "unrealistic" in str(w.message)]


def test_valid_settings_do_not_warn():
    assert _check_warnings(bounds=_bounds(), initial_guess=_initial_guess()) == []


@pytest.mark.parametrize("body_part", ["brain", "liver", "kidney"])
def test_body_part_defaults_do_not_warn(body_part):
    assert _check_warnings(body_part=body_part) == []


def test_generic_defaults_do_not_warn():
    assert _check_warnings() == []


@pytest.mark.parametrize("bounds, expected", [
    (_bounds(f=[0.6, 0.4]), "lower bound of f (0.6) is not below its upper bound (0.4)"),
    (_bounds(D=[0, 5]), "upper bound of D (5) is outside the realistic range [0, 0.01]"),
    (_bounds(Dp=[0.0001, 0.2]), "lower bound of Dp (0.0001) is outside the realistic range [0.003, 0.05]"),
    (_bounds(f=[-0.1, 1]), "lower bound of f (-0.1) is outside the realistic range [0, 1]"),
    (_bounds(S0=[0.7, 5000]), "upper bound of S0 (5000) is outside the realistic range [1, 1000]"),
    (_bounds(D=[0, 0.01], Dp=[0.005, 0.2]), "upper bound of D (0.01) is above the lower bound of Dp (0.005)"),
])
def test_unrealistic_bounds_warn(bounds, expected):
    messages = _check_warnings(bounds=bounds, initial_guess=_initial_guess())
    assert any(expected in message for message in messages), messages


@pytest.mark.parametrize("initial_guess, expected", [
    (_initial_guess(f=1.5), "initial guess of f (1.5) is outside the realistic range [0, 1]"),
    # D given in um2/ms instead of mm2/s
    (_initial_guess(D=1.0), "initial guess of D (1.0) is outside the realistic range [0, 0.01]"),
    (_initial_guess(Dp=0.3), "initial guess of Dp (0.3) is outside its bounds [0.005, 0.2]"),
])
def test_unrealistic_initial_guess_warns(initial_guess, expected):
    messages = _check_warnings(bounds=_bounds(), initial_guess=initial_guess)
    assert any(expected in message for message in messages), messages


def test_partial_bounds_are_checked():
    messages = _check_warnings(bounds={"D": [0, 5]}, initial_guess={"D": 0.001})
    assert any("upper bound of D (5)" in message for message in messages), messages


def test_warns_once_when_initiating_algorithm():
    messages = _check_warnings(algorithm="IAR_LU_biexp", bounds=_bounds(D=[0, 5]), initial_guess=_initial_guess())
    assert len([m for m in messages if "upper bound of D (5) is outside" in m]) == 1, messages
