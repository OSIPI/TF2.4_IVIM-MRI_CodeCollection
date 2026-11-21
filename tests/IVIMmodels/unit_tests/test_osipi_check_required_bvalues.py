"""
Regression test for issue #150.

`osipi_check_required_bvalues` was defined twice in OsipiBase: the second
definition was a stub with just `pass`, so it overrode the first one (which
returns True) and every call returned None instead of a boolean. The stub
definition has been removed, so the check returns True again.
Run using python -m pytest from the root folder.
"""

import os
import sys

# Ensure project root is on the path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from src.wrappers.OsipiBase import OsipiBase


def test_osipi_check_required_bvalues_returns_bool():
    """The b-values requirement check must return a boolean, not None."""
    fit = OsipiBase(bvalues=[0, 50, 200, 800])
    result = fit.osipi_check_required_bvalues()
    assert result is True
    assert isinstance(result, bool)
