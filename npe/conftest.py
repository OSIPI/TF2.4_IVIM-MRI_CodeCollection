"""Make the NPE scaffold modules importable under pytest.

The NPE project area is a flat, self-contained directory (its own copy of
``ivim_simulator``), so put it on ``sys.path`` rather than relying on package
install. Keeping this conftest at the npe/ root also pins pytest's rootdir here,
isolating collection from the repo-wide OSIPI test suite (which needs the other,
incompatible venv).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
