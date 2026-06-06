"""
uq — IVIM uncertainty-quantification & calibration study
========================================================
Original analysis layer built on top of the (unmodified) OSIPI TF2.4
``src/`` fitting engines. See the repository README for the scientific
framing; each module carries its own docstring.

Path shim
---------
The analysis modules reach into the upstream tree with absolute imports
(``from src.wrappers.OsipiBase import OsipiBase``), and OsipiBase itself
requires the project root to be present *by absolute path* on ``sys.path``
(it raises otherwise). Importing this package guarantees that, so

    python -c "import uq.calib"
    python -m uq.run_w3_calib

work regardless of the current working directory. Intra-package imports
are relative (``from .ivim_fit import ...``), so only this one hook is needed.
"""
from __future__ import annotations
import os as _os
import sys as _sys

# Repo root = parent of this package directory.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
