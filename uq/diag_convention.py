"""
diag_convention.py
==================
Settle the unified calling convention. Tests one classical and one DL method
under the corrected conventions (no bvalues arg; 2D input), so the grid can use
a single batched call for everything.

Run from repo root, venv active:
    python diag_convention.py
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np

sys.path.insert(0, os.getcwd())
from src.wrappers.OsipiBase import OsipiBase
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

METHODS = ["OGC_AmsterdamUMC_biexp", "IVIM_NEToptim"]   # classical, DL
BVALS = B_SCHEMES["clinical_sparse"]


def describe(tag, res):
    print(f"  [{tag}] OK -> type={type(res).__name__}")
    if isinstance(res, dict):
        for k, v in res.items():
            a = np.asarray(v); print(f"        {k!r:8s} shape={a.shape} sample={np.ravel(a)[:3]}")
    elif isinstance(res, (tuple, list)):
        for i, v in enumerate(res):
            a = np.asarray(v); print(f"        [{i}] shape={a.shape} sample={np.ravel(a)[:3]}")
    else:
        a = np.asarray(res); print(f"        shape={a.shape} sample={np.ravel(a)[:5]}")


def main():
    sim = simulate_repeats(**ANCHOR_TRUTHS[0], bvals=BVALS, snr=20, n_noise=20,
                           rng=np.random.default_rng(0))
    one, batch = sim["signals"][0], sim["signals"]

    for algo in METHODS:
        print(f"\n================= {algo} =================")
        model = OsipiBase(bvalues=BVALS, algorithm=algo)
        trials = [
            ("fit_2d_nobv",   lambda: model.osipi_fit(batch)),
            ("full_2d_nobv",  lambda: model.osipi_fit_full_volume(batch)),
            ("fit_1d_nobv",   lambda: model.osipi_fit(one)),
        ]
        for tag, fn in trials:
            try:
                describe(tag, fn())
            except Exception as e:
                print(f"  [{tag}] raised {type(e).__name__}: {str(e)[:110]}")


if __name__ == "__main__":
    main()
