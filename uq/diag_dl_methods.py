"""
diag_dl_methods.py
==================
Figure out why the deep-learning IVIM methods returned NaN: inspect their
return structure and the b-values they actually use, and test three calling
conventions (single-voxel, batched 2D, full-volume).

WARNING: constructing these triggers network training (~10 min each). Comment
one out in DL_METHODS if you only want to look at one first.

Run from repo root, venv active, ivim_simulator.py alongside:
    python diag_dl_methods.py
"""
from __future__ import annotations
import os, sys, traceback
import numpy as np

sys.path.insert(0, os.getcwd())
from src.wrappers.OsipiBase import OsipiBase
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

DL_METHODS = ["IVIM_NEToptim", "Super_IVIM_DC"]
BVALS = B_SCHEMES["clinical_sparse"]
TRUTH = ANCHOR_TRUTHS[0]


def describe(tag, res):
    print(f"  [{tag}] type={type(res).__name__}")
    if isinstance(res, dict):
        for k, v in res.items():
            arr = np.asarray(v)
            print(f"        key={k!r:10s} shape={arr.shape} "
                  f"sample={np.ravel(arr)[:3]}")
    elif isinstance(res, (tuple, list)):
        for i, v in enumerate(res):
            arr = np.asarray(v)
            print(f"        [{i}] shape={arr.shape} sample={np.ravel(arr)[:3]}")
    else:
        arr = np.asarray(res)
        print(f"        shape={arr.shape} sample={np.ravel(arr)[:5]}")


def main():
    sim = simulate_repeats(**TRUTH, bvals=BVALS, snr=20, n_noise=20,
                           rng=np.random.default_rng(0))
    one = sim["signals"][0]          # (n_bvals,)
    batch = sim["signals"]           # (20, n_bvals)
    print(f"eval b-values passed in: {BVALS}")
    print(f"truth: {TRUTH}\n")

    for algo in DL_METHODS:
        print(f"\n================= {algo} =================")
        try:
            model = OsipiBase(bvalues=BVALS, algorithm=algo)
        except Exception:
            print("  init FAILED:"); traceback.print_exc(); continue

        # what b-values does the model think it has?
        for attr in ("bvalues", "bvals", "b_values"):
            if hasattr(model, attr):
                print(f"  model.{attr} = {np.asarray(getattr(model, attr)).ravel()}")
        try:
            model.osipi_print_requirements()
        except Exception:
            pass

        for tag, data in (("single", one), ("batch2D", batch)):
            try:
                describe(tag, model.osipi_fit(data, BVALS))
            except Exception as e:
                print(f"  [{tag}] raised {type(e).__name__}: {str(e)[:120]}")

        if hasattr(model, "osipi_fit_full_volume"):
            try:
                describe("full_volume", model.osipi_fit_full_volume(batch, BVALS))
            except Exception as e:
                print(f"  [full_volume] raised {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
