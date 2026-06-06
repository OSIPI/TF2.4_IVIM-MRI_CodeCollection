"""
smoke_fit.py
============
Verify the unified fitting layer works end-to-end for BOTH a classical and a
deep-learning method on a single cell, before committing to a full grid run.
One DL training cycle (~15 min).

Run from repo root, venv active, with ivim_simulator.py and ivim_fit.py alongside:
    python smoke_fit.py
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np

sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

from .ivim_fit import make_model, fit_batch
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

BVALS = B_SCHEMES["clinical_sparse"]
TRUTH = ANCHOR_TRUTHS[0]

sim = simulate_repeats(**TRUTH, bvals=BVALS, snr=20, n_noise=100,
                       rng=np.random.default_rng(0))
print(f"truth: D={TRUTH['D']:.3g}  D*={TRUTH['Dstar']:.3g}  f={TRUTH['f']:.3g}"
      f"   SNR=20  n=100  clinical_sparse\n")

for algo in ["OGC_AmsterdamUMC_biexp", "IVIM_NEToptim"]:
    D, Ds, F = fit_batch(make_model(algo, BVALS), sim["signals"])
    print(algo)
    for name, arr, tv in (("D", D, TRUTH["D"]),
                          ("D*", Ds, TRUTH["Dstar"]),
                          ("f", F, TRUTH["f"])):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            print(f"  {name:3s} no successful fits"); continue
        med = np.median(a)
        iqr = np.subtract(*np.percentile(a, [75, 25]))
        print(f"  {name:3s} truth={tv:<8.4g} median={med:<10.4g} "
              f"bias={100*(med-tv)/tv:+7.1f}%  IQR/med={100*iqr/med:7.1f}%  "
              f"n_ok={a.size}")
    print()
