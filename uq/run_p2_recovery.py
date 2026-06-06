"""
run_p2_recovery.py
==================
First real recovery for the OSIPI IVIM benchmark.

Takes the known-truth signals from ivim_simulator.py, fits them through two
OSIPI algorithms via OsipiBase, and reports how well each recovers the true
(D, D*, f) -- separating BIAS (systematic offset) from PRECISION (scatter).

RUN FROM THE REPO ROOT (so `src.wrappers.OsipiBase` imports), with the venv
active, and with ivim_simulator.py placed in that same root:

    python run_p2_recovery.py
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np

sys.path.insert(0, os.getcwd())            # repo root -> import src.* and the sim
warnings.filterwarnings("ignore")          # the benign bounds/D* warnings you saw

from src.wrappers.OsipiBase import OsipiBase
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

# --- config ---------------------------------------------------------------
ALGORITHMS = [
    "OGC_AmsterdamUMC_biexp_segmented",   # two-step segmented LSQ
    "OGC_AmsterdamUMC_biexp",             # full simultaneous nonlinear LSQ
]
TRUTH   = ANCHOR_TRUTHS[0]                 # D=1.1e-3, D*=3.0e-2, f=0.15
BVALS   = B_SCHEMES["clinical_sparse"]
SNR     = 20
N_NOISE = 200                              # small for a quick first pass; scale up in P3
SEED    = 0


def parse_result(res):
    """Pull (D, Dstar, f) out of an osipi_fit return, tolerant of key naming."""
    if isinstance(res, dict):
        D  = res.get("D")
        Ds = res.get("Dp", res.get("D*", res.get("Dstar", res.get("Ds"))))
        f  = res.get("f", res.get("Fp"))
        return D, Ds, f
    raise TypeError(f"Unexpected fit return type {type(res)}: {res!r}\n"
                    "Paste this and I'll adjust the parser.")


def fit_all(algorithm, sim):
    model = OsipiBase(bvalues=sim["bvalues"], algorithm=algorithm)
    print(f"\n=== {algorithm} ===")
    try:
        model.osipi_print_requirements()
    except Exception:
        pass

    D, Ds, F = [], [], []
    fails = 0
    for sig in sim["signals"]:
        try:
            d, ds, f = parse_result(model.osipi_fit(sig, sim["bvalues"]))
            if d is None or ds is None or f is None or not np.isfinite([d, ds, f]).all():
                fails += 1; continue
            D.append(float(d)); Ds.append(float(ds)); F.append(float(f))
        except Exception:
            fails += 1
    return np.array(D), np.array(Ds), np.array(F), fails


def report(name, est, truth):
    if est.size == 0:
        print(f"  {name:5s}: no successful fits"); return
    med  = np.median(est)
    bias = 100 * (med - truth) / truth
    cov  = 100 * np.std(est) / np.mean(est)
    print(f"  {name:5s}  truth={truth:.4g}  median={med:.4g}  "
          f"bias={bias:+6.1f}%  CoV={cov:5.1f}%")


def main():
    rng = np.random.default_rng(SEED)
    sim = simulate_repeats(**TRUTH, bvals=BVALS, snr=SNR, n_noise=N_NOISE, rng=rng)
    print(f"Truth: D={TRUTH['D']:.3g}  D*={TRUTH['Dstar']:.3g}  f={TRUTH['f']:.3g}"
          f"   |  SNR={SNR}  n={N_NOISE}  scheme=clinical_sparse")

    for algo in ALGORITHMS:
        D, Ds, F, fails = fit_all(algo, sim)
        print(f"  fit failures: {fails}/{N_NOISE}")
        report("D",  D,  TRUTH["D"])
        report("D*", Ds, TRUTH["Dstar"])
        report("f",  F,  TRUTH["f"])


if __name__ == "__main__":
    main()
