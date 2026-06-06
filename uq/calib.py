"""
calib.py
========
The calibration ruler for the IVIM uncertainty study.

Given per-realization point estimates, a known truth, and a reported uncertainty
(sigma), it measures whether the stated uncertainty is honest:

  coverage(level) : fraction of realizations whose nominal-level interval
                    [estimate ± z·sigma] actually contains the truth.
                    Calibrated  -> coverage(L) ≈ L for all L.
                    Overconfident (sigma too small / bias too large) -> coverage << L.
  ECE             : mean |coverage(L) − L| over levels. 0 = perfect.
  sharpness       : mean relative interval half-width. Calibration is cheap with
                    huge intervals — must be reported alongside coverage.

Phase A uses each method's empirical ensemble SD as its best-case reported sigma.
W2 swaps that for genuinely per-voxel reported uncertainty (bootstrap / posterior
/ MC-dropout / ensemble) through the same interface.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

LEVELS = np.array([0.50, 0.68, 0.80, 0.90, 0.95, 0.99])


def coverage(estimates, truth, sigma, levels=LEVELS):
    """estimates: (N,). sigma: scalar or (N,). Returns {level: empirical coverage}."""
    est = np.asarray(estimates, float)
    sig = np.full(est.shape, sigma, float) if np.ndim(sigma) == 0 else np.asarray(sigma, float)
    m = np.isfinite(est) & np.isfinite(sig) & (sig > 0)
    est, sig = est[m], sig[m]
    if est.size == 0:
        return {round(float(l), 2): np.nan for l in levels}
    err = np.abs(est - truth)
    return {round(float(l), 2): float(np.mean(err <= norm.ppf(0.5 + l / 2) * sig))
            for l in levels}


def ece(cov):
    lev = np.array(list(cov.keys()), float)
    emp = np.array(list(cov.values()), float)
    ok = np.isfinite(emp)
    return float(np.mean(np.abs(emp[ok] - lev[ok]))) if ok.any() else np.nan


def sharpness_rel(sigma, truth, level=0.95):
    s = np.nanmean(np.atleast_1d(np.asarray(sigma, float)))
    return float(2 * norm.ppf(0.5 + level / 2) * s / abs(truth)) if truth else np.nan


# self-test: a perfectly calibrated draw must return coverage ≈ nominal
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    truth, s, N = 0.03, 0.006, 200000
    est = rng.normal(truth, s, N)             # unbiased, known sigma
    cov = coverage(est, truth, s)
    print("calibrated oracle (should track nominal):")
    for l, c in cov.items():
        print(f"  nominal={l:.2f}  empirical={c:.3f}")
    print(f"  ECE={ece(cov):.4f}  (≈0 expected)")

    biased = rng.normal(truth + 5 * s, s, N)  # confidently wrong
    cb = coverage(biased, truth, s)
    print(f"\noverconfident-and-biased ECE={ece(cb):.3f} (large expected); "
          f"95%-coverage={cb[0.95]:.3f} (≈0 expected)")
