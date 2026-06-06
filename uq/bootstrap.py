"""
bootstrap.py
============
Per-voxel residual-bootstrap uncertainty for the classical IVIM fits.

For one observed signal:
  1. fit it -> theta_hat = (D, D*, f)
  2. fitted curve S_hat(b) = ivim_signal(b, theta_hat); residuals r = S - S_hat
  3. K times: resample residuals with replacement, S* = S_hat + r*, refit -> theta*_k
  4. reported sigma = SD of {theta*_k} per parameter

This is the method's OWN per-voxel uncertainty — the honest quantity the
empirical-SD ceiling stood in for. Feeds calib.coverage() unchanged.

Caveats (for the writeup): residual resampling treats residuals as exchangeable
across b-values, which is only approximate under heteroscedastic Rician noise;
and for methods that pin to a bound, the replicates pin too, so sigma collapses
and the method reads as overconfident — a real result, not an artifact to hide.
Parametric and wild bootstrap are the natural comparison variants.
"""
from __future__ import annotations
import numpy as np
from .ivim_fit import fit_batch
from .ivim_simulator import ivim_signal


def _fit_one(model, signal_1d):
    D, Ds, F = fit_batch(model, np.asarray(signal_1d, float)[None, :])
    return float(D[0]), float(Ds[0]), float(F[0])


def bootstrap_cell(model, signals_2d, bvals, K=200, rng=None):
    """Return (est, sigma), each shape (N, 3) for [D, Dstar, f].

    est[r]   = point estimate for realization r
    sigma[r] = per-voxel bootstrap SD for realization r
    """
    rng = np.random.default_rng() if rng is None else rng
    bvals = np.asarray(bvals, float)
    sig = np.atleast_2d(np.asarray(signals_2d, float))
    N, nb = sig.shape
    est = np.full((N, 3), np.nan)
    sigma = np.full((N, 3), np.nan)

    for r in range(N):
        s = sig[r]
        theta = _fit_one(model, s)
        if not np.isfinite(theta).all():
            continue
        est[r] = theta
        shat = ivim_signal(bvals, *theta)
        resid = s - shat
        idx = rng.integers(0, nb, size=(K, nb))         # resample residuals across b
        boot = shat[None, :] + resid[idx]               # (K, nb)
        bD, bDs, bF = fit_batch(model, boot)
        for p, arr in enumerate((bD, bDs, bF)):
            a = arr[np.isfinite(arr)]
            if a.size >= 10:
                sigma[r, p] = np.std(a)
    return est, sigma
