"""
dl_uncertainty.py
=================
Uncertainty for the unsupervised/supervised DL methods (IVIM_NEToptim,
Super_IVIM_DC). The probe settled the route:

  - MC-dropout : BLOCKED. No nn.Dropout layers, and no in-memory net to hook --
                 the trained weights live in .pt files on disk.
  - bootstrap  : WRONG TOOL. These methods retrain at construction and
                 re-simulate a 1e6-sample reference per inference, so resampling
                 refits is both intractable and conflates training stochasticity.
  - ensemble   : THE ROUTE. self.stochastic = True and make_model() trains a fresh
                 net from random init, so M independent retrains give a genuine
                 deep ensemble. SD across members = the method's own run-to-run
                 (epistemic + training) uncertainty -- exactly what a user feels.

ensemble_uncertainty() returns (est, sigma, preds), est/sigma each (N, 3) in
[D, Dstar, f] so they drop into calib.coverage() like bootstrap/Bayesian. est is
the ensemble MEAN (canonical deep-ensemble point estimate); preds is the full
(M, N, 3) stack for diagnostics / degeneracy checking.

NOTE: each member is a full retrain. M members => M trainings per call. Budget
overnight for the grid; minutes-to-tens-of-minutes for a single test cell.
"""
from __future__ import annotations
import os, sys, random, contextlib
import numpy as np

from .ivim_fit import make_model, fit_batch
from .ivim_simulator import ivim_signal


@contextlib.contextmanager
def _silence():
    """Swallow the training / simulation stdout+stderr spam from the nets."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _seed(s):
    np.random.seed(s)
    random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def _build(algo, bvals, snr):
    """make_model that tolerates DL methods needing an SNR kwarg (train SNR)."""
    try:
        return make_model(algo, bvals, snr=snr)
    except TypeError:
        return make_model(algo, bvals)


def ensemble_uncertainty(algo, signals_2d, bvals, M=5, seeds=None, snr=20,
                         verbose=True):
    """Return (est, sigma, preds).

    est   : (N, 3) ensemble mean  [D, Dstar, f]
    sigma : (N, 3) ensemble SD across the M retrains (ddof=1)
    preds : (M, N, 3) per-member predictions (for degeneracy / spread checks)
    """
    sig = np.atleast_2d(np.asarray(signals_2d, float))
    N = sig.shape[0]
    seeds = list(range(M)) if seeds is None else list(seeds)
    M = len(seeds)
    preds = np.full((M, N, 3), np.nan)

    for m, s in enumerate(seeds):
        if verbose:
            print(f"    member {m + 1}/{M} (seed {s}) — retraining ...", flush=True)
        _seed(s)
        with _silence():
            model = _build(algo, bvals, snr)
            D, Ds, F = fit_batch(model, sig)
        preds[m] = np.column_stack([np.asarray(D, float),
                                    np.asarray(Ds, float),
                                    np.asarray(F, float)])

    est = np.nanmean(preds, axis=0)                 # (N, 3) ensemble mean
    with np.errstate(invalid="ignore"):
        sigma = np.nanstd(preds, axis=0, ddof=1)    # (N, 3) member spread
    return est, sigma, preds


def ensemble_degeneracy(preds):
    """Mean across-member SD per param; ~0 => retrains are deterministic and the
    ensemble is degenerate (seeds aren't reaching the training RNG)."""
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(preds, axis=0, ddof=1)       # (N, 3)
    return np.nanmean(sd, axis=0)                    # (3,)


# --- input perturbation (parametric bootstrap) --------------------------------
def _rician(clean, sigma, rng):
    """Rician-noised copy of clean signal: |clean + N(0,sigma) + i N(0,sigma)|."""
    re = clean + sigma * rng.standard_normal(clean.shape)
    im = sigma * rng.standard_normal(clean.shape)
    return np.sqrt(re ** 2 + im ** 2)


def input_perturbation_uncertainty(algo, signals_2d, bvals, snr=20, B=50,
                                   rng=None, verbose=True):
    """Predictive (aleatoric) uncertainty via parametric bootstrap.

    Train once, take the base estimate, rebuild each voxel's clean curve from it,
    re-noise at the cell SNR B times, and refit. SD across replicas captures the
    measurement-noise variance the retrain ensemble could not.

      est0  : (N, 3) base reported estimate [D, Dstar, f]
      sigma : (N, 3) SD across the B noised refits
      preds : (B, N, 3) per-replica predictions

    The refit auto-retrains for unsupervised nets (NEToptim) and infers for
    supervised ones (S_DC), so cost asymmetry is automatic. Noise scale is
    sigma = 1/snr on the S0=1 normalized signal (standard OSIPI convention).
    """
    rng = np.random.default_rng() if rng is None else rng
    bvals = np.asarray(bvals, float)
    sig = np.atleast_2d(np.asarray(signals_2d, float))
    N, nb = sig.shape
    sigma_noise = 1.0 / snr

    if verbose:
        print("    training once + base fit ...", flush=True)
    _seed(0)
    with _silence():
        model0 = _build(algo, bvals, snr)
        D0, Ds0, F0 = fit_batch(model0, sig)
    est0 = np.column_stack([np.asarray(D0, float),
                            np.asarray(Ds0, float),
                            np.asarray(F0, float)])

    # clean curve per voxel from the base estimate (parametric center)
    Shat = np.full((N, nb), np.nan)
    for r in range(N):
        if np.isfinite(est0[r]).all():
            Shat[r] = ivim_signal(bvals, est0[r, 0], est0[r, 1], est0[r, 2])

    preds = np.full((B, N, 3), np.nan)
    for b in range(B):
        if verbose:
            print(f"    replica {b + 1}/{B} ...", flush=True)
        noised = _rician(Shat, sigma_noise, rng)        # (N, nb); nan rows stay nan
        with _silence():
            Db, Dsb, Fb = fit_batch(model0, noised)
        preds[b] = np.column_stack([np.asarray(Db, float),
                                    np.asarray(Dsb, float),
                                    np.asarray(Fb, float)])

    with np.errstate(invalid="ignore"):
        sigma = np.nanstd(preds, axis=0, ddof=1)
    return est0, sigma, preds
