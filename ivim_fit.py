"""
ivim_fit.py
===========
Unified, batched IVIM fitting layer for the OSIPI benchmark / calibration study.

The OSIPI wrappers split into two calling conventions, discovered empirically:
  * classical  -> model.osipi_fit(signals_2d)              returns dict of arrays
  * deep learning -> model.osipi_fit_full_volume(signals_2d) returns dict of arrays
Both take a 2D (n_voxels, n_bvalues) array, NO bvalues argument, and return
keys D / Dp / f. fit_batch() tries both and normalizes the output.

Used by the accuracy grid and by the W-path calibration code so the calling
convention lives in exactly one place.
"""
from __future__ import annotations
import numpy as np
from src.wrappers.OsipiBase import OsipiBase


def make_model(algorithm, bvals):
    return OsipiBase(bvalues=np.asarray(bvals, dtype=float), algorithm=algorithm)


def _normalize(res):
    """dict -> (D, Dstar, f) as float arrays (Dp is OSIPI's name for D*)."""
    D  = res.get("D")
    Ds = res.get("Dp", res.get("D*", res.get("Dstar", res.get("Ds"))))
    f  = res.get("f",  res.get("Fp"))
    return np.asarray(D, float), np.asarray(Ds, float), np.asarray(f, float)


def fit_batch(model, signals_2d):
    """Fit an (n_voxels, n_bvalues) batch.

    Returns (D, Dstar, f), each shape (n_voxels,). Failed voxels -> NaN.
    """
    sig = np.atleast_2d(np.asarray(signals_2d, dtype=float))

    # path 1 — classical batched fit
    try:
        res = model.osipi_fit(sig)
        if isinstance(res, dict):
            return _normalize(res)
    except Exception:
        pass

    # path 2 — deep-learning full-volume fit
    try:
        res = model.osipi_fit_full_volume(sig)
        if isinstance(res, dict):
            return _normalize(res)
    except Exception:
        pass

    # path 3 — per-voxel fallback for any 1D-only wrapper
    D, Ds, F = [], [], []
    for v in sig:
        try:
            d, ds, f = _normalize(model.osipi_fit(v))
            D.append(float(d)); Ds.append(float(ds)); F.append(float(f))
        except Exception:
            D.append(np.nan); Ds.append(np.nan); F.append(np.nan)
    return np.array(D), np.array(Ds), np.array(F)
