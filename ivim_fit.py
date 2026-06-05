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


def _full_volume_chunked(model, sig):
    """Deep-learning full-volume fit, robust to the wrapper's voxel-count
    off-by-one (a non-finite normalized row at low SNR makes the net return
    N-1 predictions; the wrapper's np.reshape then raises and osipi_fit_full_volume
    swallows it into a `False`, which previously dumped the WHOLE cell to the
    per-voxel path and produced an all-NaN cell).

    Strategy: try the full batch; if the call fails OR returns the wrong length,
    recursively bisect and refit each half, isolating the offending voxel(s) down
    to size 1 (NaN only there). Inference on a trained net is ~free, so chunking
    costs nothing material — the one expensive train already happened at
    construction. Returns (D, Dstar, f) each shape (n,), or None if the model has
    no working full-volume path at all (so the caller can fall through).
    """
    # Only models with a real full-volume implementation get this path; anything
    # else (classical 1D-only wrappers) returns None so fit_batch falls through
    # to its per-voxel path instead of bisecting to an all-NaN cell.
    if not hasattr(model, "ivim_fit_full_volume"):
        return None

    n = len(sig)
    try:
        res = model.osipi_fit_full_volume(sig)
    except Exception:
        res = None
    if isinstance(res, dict):
        D, Ds, F = _normalize(res)
        if len(D) == n and len(Ds) == n and len(F) == n:
            return D, Ds, F
        # dict came back the wrong length — treat as a failed chunk, bisect below
    elif res is None and n > 1:
        # hard exception on a multi-voxel batch: still worth bisecting
        pass
    elif res is False or res is None:
        if n == 1:
            return (np.array([np.nan]),) * 3   # genuine single-voxel failure
        # else: bisect below
    else:
        return None                            # not a full-volume-capable model

    if n == 1:
        return (np.array([np.nan]),) * 3
    mid = n // 2
    left = _full_volume_chunked(model, sig[:mid])
    right = _full_volume_chunked(model, sig[mid:])
    if left is None or right is None:
        return None
    return (np.concatenate([left[0], right[0]]),
            np.concatenate([left[1], right[1]]),
            np.concatenate([left[2], right[2]]))


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

    # path 2 — deep-learning full-volume fit (chunked fallback isolates the
    # off-by-one voxel instead of NaN-ing the whole cell)
    fv = _full_volume_chunked(model, sig)
    if fv is not None:
        return fv

    # path 3 — per-voxel fallback for any 1D-only wrapper
    D, Ds, F = [], [], []
    for v in sig:
        try:
            d, ds, f = _normalize(model.osipi_fit(v))
            D.append(float(d)); Ds.append(float(ds)); F.append(float(f))
        except Exception:
            D.append(np.nan); Ds.append(np.nan); F.append(np.nan)
    return np.array(D), np.array(Ds), np.array(F)
