"""
bayesian.py
===========
Per-voxel uncertainty for the OSIPI "Bayesian" biexp method, which is MAP-only:
fit_bayesian() finds the maximum-a-posteriori point by scipy.minimize on
neg_log_posterior and returns nothing about spread. The probe confirmed there is
no chain to extract. So we CONSTRUCT the uncertainty two ways, both reusing the
method's OWN posterior (neg_log_posterior + neg_log_prior):

  laplace_uncertainty : 2nd-order central-difference Hessian of -log posterior at
                        the MAP; cov = H^-1; sigma = sqrt(diag). Gaussian, cheap.
                        Expected to under-cover D* (skewed, bound-pinned) exactly
                        like the empirical-SD ceiling did -- a predictable result.

  mcmc_uncertainty    : emcee EnsembleSampler on log_prob = -neg_log_posterior with
                        hard bound rejection. Returns posterior SD *and* 2.5/97.5
                        quantile interval, which handles the D* skew the Gaussian
                        cannot. The quantile-interval coverage is the paper point.

Both return the method's actual MAP as the point estimate (from fit_batch), so the
only thing that varies across bootstrap / Laplace / MCMC is the sigma construction.

Param-order note: neg_log_posterior expects [D, f, Dp, S0]; the rest of the
pipeline (calib, bootstrap) uses [D, Dstar, f]. We transpose at the boundary and
fix S0 = 1 (signals are normalized to b=0), characterizing the 3-param posterior
[D, f, Dp]. The fitS0=False code path in the wrapper confirms neg_log_posterior
accepts a 3-vector. Free-S0 (4-param) is the natural sensitivity variant.

Outputs (est, sigma[, lo, hi]) each shape (N, 3) in [D, Dstar, f], drop straight
into calib.coverage() like bootstrap_cell.
"""
from __future__ import annotations
import warnings
import numpy as np

from .ivim_fit import fit_batch

# --- locate the method's own posterior pieces (probe-confirmed module) --------
_NLP_MODULE = "src.original.fitting.OGC_AmsterdamUMC.LSQ_fitting"
try:
    from src.original.fitting.OGC_AmsterdamUMC.LSQ_fitting import (  # type: ignore
        neg_log_posterior,
    )
except Exception as e:  # pragma: no cover - surfaced clearly by the runner self-test
    neg_log_posterior = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

try:
    from src.original.fitting.OGC_AmsterdamUMC.LSQ_fitting import (  # type: ignore
        flat_neg_log_prior,
    )
except Exception:
    flat_neg_log_prior = None


# --- param-order bookkeeping --------------------------------------------------
# calib / pipeline order : [D, Dstar, f]      (indices 0,1,2)
# neg_log_posterior order: [D, f,    Dp]      (Dp == Dstar), S0 fixed = 1
_CALIB_TO_NLP = (0, 2, 1)   # calib[D,Dstar,f] -> nlp[D,f,Dp]
_NLP_TO_CALIB = (0, 2, 1)   # nlp[D,f,Dp]      -> calib[D,Dstar,f]


def _nlp_from_calib(theta_calib):
    """[D, Dstar, f] -> [D, f, Dp] for neg_log_posterior."""
    t = np.asarray(theta_calib, float)
    return np.array([t[_CALIB_TO_NLP[0]], t[_CALIB_TO_NLP[1]], t[_CALIB_TO_NLP[2]]])


def _calib_from_nlp(v_nlp):
    """[*, D][*, f][*, Dp] -> [*, D][*, Dstar][*, f] (works on length-3 vectors)."""
    v = np.asarray(v_nlp, float)
    return np.array([v[_NLP_TO_CALIB[0]], v[_NLP_TO_CALIB[1]], v[_NLP_TO_CALIB[2]]])


def _resolve_prior(model):
    """Pull the method's own prior off the model; fall back to a flat prior."""
    prior = getattr(model, "neg_log_prior", None)
    if prior is not None:
        return prior
    if flat_neg_log_prior is None:
        raise RuntimeError(
            "model has no .neg_log_prior and flat_neg_log_prior could not be "
            f"imported from {_NLP_MODULE}."
        )
    b = getattr(model, "bounds", None)
    if b is None:
        # wrapper defaults from the probe: ([0,0,0,0],[0.005,1.5,2,2.5]) in D,f,Dp,S0
        return flat_neg_log_prior([0, 0.005], [0, 1.5], [0, 2], [0, 2.5])
    return flat_neg_log_prior(b["D"], b["f"], b["Dp"], b["S0"])


def _bounds_nlp3(model):
    """Lower/upper bounds for [D, f, Dp] from the model (or probe defaults)."""
    b = getattr(model, "bounds", None)
    if b is None:
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([0.005, 1.5, 2.0])
    else:
        lo = np.array([b["D"][0], b["f"][0], b["Dp"][0]], float)
        hi = np.array([b["D"][1], b["f"][1], b["Dp"][1]], float)
    return lo, hi


def _map_points(model, signals_2d):
    """Method's actual MAP per voxel, shape (N, 3) in [D, Dstar, f]."""
    sig = np.atleast_2d(np.asarray(signals_2d, float))
    D, Ds, F = fit_batch(model, sig)
    return np.column_stack([np.asarray(D, float),
                            np.asarray(Ds, float),
                            np.asarray(F, float)])


def _objective(bvals, signal, prior):
    """f(theta3) = neg_log_posterior at fixed S0=1, theta3 = [D, f, Dp]."""
    bvals = np.asarray(bvals, float)
    signal = np.asarray(signal, float)

    def f(theta3):
        return float(neg_log_posterior(np.asarray(theta3, float), bvals, signal, prior))

    return f


# --- Laplace ------------------------------------------------------------------
def _hessian(f, x, rel=1e-3, absmin=1e-9):
    """2nd-order central-difference Hessian of scalar f at x."""
    x = np.asarray(x, float)
    n = x.size
    h = np.maximum(np.abs(x) * rel, absmin)
    f0 = f(x)
    if not np.isfinite(f0):
        return None
    H = np.zeros((n, n))
    e = np.eye(n)
    for i in range(n):
        xi = h[i] * e[i]
        fip = f(x + xi)
        fim = f(x - xi)
        H[i, i] = (fip - 2.0 * f0 + fim) / (h[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = h[i] * e[i], h[j] * e[j]
            fpp = f(x + xi + xj)
            fpm = f(x + xi - xj)
            fmp = f(x - xi + xj)
            fmm = f(x - xi - xj)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    return H if np.isfinite(H).all() else None


def laplace_uncertainty(model, signals_2d, bvals, rel=1e-3):
    """Return (est, sigma), each (N, 3) in [D, Dstar, f].

    sigma is the Laplace posterior SD (sqrt of the diagonal of H^-1 at the MAP).
    NaN where the MAP is non-finite or the local curvature is not invertible /
    not positive (an honest signal that the Gaussian approx fails there).
    """
    est = _map_points(model, signals_2d)
    N = est.shape[0]
    prior = _resolve_prior(model)
    sig = np.atleast_2d(np.asarray(signals_2d, float))
    sigma = np.full((N, 3), np.nan)

    for r in range(N):
        theta_c = est[r]
        if not np.isfinite(theta_c).all():
            continue
        f = _objective(bvals, sig[r], prior)
        x = _nlp_from_calib(theta_c)           # [D, f, Dp]
        H = _hessian(f, x, rel=rel)
        if H is None:
            continue
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        d = np.diag(cov)
        sd_nlp = np.where(d > 0, np.sqrt(np.abs(d)), np.nan)   # [D, f, Dp]
        sigma[r] = _calib_from_nlp(sd_nlp)                     # [D, Dstar, f]
    return est, sigma


# --- MCMC ---------------------------------------------------------------------
def mcmc_uncertainty(model, signals_2d, bvals, nwalkers=32, nsteps=1500,
                     burn=500, thin=5, rng=None):
    """Return (est, sigma, lo, hi), each (N, 3) in [D, Dstar, f].

    est : method's MAP (point actually reported by the method)
    sigma: posterior SD from the emcee chain
    lo,hi: 2.5 / 97.5 percentile credible interval (the skew-aware variant)

    log_prob = -neg_log_posterior with hard bound rejection (-inf outside box).
    """
    try:
        import emcee
    except ImportError as e:
        raise ImportError(
            "mcmc_uncertainty needs emcee:  pip install emcee"
        ) from e

    rng = np.random.default_rng() if rng is None else rng
    est = _map_points(model, signals_2d)
    N = est.shape[0]
    prior = _resolve_prior(model)
    lo_b, hi_b = _bounds_nlp3(model)
    sig = np.atleast_2d(np.asarray(signals_2d, float))

    sigma = np.full((N, 3), np.nan)
    lo = np.full((N, 3), np.nan)
    hi = np.full((N, 3), np.nan)

    ndim = 3
    for r in range(N):
        theta_c = est[r]
        if not np.isfinite(theta_c).all():
            continue
        f = _objective(bvals, sig[r], prior)
        x0 = _nlp_from_calib(theta_c)          # [D, f, Dp]

        def log_prob(theta):
            if np.any(theta < lo_b) or np.any(theta > hi_b):
                return -np.inf
            val = f(theta)
            return -val if np.isfinite(val) else -np.inf

        # walkers in a small ball around the MAP, scaled per-param, clipped to box
        scale = np.maximum(np.abs(x0) * 0.05, 1e-6)
        p0 = x0[None, :] + scale[None, :] * rng.standard_normal((nwalkers, ndim))
        p0 = np.clip(p0, lo_b + 1e-9, hi_b - 1e-9)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler.run_mcmc(p0, nsteps, progress=False)
        chain = sampler.get_chain(discard=burn, thin=thin, flat=True)  # (M, 3) nlp
        if chain.size == 0 or not np.isfinite(chain).any():
            continue

        sd_nlp = np.nanstd(chain, axis=0)
        q_lo = np.nanpercentile(chain, 2.5, axis=0)
        q_hi = np.nanpercentile(chain, 97.5, axis=0)
        sigma[r] = _calib_from_nlp(sd_nlp)
        lo[r] = _calib_from_nlp(q_lo)
        hi[r] = _calib_from_nlp(q_hi)
    return est, sigma, lo, hi
