"""
Weighted Least Squares (WLS) / Robust Linear Model (RLM) IVIM fitting.

Author: Devguru Tiwari, IIIT Nagpur
Date: 2026-03-01

Implements a segmented approach for IVIM parameter estimation:
1. Estimate D from high b-values using weighted/robust linear regression on log-signal
2. Estimate f from the intercept of the Step 1 fit
3. Estimate D* from residuals at low b-values using weighted/robust linear regression

Two regression methods are available:
- WLS: Weighted Linear Least Squares with Veraart weights (w = S^2)
- RLM: Robust Linear Model using Huber's T norm (statsmodels)

Reference:
    Veraart, J. et al. (2013). "Weighted linear least squares estimation of
    diffusion MRI parameters: strengths, limitations, and pitfalls."
    NeuroImage, 81, 335-346.
    DOI: 10.1016/j.neuroimage.2013.05.028

Requirements:
    numpy
    statsmodels (only for method="RLM")
"""

import numpy as np
import warnings


def _weighted_linreg(x, y, weights):
    """Fast weighted linear regression: y = a + b*x.

    Uses Veraart et al. (2013) approach with weights = S^2.

    Args:
        x: 1D array, independent variable.
        y: 1D array, dependent variable.
        weights: 1D array, weights for each observation.

    Returns:
        (intercept, slope) tuple.
    """
    W = np.diag(weights)
    X = np.column_stack([np.ones_like(x), x])
    # Weighted normal equations: (X^T W X) beta = X^T W y
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta[0], beta[1]  # intercept, slope


def _rlm_linreg(x, y):
    """Robust linear regression using statsmodels RLM with Huber's T norm.

    RLM down-weights outlier observations via iteratively reweighted least
    squares (IRLS), making the fit resistant to corrupted/noisy voxels.

    Args:
        x: 1D array, independent variable.
        y: 1D array, dependent variable.

    Returns:
        (intercept, slope) tuple.
    """
    import statsmodels.api as sm
    X = sm.add_constant(x)
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    result = model.fit()
    return result.params[0], result.params[1]  # intercept, slope


def wls_ivim_fit(bvalues, signal, cutoff=200, method="WLS"):
    """
    IVIM fit using WLS or RLM (segmented approach).

    Step 1: Fit D from high b-values on log-signal.
    Step 2: Fit D* from residuals at low b-values.

    Args:
        bvalues (array-like): 1D array of b-values (s/mm²).
        signal (array-like): 1D array of signal intensities (will be normalized).
        cutoff (float): b-value threshold separating D from D* fitting.
                        Default: 200 s/mm².
        method (str): Regression method to use.
            - "WLS": Weighted Least Squares with Veraart S² weights (default).
            - "RLM": Robust Linear Model with Huber's T norm (statsmodels).

    Returns:
        tuple: (D, f, Dp) where
            D (float): True diffusion coefficient (mm²/s).
            f (float): Perfusion fraction (0-1).
            Dp (float): Pseudo-diffusion coefficient (mm²/s).
    """
    method = method.upper()
    if method not in ("WLS", "RLM"):
        raise ValueError(f"Unknown method '{method}'. Use 'WLS' or 'RLM'.")

    bvalues = np.array(bvalues, dtype=float)
    signal = np.array(signal, dtype=float)

    # Normalize signal to S(b=0)
    s0_vals = signal[bvalues == 0]
    if len(s0_vals) == 0 or np.mean(s0_vals) <= 0:
        return 0.0, 0.0, 0.0
    s0 = np.mean(s0_vals)
    signal = signal / s0

    try:
        # ── Step 1: Estimate D from high b-values ─────────────────────
        # At high b, perfusion component ≈ 0, so:
        #   S(b) ≈ (1 - f) * exp(-b * D)
        #   ln(S(b)) = ln(1 - f) - b * D
        high_mask = bvalues >= cutoff
        b_high = bvalues[high_mask]
        s_high = signal[high_mask]

        # Guard against zero/negative signal values
        s_high = np.maximum(s_high, 1e-8)
        log_s = np.log(s_high)

        if method == "WLS":
            # Veraart weights: w = S^2 (corrects for noise in log-domain)
            weights_high = s_high ** 2
            intercept, D = _weighted_linreg(-b_high, log_s, weights_high)
        else:
            # RLM: robust regression, no explicit weights needed
            intercept, D = _rlm_linreg(-b_high, log_s)

        # Extract f from intercept: intercept = ln(1 - f)
        f = 1.0 - np.exp(intercept)

        # Clamp to physically meaningful ranges
        D = np.clip(D, 0, 0.005)
        f = np.clip(f, 0, 1)

        # ── Step 2: Estimate D* from low b-value residuals ────────────
        # Subtract the diffusion component:
        #   residual(b) = S(b) - (1 - f) * exp(-b * D)
        #   ≈ f * exp(-b * D*)
        #   ln(residual) = ln(f) - b * D*
        residual = signal - (1 - f) * np.exp(-bvalues * D)

        low_mask = (bvalues < cutoff) & (bvalues > 0)
        b_low = bvalues[low_mask]
        r_low = residual[low_mask]

        # Guard against zero/negative residuals
        r_low = np.maximum(r_low, 1e-8)
        log_r = np.log(r_low)

        if len(b_low) >= 2:
            if method == "WLS":
                weights_low = r_low ** 2
                _, Dp = _weighted_linreg(-b_low, log_r, weights_low)
            else:
                _, Dp = _rlm_linreg(-b_low, log_r)
            Dp = np.clip(Dp, 0.005, 0.2)
        else:
            Dp = 0.01  # fallback

        # Ensure D* > D (by convention)
        if Dp < D:
            D, Dp = Dp, D
            f = 1 - f

        return D, f, Dp

    except Exception:
        # If fit fails, return zeros (consistent with other algorithms)
        return 0.0, 0.0, 0.0
