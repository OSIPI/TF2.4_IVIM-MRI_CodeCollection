"""
Weighted Least Squares (WLS) IVIM fitting.

Author: Devguru Tiwari, IIIT Nagpur
Date: 2026-03-01

Implements a segmented approach for IVIM parameter estimation:
1. Estimate D from high b-values using weighted linear regression on log-signal
2. Estimate f from the intercept of the Step 1 fit
3. Estimate D* from residuals at low b-values using weighted linear regression

Weighting follows Veraart et al. (2013): weights = signal^2 to account
for heteroscedasticity introduced by the log-transform.

Reference:
    Veraart, J. et al. (2013). "Weighted linear least squares estimation of
    diffusion MRI parameters: strengths, limitations, and pitfalls."
    NeuroImage, 81, 335-346.
    DOI: 10.1016/j.neuroimage.2013.05.028

Requirements:
    numpy
"""

import numpy as np
import warnings


def _weighted_linreg(x, y, weights):
    """Fast weighted linear regression: y = a + b*x.

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


def wls_ivim_fit(bvalues, signal, cutoff=200):
    """
    Weighted Least Squares IVIM fit (segmented approach).

    Step 1: Fit D from high b-values using WLS on log-signal.
            Weights = S(b)^2 (Veraart et al. 2013).
    Step 2: Fit D* from residuals at low b-values using WLS.

    Args:
        bvalues (array-like): 1D array of b-values (s/mm²).
        signal (array-like): 1D array of signal intensities (will be normalized).
        cutoff (float): b-value threshold separating D from D* fitting.
                        Default: 200 s/mm².

    Returns:
        tuple: (D, f, Dp) where
            D (float): True diffusion coefficient (mm²/s).
            f (float): Perfusion fraction (0-1).
            Dp (float): Pseudo-diffusion coefficient (mm²/s).
    """
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
        # Weighted linear fit: weights = S(b)^2 (Veraart correction)

        high_mask = bvalues >= cutoff
        b_high = bvalues[high_mask]
        s_high = signal[high_mask]

        # Guard against zero/negative signal values
        s_high = np.maximum(s_high, 1e-8)
        log_s = np.log(s_high)

        # Veraart weights: w = S^2 (corrects for noise amplification in log-domain)
        weights_high = s_high ** 2

        # WLS: ln(S) = intercept + slope * (-b)  ⟹  slope = D
        intercept, D = _weighted_linreg(-b_high, log_s, weights_high)

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

        weights_low = r_low ** 2

        if len(b_low) >= 2:
            _, Dp = _weighted_linreg(-b_low, log_r, weights_low)
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
